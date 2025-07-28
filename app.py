"""
This module defines the Flask application endpoints for the Text-to-Regex tool.

It provides the following routes:
    - TODO

To run the application, execute: python app.py
"""

# Native
import random
import uuid
import hashlib
import logging
import re
from functools import wraps
from collections import defaultdict
from typing import NoReturn

# Third-party
from flask import (
    Flask,
    request,
    redirect,
    jsonify,
    url_for,
    make_response,
    render_template,
    send_from_directory,
    g,
)
from flask.logging import default_handler
from sqlalchemy import select
import user_agents

# Local
from app_config import Config
from custom_types import ClassificationType, ClassificationLabel
from study_config import (
    STUDY_PROBLEMS,
    CONFIDENCE_THRESHOLD,
    UNSURE_THRESHOLD,
    STUDY_VERSION,
    SHOW_CANDIDATES,
    SHOW_LABELS,
    ELIMINATION_THRESHOLD,
)
import regex_node as regex
from regex_node import RegexNode, RegexRelationship
from database import (
    db,
    User,
    ToolSession,
    CandidateExpression,
    TextDescription,
    WordClassification,
    ExpressionClassificationAgreement,
    WordReflection,
    FollowUpResponse,
)
from db_functions import (
    get_incorrect_classifications,
    get_classified_words,
    get_first_run,
    mark_user_consented,
    mark_user_ack_instructions,
    mark_user_screener_pass,
    mark_user_walkthrough,
    session_has_classifications,
    get_latest_classifications,
    get_unsure_words,
    submit_classification,
    get_candidate_regexes,
    get_latest_description,
    get_user_from_uuid,
    complete_session,
    expression_status,
)

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

root_logger = app.logger
root_logger.addHandler(default_handler)
root_logger.addHandler((logging.getLogger("sqlalchemy")))

# When the app starts, drop and create all DB tables.
with app.app_context():
    # db.drop_all()
    db.create_all()


def user_study_complete(user: User) -> bool:
    """Return whether or not a given user has completed the study.
    Args:
        user: the User object.

    Returns:
        True if the user has completed the study, false otherwise.
    """

    complete = bool(user.study_complete)

    if complete:
        assert (
            user.study_pos >= len(STUDY_PROBLEMS)
        ) is True, "User hasn't completed all study problem blocks."

    return complete


def require_user(consent_required: bool = False, session_required: bool = False):
    """Create a decorator function for the Flask endpoints which checks that
    the user exists, and if specified, whether they've consented and/or
    have an active session.

    Args:
        consent_required: whether participation consent is required.
        session_required: whether an active tool session is required.
    Returns:
        Decorator function for Flask app endpoints.
    """

    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            cookie_uuid = request.cookies.get("user_uuid")
            user = get_user_from_uuid(cookie_uuid)

            if not user:
                return redirect(url_for("setup_user"))

            if consent_required and not user.consent_status:
                return redirect(url_for("consent"))

            g.user = user  # Save user in global 'g' object.

            if session_required:
                # Note: g.user.consent_status could be None or False.
                if not hasattr(g, "user") or not g.user.consent_status:
                    return redirect(url_for("setup_user"))

                study_order = g.user.study_order
                study_pos = g.user.study_pos

                # If the user has completed all blocks in the study:
                if study_pos >= len(study_order):
                    g.user.study_complete = True
                    # Update the DB to confirm the user has finished the study.
                    db.session.commit()
                    # Remove the session from g.
                    g.tool_session = None
                else:
                    study_prob_id = STUDY_PROBLEMS[study_order[study_pos]]["id"]
                    session = ToolSession.query.filter_by(
                        user_uuid=g.user.user_uuid, study_problem=study_prob_id
                    ).first()
                    if not session:
                        # Session for this user & problem does not exist.
                        session = setup_session(g.user)
                    else:
                        # Session for this user & problem exists.
                        g.tool_session = session

            return view_func(*args, **kwargs)

        return wrapper

    return decorator


def setup_session(user: User) -> ToolSession | None:
    """Create a ToolSession for the current study block.

    Args:
        user: Current tool user's User object.
    """
    study_order = list(user.study_order)
    study_pos = user.study_pos

    # If the user's completed the study, don't return a session.
    if study_pos >= len(study_order):
        user.study_complete = True
        db.session.commit()
        return None

    study_problem = STUDY_PROBLEMS[study_order[study_pos]]["id"]

    # Check if this session already exists in DB.
    existing_session = ToolSession.query.filter_by(
        user_uuid=user.user_uuid, study_problem=study_problem
    ).first()

    if existing_session:
        g.tool_session = existing_session
        return existing_session

    # Create a new session in DB.
    block = STUDY_PROBLEMS[study_order[study_pos]]
    tool_session = ToolSession(user_uuid=user.user_uuid, study_problem=study_problem)
    db.session.add(tool_session)
    db.session.commit()
    g.tool_session = tool_session

    # Add text description to DB.
    db.session.add(
        TextDescription(
            session_id=tool_session.session_id, description=block["description"]
        )
    )

    # Add candidate regexes to DB.
    rng = random.Random(1234)
    for rc in rng.sample(block["candidates"], len(block["candidates"])):
        db.session.add(
            CandidateExpression(
                session_id=tool_session.session_id,
                regex_str=rc,
                confidence=0,
            )
        )

    db.session.commit()

    return tool_session


def generate_words(regexes: list[str], exclude=None, seed=None) -> tuple[str, str]:
    """
    Pick two words according to the size of *regexes*.

    • 0 regexes → ("", "")
    • 1 regex   → two random words matching that regex
    • 2 regexes → two words drawn from the symmetric-difference of the pair
    • ≥3 regexes→ one word from diff(regex[0], regex[1])
                  and one word from diff(regex[1], regex[2])
    """
    # Use user classification history to perturb the seed
    if exclude:
        # Derive a hash-based seed from the sorted exclude set
        exclude_str = ",".join(sorted(exclude))
        hash_digest = hashlib.md5(exclude_str.encode()).hexdigest()
        dynamic_seed = int(hash_digest[:8], 16)  # Take 32 bits
    else:
        dynamic_seed = 1234

    # Mix in the original seed if provided
    combined_seed = (seed or 0) ^ dynamic_seed
    rng = random.Random(combined_seed)

    # Grab eliminated regexes (used when only one survivor).
    all_regexes = STUDY_PROBLEMS[g.user.study_order[g.user.study_pos]]["candidates"]
    eliminated = [r for r in all_regexes if r not in regexes]
    words = []

    if not regexes:
        return ("", "")

    if len(regexes) == 1:
        node = RegexNode(regexes[0])
        word_in = node.generate_random_words(num_words=1, exclude=exclude, seed=seed)
        word_out = ""

        rng.shuffle(eliminated)
        # TODO: With a pre-set seed, the elim regex used to generate the out word will always be the same.

        for elim in eliminated:
            out_word = regex.generate_diff_words(
                elim, regexes[0], num_words=1, exclude=exclude, seed=seed
            )
            if out_word:
                word_out = out_word[0]
                break

        if not word_in or not word_out:
            words = node.generate_random_words(num_words=2, exclude=exclude, seed=seed)
        else:
            words = [word_in[0], word_out]

        rng.shuffle(words)  # ← always shuffle, regardless of branch
        return tuple(words)

    if len(regexes) == 2:
        words = regex.generate_diff_words(
            regexes[0],
            regexes[1],
            num_words=2,
            exclude=exclude,
            seed=rng.randint(0, 2**32 - 1),
        )

        if len(words) == 2:
            return (words[0], words[1])

        # Try fallback: generate 2 words from regex[0].
        node = RegexNode(regexes[0])
        fallback = node.generate_random_words(num_words=2, exclude=exclude, seed=seed)
        if len(fallback) == 2:
            return (fallback[0], fallback[1])

        # Final fallback.
        return ("", "")

    def one_diff(a, b):
        w = regex.generate_diff_words(a, b, num_words=1, exclude=exclude, seed=seed)
        return w[0] if w else ""

    word1 = one_diff(regexes[0], regexes[1])
    exclude.add(word1)
    word2 = one_diff(regexes[1], regexes[2])

    return (word1, word2)


def generate_words_gt(
    ground_truth: str, regexes: list[str], exclude=None, seed=None
) -> tuple[str, str]:
    """
    Pick two words according to the size of *regexes*.

    • 0 regexes → ("", "")
    • 1 regex   → two random words matching that regex
    • 2 regexes → two words drawn from the symmetric-difference of the pair
    • ≥3 regexes→ one word from diff(regex[0], regex[1])
                  and one word from diff(regex[1], regex[2])
    """
    # Use user classification history to perturb the seed
    if exclude:
        # Derive a hash-based seed from the sorted exclude set
        exclude_str = ",".join(sorted(exclude))
        hash_digest = hashlib.md5(exclude_str.encode()).hexdigest()
        dynamic_seed = int(hash_digest[:8], 16)  # Take 32 bits
    else:
        dynamic_seed = 1234

    # Mix in the original seed if provided
    combined_seed = (seed or 0) ^ dynamic_seed
    rng = random.Random(combined_seed)

    # Grab eliminated regexes (used when only one survivor).
    all_regexes = STUDY_PROBLEMS[g.user.study_order[g.user.study_pos]]["candidates"]
    eliminated = [r for r in all_regexes if r not in regexes]
    words = []

    if not regexes:
        return ("", "")

    if len(regexes) == 1:
        dist_node = RegexNode(regexes[0])
        gt_node = RegexNode(ground_truth)

        if dist_node.relationship(gt_node) is RegexRelationship.SUBSET:
            dist_words = dist_node.generate_random_words(
                num_words=1, exclude=exclude, seed=seed
            )
            word_1 = dist_words[0] if dist_words else None

            gt_words = regex.generate_diff_words(
                regexes[0], ground_truth, num_words=1, exclude=exclude, seed=seed
            )
            word_2 = gt_words[0] if gt_words else None

            if not word_1 or not word_2:
                words = dist_node.generate_random_words(
                    num_words=2, exclude=exclude, seed=seed
                )
            else:
                words = [word_1, word_2]

        else:
            gt_words = regex.generate_diff_words(
                regexes[0], ground_truth, num_words=2, exclude=exclude, seed=seed
            )

            if len(gt_words) != 2:
                words = dist_node.generate_random_words(
                    num_words=2, exclude=exclude, seed=seed
                )
            else:
                words = [gt_words[0], gt_words[1]]

        rng.shuffle(words)

        return tuple(words)

    if len(regexes) == 2:
        words = regex.generate_diff_words(
            regexes[0],
            regexes[1],
            num_words=2,
            exclude=exclude,
            seed=rng.randint(0, 2**32 - 1),
        )

        if len(words) == 2:
            return (words[0], words[1])

        # Try fallback: generate 2 words from regex[0].
        node = RegexNode(regexes[0])
        fallback = node.generate_random_words(num_words=2, exclude=exclude, seed=seed)
        if len(fallback) == 2:
            return (fallback[0], fallback[1])

        # Final fallback.
        return ("", "")

    def one_diff(a, b):
        w = regex.generate_diff_words(a, b, num_words=1, exclude=exclude, seed=seed)
        return w[0] if w else ""

    word1 = one_diff(regexes[0], regexes[1])
    word2 = one_diff(regexes[1], regexes[2])

    return (word1, word2)

def ready_for_review(session_id: int) -> bool:
    """Returns True iff no survivors, all survivors confident,
    or #(unsure) >= threshold."""
    survivors, all_confident = expression_status(session_id)

    if not survivors:
        return {"ready": True, "reason": "No Survivors"}
    elif all_confident and len(survivors) == 1:
        return {"ready": True, "reason": "All Confident"}
    elif len(get_unsure_words(session_id)) >= UNSURE_THRESHOLD:
        return {"ready": True, "reason": "Unsure"}
    else:
        return {"ready": False, "reason": None}


def get_candidate_list_status(session_id: int) -> list:
    regex_status_with_ids = []

    candidate_regexes = get_candidate_regexes(session_id)

    for idx, (expr, in_play) in enumerate(candidate_regexes.items(), start=1):
        rid = f"R{idx}"  # R1, R2, ...
        regex_status_with_ids.append({"id": rid, "expr": expr, "in_play": in_play})

    return regex_status_with_ids


def generate_study_order(seed=None) -> list[int]:
    # Problem ID 2 is dates, should go in position 2 or 3.
    # TODO: Generalize this for more problems than the current set.
    id_list = [0, 1, 3]
    random.shuffle(id_list)
    choice = random.choice([1, 2])
    id_list.insert(choice, 2)
    return id_list


@app.route("/setup", methods=["GET", "POST"])
def setup_user():
    # On GET, look for a query-param; on POST, read the form
    if request.method == "POST":
        prolific_id = request.form.get("prolific_id", "").strip()
    else:
        prolific_id = request.args.get("PROLIFIC_PID", "").strip()

    # If still empty, render the PID entry form
    if not prolific_id or not re.fullmatch(r"[0-9a-f]{24}", prolific_id):
        error = None
        if request.method == "POST" and prolific_id:
            error = "Please enter a valid 24-character Prolific ID."
        return render_template("enter_pid.html", prolific_id=prolific_id, error=error)

    # Check if the user already exists.
    user = User.query.filter_by(prolific_id=prolific_id).first()

    if not user:
        user_uuid = uuid.uuid4()

        ua_string = request.headers.get("User-Agent")
        user_agent = user_agents.parse(ua_string)

        user = User(
            user_uuid=user_uuid,
            prolific_id=prolific_id,
            study_order=generate_study_order(),
            study_pos=0,
            user_agent=str(user_agent),
        )

        db.session.add(user)
        db.session.commit()

        if user_agent.is_mobile or user_agent.is_tablet:
            return redirect(
                "https://app.prolific.com/submissions/complete?cc=C8Z294SH", code=302
            )
            # return render_template("mobile.html")
    else:
        # ─── Existing user logic ───
        if not user.consent_status:
            return redirect(url_for("consent"))
        if user.screener_passed is None:
            return redirect(url_for("screener"))
        if user.screener_passed is False:
            return redirect(
                "https://app.prolific.com/submissions/complete?cc=C1I0FSLH", code=302
            )
            # return render_template("screener_disqualified.html")
        if user.instruction_status is False:
            return redirect(
                "https://app.prolific.com/submissions/complete?cc=CXJY2BLY", code=302
            )
            # return redirect(url_for("no_instructions_ack"))
        if user.study_complete:
            return redirect(
                "https://app.prolific.com/submissions/complete?cc=C1GXEDZT", code=302
            )
            # return render_template("complete.html")

    response = make_response(redirect(url_for("consent")))
    response.set_cookie("user_uuid", str(user.user_uuid), max_age=60 * 60 * 24 * 7)
    return response


@app.route("/followup_with_list.html")
def folowup_with_list():
    """Return followup WITH list page."""
    return render_template("followup_with_list.html")


@app.route("/followup_without_list.html")
def folowup_without_list():
    """Return followup WITHOUT list page."""
    return render_template("followup_without_list.html")


@app.route("/interface.png")
def interface_image():
    """Serve the interface image."""
    # app.static_folder is "static" by default
    return send_from_directory(app.static_folder, "interface.png")


@app.route("/consent", methods=["POST", "GET"])
@require_user(consent_required=False, session_required=False)
def consent():
    """TODO: Write docstring."""
    if request.method == "POST":
        consent_response = request.form.get("consent")

        if consent_response not in ["true", "false"]:
            return "Invalid consent option", 400

        mark_user_consented(g.user, consent_response == "true")
        db.session.commit()

        if g.user.consent_status:
            return redirect(url_for("screener"))
        else:
            return redirect(
                "https://app.prolific.com/submissions/complete?cc=CFN2Z4FW", code=302
            )
            # return render_template("no_consent.html")

    # GET Request:
    return render_template("consent.html")


@app.route("/screener", methods=["GET", "POST"])
@require_user(consent_required=True, session_required=False)
def screener():
    """
    Show screener questions and store responses.
    If passed, continue to /instructions. Otherwise, disqualify.
    """
    if request.method == "POST":
        prog_screener_answer = request.form.get("q1", "").strip().lower()
        regex_screener_answer = request.form.get("q2", "").strip().lower()

        # Store responses.
        g.user.programmer_screener = prog_screener_answer
        g.user.regex_screener = regex_screener_answer

        # Define pass condition
        passed = (prog_screener_answer == "yes") and (regex_screener_answer != "none")
        mark_user_screener_pass(g.user, passed)
        db.session.commit()

        if passed:
            return redirect(url_for("instructions"))
        else:
            return redirect(
                "https://app.prolific.com/submissions/complete?cc=C1I0FSLH", code=302
            )
            # return render_template("screener_disqualified.html")

    return render_template("screener.html")


@app.route("/instructions", methods=["POST", "GET"])
@require_user(consent_required=True, session_required=False)
def instructions():
    """TODO: Write docstring."""
    if request.method == "POST":
        understand_response = request.form.get("understand")

        if understand_response not in ["true", "false"]:
            return "Invalid consent option", 400

        mark_user_ack_instructions(g.user, understand_response == "true")
        db.session.commit()

        if g.user.instruction_status:
            return redirect(url_for("study"))
        else:
            return redirect(
                "https://app.prolific.com/submissions/complete?cc=CXJY2BLY", code=302
            )
            # return render_template("no_instruction_ack.html")

    # GET Request:
    return render_template("instructions.html")


def get_ground_truth(study_problem_id: int) -> str:
    return


@app.route("/study", methods=["GET"])
@require_user(consent_required=True, session_required=True)
def study():
    user = g.user

    from_review = request.args.get("from_review") == "true"

    # Case: study is over
    if user_study_complete(user):
        return redirect(url_for("reflection"))

    session_id = g.tool_session.session_id

    # If session is ready for review.
    rfr = ready_for_review(session_id)
    if rfr["ready"]:
        return redirect(url_for("review", reason=rfr["reason"]))

    candidate_regexes = get_candidate_regexes(session_id)

    candidates_in_play = [
        expr for expr, in_play in candidate_regexes.items() if in_play
    ]

    regex_with_ids = []
    expr_to_id = {}

    for idx, (expr, in_play) in enumerate(candidate_regexes.items(), start=1):
        rid = f"R{idx}"  # R1, R2, ...
        regex_with_ids.append({"id": rid, "expr": expr, "in_play": in_play})
        expr_to_id[expr] = rid

    classified_words = get_classified_words(session_id)
    text_description = get_latest_description(session_id)

    # Check if session is a ground truth mode problem.
    gt_mode = g.tool_session.study_problem == 2

    word1 = word2 = None

    if gt_mode:

        def get_correct_regex(session):
            for block in STUDY_PROBLEMS:
                if block["id"] == session.study_problem:
                    return block["correct"]
            return None

        ground_truth = get_correct_regex(g.tool_session)

        word1, word2 = generate_words_gt(
            ground_truth, candidates_in_play, exclude=classified_words, seed=1234
        )

    else:
        word1, word2 = generate_words(
            candidates_in_play, exclude=classified_words, seed=1234
        )

    def matching_ids(word):
        return [
            expr_to_id[expr] for expr in candidate_regexes if re.fullmatch(expr, word)
        ]

    word_matches = {
        word1: matching_ids(word1),
        word2: matching_ids(word2),
    }

    first_in_session = not session_has_classifications(session_id)

    first_run = False

    if first_in_session:
        first_run = get_first_run(user.user_uuid)

    accepted_words = rejected_words = unsure_words = set()
    if not first_in_session:
        classified_words = get_latest_classifications(session_id)
        accepted_words = classified_words[ClassificationLabel.ACCEPT]
        rejected_words = classified_words[ClassificationLabel.REJECT]
        unsure_words = classified_words[ClassificationLabel.UNSURE]

        for word in list(accepted_words + rejected_words + unsure_words):
            word_matches[word] = matching_ids(word)

    return render_template(
        "study.html",
        text_description=text_description,
        word1=word1,
        word2=word2,
        first_run=first_run,
        session_id=session_id,
        first_in_session=first_in_session,
        accepted_words=accepted_words,
        rejected_words=rejected_words,
        unsure_words=unsure_words,
        from_review=from_review,
        regex_with_ids=regex_with_ids,
        word_matches=word_matches,
        show_candidates=SHOW_CANDIDATES,
        show_labels=SHOW_LABELS,
    )


from datetime import datetime


@app.route("/mark_walkthrough_complete", methods=["POST"])
@require_user(consent_required=True, session_required=True)
def mark_walkthrough_complete():
    mark_user_walkthrough(g.user)
    return "OK", 200


@app.route("/classify", methods=["POST"])
@require_user(consent_required=True, session_required=True)
def classify():
    """
    Handle a single word-classification.

    Redirect to /review **only** when BOTH:
        1. this was the second word on the page  (submission_number == 2)
        2. every surviving candidate has ≥ CONFIDENCE_THRESHOLD *or* none survive
    Otherwise return 204 so the UI can keep working in the current screen.
    """
    data = request.get_json()
    word = data.get("word")
    label = data.get("classification")
    response_session_id = int(data.get("session_id"))
    submission_number = int(data.get("submission_number", 0))  # defensive default

    if not word or label not in ["accept", "reject", "unsure"]:
        return "Invalid data", 400

    if g.user.study_complete or not hasattr(g, "tool_session"):
        return redirect(
            "https://app.prolific.com/submissions/complete?cc=C1GXEDZT", code=302
        )
        # return render_template("complete.html")

    if response_session_id != g.tool_session.session_id:
        return "Invalid session", 400


    # Look up the very last classification for this word + session.
    last = WordClassification.latest(response_session_id, word)

    # If they just clicked the same label twice, otherwise submit.
    # TODO: Add this check for other classification types.
    if not (
        last
        and last.classification_type == ClassificationType.CLASSIFY
        and last.classification.value == label
    ):
        # Add the WordClassification and compute agreement updates.
        submit_classification(
            response_session_id,
            word,
            label,
            ClassificationType.CLASSIFY,
            g.user.study_order,
            g.user.study_pos,
        )

    rfr = ready_for_review(response_session_id)
    if submission_number == 2 and rfr["ready"]:
        return jsonify({"redirect": url_for("review", reason=rfr["reason"])})

    regex_data = get_candidate_list_status(response_session_id)

    # Otherwise stay on /study and calculate new word suggestions.
    return jsonify({"regexes": regex_data})


@app.route("/reclassify", methods=["POST"])
@require_user(consent_required=True, session_required=True)
def reclassify():
    """
    Re-classify a word from the review screen.

    Logic
    -----
    • Insert a fresh WordClassification row (history is preserved).
    • Drop *all* previous ExpressionClassificationAgreement rows for this word,
      then rebuild them **except** when the new label is *unsure*
      (unsure ≙ “no evidence”, so we record nothing).
    • Recompute each candidate’s confidence score.
    • Decide next page:
        – /review  → if zero survivors OR every survivor is confident
        – /study   → otherwise (survivors still need work)
    Returns
    -------
    200 JSON {redirect: "/study" | "/review"}    – front-end will navigate.
    400 on bad input.
    """
    data = request.get_json()
    word = data.get("word")
    new_label = data.get("new_classification")
    response_session_id = int(data.get("session_id"))
    classification_type = data.get("classification_type")

    if new_label not in ["accept", "reject", "unsure"]:
        print("invalid label")
        return "Invalid classification", 400

    if classification_type not in ["reclassify", "review"]:
        print("invalid class type")
        return "Invalid classification", 400

    if g.user.study_complete or not hasattr(g, "tool_session"):
        return redirect(
            "https://app.prolific.com/submissions/complete?cc=C1GXEDZT", code=302
        )
        # return render_template("complete.html")

    if response_session_id != g.tool_session.session_id:
        print("bad session")
        return "Invalid session", 400

    # Submit new classification.
    submit_classification(
        response_session_id,
        word,
        new_label,
        classification_type,
        g.user.study_order,
        g.user.study_pos,
    )

    regex_data = get_candidate_list_status(response_session_id)

    return jsonify({"regexes": regex_data})


@app.route("/review", methods=["GET", "POST"])
@require_user(consent_required=True, session_required=True)
def review():
    """
    GET   → show a summary of the latest-state buckets so the user can
            re-classify any mistakes.

    POST  → user clicks “Approve & Continue”.

            • If `ready_for_review(session_id)` is True
                  ⇢ advance to the next study block (sets up the new session)

            • Otherwise
                  ⇢ return to /study so the participant can keep supplying
                    evidence for the surviving expressions.
    """
    session_id = g.tool_session.session_id

    REVIEW_REASON_TEXT = {
        "Unsure": "You've reached the maximum number of <strong>Unsure</strong> classifications for this problem.",
        "All Confident": "At least one candidate regex supports all of your classifications so far.",
        "No Survivors": "Every candidate regex has been eliminated.",
    }

    reason_key = request.args.get("reason")
    reason_msg = REVIEW_REASON_TEXT.get(reason_key)

    # --------- GET : render review page ------------------------------------
    if request.method == "GET":
        text_description = get_latest_description(session_id)

        classified_words = get_latest_classifications(session_id)
        accepted_words = classified_words[ClassificationLabel.ACCEPT]
        rejected_words = classified_words[ClassificationLabel.REJECT]
        unsure_words = classified_words[ClassificationLabel.UNSURE]

        return render_template(
            "review.html",
            session_id=session_id,
            text_description=text_description,
            reason_message=reason_msg,
            accepted_words=accepted_words,
            rejected_words=rejected_words,
            unsure_words=unsure_words,
        )

    # --------- POST : "Approve & Continue" ---------------------------------
    if ready_for_review(session_id)["ready"]:
        g.user.study_pos += 1  # move to next block
        complete_session(session_id)
        db.session.commit()
        g.pop("tool_session", None)  # Clear the cached session.
        return redirect(url_for("study"))

    # still work to do → loop back to study
    return redirect(url_for("study", from_review="true"))


@app.route("/reflection", methods=["GET", "POST"])
@require_user(consent_required=True, session_required=False)
def reflection():
    # ───────────────── POST (unchanged) ───────────────────────────
    if request.method == "POST":
        for key, text in request.form.items():
            if text.strip():
                try:
                    cls_id = int(key)
                    db.session.merge(
                        WordReflection(
                            user_uuid=g.user.user_uuid,
                            classification_id=cls_id,
                            explanation=text.strip(),
                        )
                    )
                except ValueError:
                    pass
        g.user.study_complete = True
        db.session.commit()
        return redirect(url_for("followup"))

    # ───────────────── GET ────────────────────────────────────────
    wrong_rows = get_incorrect_classifications(g.user)
    if not wrong_rows:
        g.user.study_complete = True
        db.session.commit()
        return redirect(url_for("followup"))

    # ---- 1. map study_problem → {description, correct-regex} -----
    meta = {
        blk["id"]: {
            "description": blk["description"],
            "correct": RegexNode(blk["correct"]),
        }
        for blk in STUDY_PROBLEMS
    }

    # ---- 2. cache  session_id → study_problem  -------------------
    session_ids = {row.session_id for row in wrong_rows}

    if session_ids:
        rows = db.session.execute(
            select(ToolSession.session_id, ToolSession.study_problem).where(
                ToolSession.session_id.in_(session_ids)
            )
        )
        sess_map = {sid: sp for sid, sp in rows}  # ← build dict row-by-row
    else:
        sess_map = {}

    # ---- 3. build grouped data -----------------------------------
    blocks = defaultdict(lambda: {"description": "", "wrong": []})

    for row in wrong_rows:
        sp = sess_map[row.session_id]  # ← FIX
        correct_regex = meta[sp]["correct"]
        expected = "accept" if correct_regex.accepts(row.word) else "reject"

        blk = blocks[sp]
        blk["description"] = meta[sp]["description"]
        blk["wrong"].append(
            {
                "classification_id": row.classification_id,
                "word": row.word,
                "response": row.classification.value,
                "expected": expected,
            }
        )

    # ---- 4. preserve user’s original study order -----------------
    ordered_blocks = [
        {"description": blocks[sp]["description"], "wrong": blocks[sp]["wrong"]}
        for sp in g.user.study_order
        if sp in blocks
    ]

    return render_template("reflection.html", blocks=ordered_blocks)


@app.route("/followup", methods=["GET", "POST"])
@require_user(consent_required=True, session_required=False)
def followup():
    if request.method == "POST":
        row = FollowUpResponse.query.get(g.user.user_uuid) or FollowUpResponse(
            user_uuid=g.user.user_uuid
        )

        # Q1 – Helpfulness of candidate list
        row.q1_response = request.form.get("q1", "").strip() or None

        # Q2 – Per-task confidence
        row.q2_ab = request.form.get("q2_ab", "").strip() or None
        row.q2_dates = request.form.get("q2_dates", "").strip() or None
        row.q2_times = request.form.get("q2_times", "").strip() or None
        row.q2_variables = request.form.get("q2_variables", "").strip() or None

        # Q3 – Free response: technical issues
        row.q3_response = request.form.get("q3", "").strip() or None

        db.session.merge(row)

        # Mark study complete
        g.user.study_complete = True
        db.session.commit()

        return redirect(
            "https://app.prolific.com/submissions/complete?cc=C1GXEDZT", code=302
        )

    if SHOW_CANDIDATES:
        return render_template("followup_with_list.html")
    return render_template("followup_without_list.html")


@app.route("/debug/user_timeline")
def user_timeline():

    def get_correct_regex(session):
        for block in STUDY_PROBLEMS:
            if block["id"] == session.study_problem:
                return block["correct"]
        return None

    def get_session_candidates(session):
        for block in STUDY_PROBLEMS:
            if block["id"] == session.study_problem:
                return block["candidates"]
        return None

    output = []

    # TODO: Define these parameters.
    study_parameters = {
        "Confidence Threshold": CONFIDENCE_THRESHOLD,
        "Eliminiation Threshold": ELIMINATION_THRESHOLD,
        "Unsure Threshold": UNSURE_THRESHOLD,
        "Problem Data": STUDY_PROBLEMS,
        "Study Version": STUDY_VERSION,
        "Show Candidates": SHOW_CANDIDATES,
        "Show Labels": SHOW_LABELS,
    }

    output.append({"Study Parameters": study_parameters})

    users = User.query.all()
    for user in users:
        user_data = {"user_data": user.to_dict(), "sessions": [], "reflections": []}

        # Get sessions ordered by creation time
        sessions = (
            ToolSession.query.filter_by(user_uuid=user.user_uuid)
            .order_by(ToolSession.created_at)
            .all()
        )

        for session in sessions:
            classifications = (
                WordClassification.query.filter_by(session_id=session.session_id)
                .order_by(WordClassification.submitted_at)
                .all()
            )

            expressions = CandidateExpression.query.filter_by(
                session_id=session.session_id
            ).all()

            # Generate list of [confidence, expression regex]
            expr_and_confs = [[expr.confidence, expr.regex_str] for expr in expressions]

            session_data = {
                "session_id": session.session_id,
                "study_problem": session.study_problem,
                "created_at": str(session.created_at),
                "completed_at": str(session.completed_at),
                "candidate_list": expr_and_confs,
                "actions": [],
            }

            # Track previous exp_in_play
            previous_exp_in_play = set([0, 1, 2, 3])

            for classification in classifications:
                correct_regex_str = get_correct_regex(session)

                correct_regex_node = RegexNode(correct_regex_str)

                expected = (
                    "accept"
                    if correct_regex_node.accepts(classification.word)
                    else "reject"
                )
                actual = classification.classification.value

                # Only consider definite classifications
                if actual == "unsure":
                    correct = None
                elif expected == actual:
                    correct = True
                elif expected != actual:
                    correct = False
                else:
                    correct = "error"

                current_exp_in_play = set(classification.exp_in_play or [])
                added = sorted(current_exp_in_play - previous_exp_in_play)
                removed = sorted(previous_exp_in_play - current_exp_in_play)

                session_data["actions"].append(
                    {
                        "time": classification.submitted_at.isoformat(),
                        "c_id": classification.classification_id,
                        "word": classification.word,
                        "label": classification.classification.value,
                        "c_type": classification.classification_type,
                        "exp_in_play": classification.exp_in_play,
                        "added": added,
                        "removed": removed,
                        "correct": correct,
                    }
                )

                previous_exp_in_play = current_exp_in_play

            user_data["sessions"].append(session_data)

        reflections = (
            WordReflection.query.filter_by(user_uuid=user.user_uuid)
            .order_by(WordReflection.submitted_at)
            .all()
        )

        for ref in reflections:
            classification = ref.classification
            session = ToolSession.query.get(classification.session_id)

            correct_regex_str = get_correct_regex(session)
            correct_regex_node = RegexNode(correct_regex_str)

            expected = (
                "accept"
                if correct_regex_node.accepts(classification.word)
                else "reject"
            )
            actual = classification.classification.value

            user_data["reflections"].append(
                {
                    "time": ref.submitted_at.isoformat(),
                    "reflection_id": ref.id,
                    "classification_id": ref.classification_id,
                    "explanation": ref.explanation,
                    "word": classification.word,
                    "expected": expected,
                    "actual": actual,
                }
            )

        fup = FollowUpResponse.query.get(user.user_uuid)
        if fup:
            user_data["followup"] = {
                "q1_response": fup.q1_response,
                "q2_ab": fup.q2_ab,
                "q2_dates": fup.q2_dates,
                "q2_times": fup.q2_times,
                "q2_variables": fup.q2_variables,
                "q3_response": fup.q3_response,
                "submitted_at": fup.submitted_at.isoformat(),
            }
        else:
            user_data["followup"] = None

        output.append(user_data)  # ← keep this as it was

    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)

# TODO: Rework current style.
# TODO: Make colorblind friendly.
