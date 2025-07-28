"""TODO: Module docstring."""

# Native
from typing import Dict, Set
from operator import attrgetter
import uuid


# Third-party
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import (
    desc
)
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError

# Local
from regex_node import RegexNode
from study_config import STUDY_PROBLEMS, ELIMINATION_THRESHOLD, CONFIDENCE_THRESHOLD
from custom_types import ClassificationLabel, ClassificationType
from database import (
    db,
    User,
    ToolSession,
    CandidateExpression,
    TextDescription,
    WordClassification,
    ExpressionClassificationAgreement,
)


def get_user_from_uuid(cookie_uuid: str) -> User:
    """TODO: Write docstring."""
    if not cookie_uuid:
        return None

    try:
        cookie_uuid = uuid.UUID(cookie_uuid)
    except ValueError:
        return None

    return User.query.filter_by(user_uuid=cookie_uuid).first()


def get_incorrect_classifications(user: User) -> list[WordClassification]:
    """
    Return the latest classification for every word that the user labelled
    **incorrectly** according to STUDY_PROBLEMS[*study_problem*]["correct"].

    • “Unsure” rows are ignored.
    • Blocks whose `"correct"` is "N/A" are skipped.
    """
    out: list[WordClassification] = []

    for session in ToolSession.query.filter_by(user_uuid=user.user_uuid):
        block = STUDY_PROBLEMS[session.study_problem]
        correct_regex = block["correct"]
        if correct_regex == "N/A":
            continue  # nothing to grade against

        correct = RegexNode(correct_regex)

        words = (
            db.session.query(WordClassification.word)
            .filter_by(session_id=session.session_id)
            .distinct()
        )

        for (w,) in words:
            row = WordClassification.latest(session.session_id, w)
            if row.classification == ClassificationLabel.UNSURE:
                out.append(row)
                continue

            expected = (
                ClassificationLabel.ACCEPT
                if correct.accepts(w)
                else ClassificationLabel.REJECT
            )

            if row.classification != expected:
                out.append(row)

    return out


def mark_user_consented(user: User, consent: bool):
    """Set user's consent timestamp and flag, only if not already set."""
    if user.consent_time is None:
        user.consent_status = consent
        user.consent_time = func.now()

    db.session.commit()


def mark_user_walkthrough(user: User):
    """Set user's walkthrough timestamp, only if not already set."""
    if user.walkthrough_time is None:
        user.walkthrough_time = func.now()

    db.session.commit()


def mark_user_ack_instructions(user: User, ack: bool):
    """Set user's instruction acknowledgment time, only if not already set."""
    if user.instruction_time is None:
        user.instruction_status = ack
        user.instruction_time = func.now()

    db.session.commit()


def mark_user_screener_pass(user: User, passed: bool):
    """Set user's screener passage & timestamp, only if not already set."""
    if user.screener_passed is None:
        user.screener_time = func.now()
        user.screener_passed = passed

    db.session.commit()


def complete_session(session_id: int):
    """
    Mark the given ToolSession as completed by setting its `completed_at` field
    to the database server's current timestamp, if not already set.
    """
    session = ToolSession.query.get(session_id)
    if session and session.completed_at is None:
        session.completed_at = func.now()
        db.session.commit()


def get_first_run(user_uuid: uuid.UUID) -> bool:
    """
    Return True if the user has never classified any word across all sessions.
    False otherwise.
    """
    # Fetch session IDs (limited to 2).
    sessions = (
        db.session.query(ToolSession.session_id)
        .filter_by(user_uuid=user_uuid)
        .limit(2)
        .all()
    )

    if len(sessions) == 0:
        return True  # No sessions -> first run.

    if len(sessions) > 1:
        return False  # More than one session -> not first run.

    # Check if any WordClassification exists in single session.
    (session_id,) = sessions[0]

    has_classified = (
        db.session.query(WordClassification.classification_id)
        .filter_by(session_id=session_id)
        .limit(1)
        .first()
        is not None
    )

    return not has_classified


def get_classified_words(session_id):
    """TODO: Write docstring."""
    return {
        c.word for c in WordClassification.query.filter_by(session_id=session_id).all()
    }


def get_latest_classification(
    session_id: int, word: str
) -> "WordClassification | None":
    """
    Return the most–recent `WordClassification` row for *word* in *session_id*,
    or ``None`` if that word has never been classified in the session.
    """
    return WordClassification.latest(session_id, word)


def get_latest_classifications(session_id: int) -> Dict[str, list[str]]:
    """
    Return a dictionary with keys 'accept', 'reject', and 'unsure',
    each containing a list of words ordered by their most recent classification time
    (oldest to newest).
    """
    # Query all classifications for the session
    rows = (
        db.session.query(WordClassification)
        .filter_by(session_id=session_id)
        .order_by(WordClassification.classification_order.desc())
        .all()
    )

    latest_by_word = {}
    for row in rows:
        if row.word not in latest_by_word:
            latest_by_word[row.word] = (
                row  # keep only most recent classification per word
            )

    # Sort words by submitted_at (oldest → newest)
    latest_classifications = sorted(
        latest_by_word.values(), key=attrgetter("submitted_at")
    )

    buckets: Dict[str, list[str]] = {
        ClassificationLabel.ACCEPT: [],
        ClassificationLabel.REJECT: [],
        ClassificationLabel.UNSURE: [],
    }

    for row in latest_classifications:
        label = row.classification.value
        buckets[label].append(row.word)

    return buckets


def session_has_classifications(session_id: int) -> bool:
    """
    Return ``True`` if *any* WordClassification exists for *session_id*,
    otherwise ``False``.  Uses an efficient `EXISTS`/`LIMIT 1` query.
    """
    return (
        db.session.query(WordClassification.classification_id)
        .filter_by(session_id=session_id)
        .limit(1)
        .first()
        is not None
    )


def get_unsure_words(session_id):
    return get_latest_classifications(session_id)[ClassificationLabel.UNSURE]


def submit_classification(
    session_id: int,
    word: str,
    label: ClassificationLabel,
    ctype: ClassificationType,
    study_order: list[int],
    study_pos: int,
) -> WordClassification:
    """
    Insert a new WordClassification row, build the corresponding
    ExpressionClassificationAgreement rows, and compute expression confidence scores.
    """
    # Locate the previous classification of this word, if it exists.
    prev_wc: WordClassification | None = get_latest_classification(session_id, word)

    # Create a new classification row with incremented order.
    next_order = (prev_wc.classification_order if prev_wc else 0) + 1
    new_wc = WordClassification(
        session_id=session_id,
        word=word,
        classification=label,
        classification_type=ctype,
        classification_order=next_order,
    )
    # Add new_wc to the database for state sync without committing.
    db.session.add(new_wc)
    db.session.flush()

    # Build the new agreement rows if the new label isn't unsure.
    if label != ClassificationLabel.UNSURE:
        if prev_wc and prev_wc.classification != ClassificationLabel.UNSURE:
            # Reuse the old agreements by flipping the Boolean.
            flip = prev_wc.classification != label
            prev_agreements = ExpressionClassificationAgreement.query.filter_by(
                classification_id=prev_wc.classification_id
            ).all()

            for pa in prev_agreements:
                db.session.add(
                    ExpressionClassificationAgreement(
                        classification_id=new_wc.classification_id,
                        expression_id=pa.expression_id,
                        supports_expression=(
                            (not pa.supports_expression)
                            if flip
                            else pa.supports_expression
                        ),
                    )
                )
        else:
            # First time, or old label was UNSURE -> compute fresh agreement.
            for expr in CandidateExpression.query.filter_by(session_id=session_id):
                word_in_expr = RegexNode(expr.regex_str).accepts(word)
                supports = (
                    word_in_expr
                    if label == ClassificationLabel.ACCEPT
                    else not word_in_expr
                )
                db.session.add(
                    ExpressionClassificationAgreement(
                        classification_id=new_wc.classification_id,
                        expression_id=expr.expression_id,
                        supports_expression=supports,
                    )
                )

    # 4.  Prune all older agreement rows for this word.
    if prev_wc:
        ExpressionClassificationAgreement.query.filter(
            ExpressionClassificationAgreement.classification_id.in_(
                db.session.query(WordClassification.classification_id)
                .filter_by(session_id=session_id, word=word)
                .filter(
                    WordClassification.classification_id != new_wc.classification_id
                )
            )
        ).delete(synchronize_session=False)

    # Log expression IDs still in play.
    candidates_in_play = [
        expr for expr, in_play in get_candidate_regexes(session_id).items() if in_play
    ]
    og_problem_candidates = STUDY_PROBLEMS[study_order[study_pos]]["candidates"]

    ids_in_play = [og_problem_candidates.index(expr) for expr in candidates_in_play]

    new_wc.exp_in_play = ids_in_play

    # Commit everything (and retry on race condition).
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        # race condition → recurse once
        return submit_classification(
            session_id, word, label, ctype, study_order, study_pos
        )

    # 6.  Recompute confidence scores for the session.
    recompute_confidence_scores(session_id)

    return new_wc


def get_expression_dict(session_id: int):
    """TODO: Write Docstring."""


def get_candidate_regexes(session_id: int):
    """
    Retrieve all candidate regexes for a session, and mark them 'in play'
    unless they have at least `ELIMINATION_THRESHOLD` contradictory classifications.
    """
    # 1. Count contradictions per expression
    contradicted_counts = (
        db.session.query(
            ExpressionClassificationAgreement.expression_id,
            func.count(ExpressionClassificationAgreement.id).label("n_contradictions"),
        )
        .join(
            WordClassification,
            ExpressionClassificationAgreement.classification_id
            == WordClassification.classification_id,
        )
        .filter(
            WordClassification.session_id == session_id,
            ExpressionClassificationAgreement.supports_expression.is_(False),
        )
        .group_by(ExpressionClassificationAgreement.expression_id)
        .all()
    )
    # Build set of expression_ids with too many contradictions
    eliminated_ids = {
        expr_id for expr_id, n in contradicted_counts if n >= ELIMINATION_THRESHOLD
    }

    # 2. Fetch all expressions and mark in_play appropriately
    expressions = (
        CandidateExpression.query.filter_by(session_id=session_id)
        .order_by(CandidateExpression.expression_id)
        .all()
    )

    return {
        expr.regex_str: (expr.expression_id not in eliminated_ids)
        for expr in expressions
    }


def get_latest_description(session_id: int):
    """
    Return the most recent natural language description (nl_description)
    for a given ToolSession.

    Parameters
    ----------
    session_id : int

    Returns
    -------
    str or None
        The most recent nl_description string, or None if none exist.
    """
    latest = (
        TextDescription.query.filter_by(session_id=session_id)
        .order_by(desc(TextDescription.submitted_at))
        .first()
    )
    return latest.description if latest else None


def recompute_confidence_scores(session_id: int) -> None:
    """
    Recalculate `CandidateExpression.confidence` for _every_ expression in the
    given session.

    Definition of confidence
    ------------------------
    The number of supporting and accepting ExpressionClassificationAgreement rows
    (``supports_expression = True``) that stem from classifications made **in
    this session**.
    """
    # ── 1.  Count supporting agreements per expression_id ────────────────────
    support_counts: dict[int, int] = dict(
        db.session.query(
            ExpressionClassificationAgreement.expression_id,
            func.count(ExpressionClassificationAgreement.id),
        )
        # Join to WordClassification so we can limit by session_id
        .join(
            WordClassification,
            WordClassification.classification_id
            == ExpressionClassificationAgreement.classification_id,
        )
        .filter(
            WordClassification.session_id == session_id,
            # Confirm classification is supporting.
            ExpressionClassificationAgreement.supports_expression.is_(True),
            # Confirm classification is accept, not reject or unsure.
            WordClassification.classification == ClassificationLabel.ACCEPT,
        )
        .group_by(ExpressionClassificationAgreement.expression_id)
        .all()
    )

    # ── 2.  Apply the new counts to every candidate in the session ───────────
    for expr in CandidateExpression.query.filter_by(session_id=session_id):
        expr.confidence = support_counts.get(expr.expression_id, 0)

    db.session.commit()


def expression_status(session_id: int) -> tuple[list[CandidateExpression], bool]:
    """
    Return (survivors, all_confident) for session_id,
    where a regex is eliminated only after `ELIMINATION_THRESHOLD` contradictions.
    """
    # Count contradictions per expression
    contradicted_counts = dict(
        db.session.query(
            ExpressionClassificationAgreement.expression_id,
            func.count(ExpressionClassificationAgreement.id),
        )
        .join(WordClassification)
        .filter(
            WordClassification.session_id == session_id,
            ExpressionClassificationAgreement.supports_expression.is_(False),
        )
        .group_by(ExpressionClassificationAgreement.expression_id)
        .all()
    )

    # Identify survivors
    survivors = []
    for expr in CandidateExpression.query.filter_by(session_id=session_id):
        n_bad = contradicted_counts.get(expr.expression_id, 0)
        if n_bad < ELIMINATION_THRESHOLD:
            survivors.append(expr)

    all_confident = bool(survivors) and all(
        e.confidence >= CONFIDENCE_THRESHOLD for e in survivors
    )

    return survivors, all_confident
