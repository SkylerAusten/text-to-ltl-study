"""TODO: Module docstring."""

# Native
import uuid
import json

# Third-party
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (
    desc,
    ForeignKey,
    Integer,
    String,
    Boolean,
    DateTime,
    Enum,
    Text,
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.types import TypeDecorator, JSON
from sqlalchemy.ext.hybrid import hybrid_property

# Local
from custom_types import ClassificationLabel, ClassificationType

db = SQLAlchemy()


# Type decorator to store UUIDs as CHAR(36) in MySQL.
class UUID(TypeDecorator):
    """TODO: Write Docstring."""

    impl = CHAR
    cache_ok = True

    @property
    def python_type(self):
        return uuid.UUID

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(CHAR(36))  # Store as string

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            return str(uuid.UUID(value))
        return str(value)

    def process_result_value(self, value, dialect):
        return uuid.UUID(value) if value else None

    def process_literal_param(self, value, dialect):
        return f"'{str(value)}'" if value is not None else "NULL"


# Type decorator to store Python sets as JSON in MySQL.
class ListJSON(TypeDecorator):
    """
    Stores a Python list as a JSON column.
    Ensures list structure is preserved.
    """

    impl = JSON
    cache_ok = True

    @property
    def python_type(self):
        return list

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("ListJSON must be given a list.")
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        try:
            return json.loads(value)
        except Exception:
            return []

    def process_literal_param(self, value, dialect):
        return f"'{json.dumps(value)}'" if value else "NULL"


class User(db.Model):
    """TODO: Write docstring."""

    user_uuid = db.Column(UUID(), primary_key=True)
    prolific_id = db.Column(String(50), unique=True, nullable=False)
    user_agent = db.Column(String(255), nullable=True)
    programmer_screener = db.Column(String(5), nullable=True)
    regex_screener = db.Column(String(10), nullable=True)
    screener_passed = db.Column(Boolean, nullable=True)
    consent_status = db.Column(Boolean, nullable=True)
    consent_time = db.Column(DateTime, nullable=True)
    screener_time = db.Column(DateTime, nullable=True)
    instruction_status = db.Column(Boolean, nullable=True)
    instruction_time = db.Column(DateTime, nullable=True)
    walkthrough_time = db.Column(
        DateTime, nullable=True
    )  # Walkthrough completion time.
    created_at = db.Column(DateTime, server_default=func.sysdate(), nullable=False)
    study_order = db.Column(ListJSON(), default=list, nullable=True)
    study_pos = db.Column(Integer, default=0, nullable=True)
    study_complete = db.Column(Boolean, default=False, nullable=False)

    def to_dict(self):
        """TODO: Write docstring."""
        return {
            "user_uuid": str(self.user_uuid),  # Convert UUID to string for JSON
            "prolific_id": self.prolific_id,
            "user_agent": self.user_agent,
            "programmer_screener": self.programmer_screener,
            "regex_screener": self.regex_screener,
            "screener_passed": self.screener_passed,
            "consent_status": self.consent_status,
            "consent_time": str(self.consent_time),
            "instruction_status": self.instruction_status,
            "instruction_time": str(self.instruction_time),
            "walkthrough_time": str(self.walkthrough_time),
            "created_at": str(self.created_at),
            "study_order": list(self.study_order),
            "study_pos": self.study_pos,
            "study_complete": self.study_complete,
        }


class ToolSession(db.Model):
    """TODO: Write docstring."""

    session_id = db.Column(Integer, primary_key=True, autoincrement=True)
    user_uuid = db.Column(UUID(), ForeignKey("user.user_uuid"), nullable=False)
    created_at = db.Column(DateTime, server_default=func.sysdate(), nullable=False)
    study_problem = db.Column(Integer, nullable=True)
    completed_at = db.Column(DateTime, nullable=True)

    descriptions = db.relationship(
        "TextDescription",
        backref="tool_session",
        lazy="dynamic",  # So we can filter/sort
        cascade="all, delete-orphan",
    )

    @hybrid_property
    def latest_description(self):
        return self.descriptions.order_by(desc(TextDescription.submitted_at)).first()

    def to_dict(self):
        """TODO: Write docstring."""
        return {
            "session_id": self.session_id,
            "user_uuid": str(self.user_uuid),
            "created_at": str(self.created_at),
            "study_problem": self.study_problem,
            "completed_at": str(self.completed_at),
            "description_versions": [d.to_dict() for d in self.descriptions.all()],
            "latest_description": (
                self.latest_description.to_dict() if self.latest_description else None
            ),
        }


class TextDescription(db.Model):
    """Stores a natural language description and the time it was recorded."""

    description_id = db.Column(Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(
        Integer, ForeignKey("tool_session.session_id"), nullable=False
    )
    submitted_at = db.Column(
        DateTime, server_default=func.sysdate(), nullable=False, index=True
    )
    description = db.Column(Text, default=None, nullable=True)

    def to_dict(self):
        """Convert model to dictionary for serialization."""
        return {
            "description_id": self.description_id,
            "session_id": self.session_id,
            "submitted_at": self.submitted_at.isoformat(),
            "description": self.description,
        }


class CandidateExpression(db.Model):
    """TODO: Write docstring."""

    expression_id = db.Column(Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(
        Integer, ForeignKey("tool_session.session_id"), nullable=False
    )
    generated_at = db.Column(DateTime, server_default=func.sysdate(), nullable=False)
    regex_str = db.Column(Text, nullable=False)
    confidence = db.Column(Integer, default=0, nullable=False)

    # classification_links = db.relationship(
    #     "ExpressionClassificationAgreement",
    #     backref="expression",
    #     cascade="all, delete-orphan",
    #     lazy="dynamic",
    # )

    def to_dict(self):
        """TODO: Write docstring."""
        return {
            "expression_id": self.expression_id,
            "session_id": self.session_id,
            "generated_at": str(self.generated_at),
            "regex_str": self.regex_str,
            "confidence": self.confidence,
        }


class WordClassification(db.Model):
    """TODO: Write docstring."""

    classification_id = db.Column(Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(
        Integer, ForeignKey("tool_session.session_id"), nullable=False
    )
    word = db.Column(String(100), nullable=False)
    # How the user classified the word:
    classification = db.Column(Enum(ClassificationLabel), nullable=False)

    # Whether it was a classification, reclassification, or review:
    classification_type = db.Column(Enum(ClassificationType), nullable=False)

    # Tracker for home many times the word has been classified/reclassified.
    classification_order = db.Column(Integer, nullable=False)

    # Expressions IDs still in play.
    exp_in_play = db.Column(ListJSON(), nullable=True)

    # Submission time.
    submitted_at = db.Column(DateTime, server_default=func.sysdate(), nullable=False)

    @classmethod
    def latest(cls, session_id, word):
        """Return the most-recent classification for a word in a session."""
        return (
            db.session.query(cls)
            .filter_by(session_id=session_id, word=word)
            .order_by(cls.classification_order.desc())
            .first()
        )

    def to_dict(self):
        """TODO: Write docstring."""
        return {
            "classification_id": self.classification_id,
            "session_id": self.session_id,
            "word": self.word,
            "classification": self.classification,
            "classification_type": self.classification_type.value,
            "classification_order": self.classification_order,
            "submitted_at": self.submitted_at,
        }


class ExpressionClassificationAgreement(db.Model):
    """
    Link a single WordClassification to a CandidateExpression, recording
    whether that classification *supports* the expression.
    """

    __tablename__ = "expression_classification_agreement"

    id = db.Column(Integer, primary_key=True, autoincrement=True)

    classification_id = db.Column(
        Integer,
        ForeignKey("word_classification.classification_id"),
        nullable=False,
    )
    expression_id = db.Column(
        Integer,
        ForeignKey("candidate_expression.expression_id"),
        nullable=False,
    )
    supports_expression = db.Column(Boolean, nullable=False)

    # Fast lookup: one agreement per (classification, expression).
    __table_args__ = (
        db.UniqueConstraint("classification_id", "expression_id", name="uq_class_expr"),
    )

    # -------- convenience relationships (optional) -------------------------
    classification = db.relationship("WordClassification", lazy="joined")
    expression = db.relationship("CandidateExpression", lazy="joined")

    def to_dict(self):
        return {
            "id": self.id,
            "classification_id": self.classification_id,
            "expression_id": self.expression_id,
            "supports_expression": self.supports_expression,
        }


class WordReflection(db.Model):
    """
    One free-text explanation attached to a (wrong) WordClassification row.
    """

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_uuid = db.Column(
        UUID(),
        ForeignKey("user.user_uuid"),
        nullable=False,
    )
    classification_id = db.Column(
        Integer,
        ForeignKey("word_classification.classification_id"),
        nullable=False,
        unique=True,  # one reflection per classification
    )
    explanation = db.Column(db.Text, nullable=False)

    submitted_at = db.Column(DateTime, server_default=func.sysdate(), nullable=False)

    classification = db.relationship("WordClassification", lazy="joined")

    def to_dict(self):
        return {
            "id": self.id,
            "user_uuid": str(self.user_uuid),
            "classification_id": self.classification_id,
            "explanation": self.explanation,
            "submitted_at": str(self.submitted_at),
        }


class FollowUpResponse(db.Model):
    """Store post-study follow-up answers (one row per user)."""

    user_uuid = db.Column(UUID(), ForeignKey("user.user_uuid"), primary_key=True)

    # Q1 – Helpfulness of candidate regex list
    q1_response = db.Column(String(20), nullable=True)

    # Q2 – Per-task confidence in picking a regex without testing
    q2_ab = db.Column(String(10), nullable=True)
    q2_dates = db.Column(String(10), nullable=True)
    q2_times = db.Column(String(10), nullable=True)
    q2_variables = db.Column(String(10), nullable=True)

    # Q3 – Technical issues (free text)
    q3_response = db.Column(Text, nullable=True)

    submitted_at = db.Column(DateTime, server_default=func.now(), nullable=False)

    user = db.relationship("User", backref="followup", lazy="joined")

    def to_dict(self):
        return {
            "user_uuid": str(self.user_uuid),
            "q1_response": self.q1_response,
            "q2_ab": self.q2_ab,
            "q2_dates": self.q2_dates,
            "q2_times": self.q2_times,
            "q2_variables": self.q2_variables,
            "q3_response": self.q3_response,
            "submitted_at": self.submitted_at.isoformat(),
        }
