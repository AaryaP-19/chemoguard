"""One structured JSON record per prediction.

Every record carries the inputs *with their units*, the values actually fed
to each estimator and in what unit, the outputs and their probabilities, the
model versions, and a timestamp. That set is chosen so a past prediction can
be reconstructed and re-checked: the recurring failure in this codebase is a
number whose scale nobody recorded, and a log that repeats that mistake
would be no evidence at all.

Refusals are logged too, with the same schema and ``outcome: "refused"``. A
prediction that did not happen because a unit was missing is the safety
mechanism working, and it is worth being able to count them.

Patient data. A record contains a person's blood counts and the id of the
user who requested them. Treat the log as clinical data: keep it on systems
approved for it, apply the retention schedule, and do not forward it to
third-party log aggregation without checking that first.

Output goes to the ``chemoguard.prediction`` logger. Configuring handlers is
the host application's job; ``configure_default_handler`` provides a plain
line-per-JSON-object stderr handler for development. JSON is emitted with
``ensure_ascii=True`` so a record containing a micro sign cannot raise
UnicodeEncodeError on a cp1252 console -- the failure mode recorded in
CLAUDE.md under "Fixes that look done and aren't".
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

LOGGER_NAME = "chemoguard.prediction"

#: Bumped when the shape of a record changes, so a reader can tell which
#: schema it is looking at.
LOG_SCHEMA_VERSION = 1

logger = logging.getLogger(LOGGER_NAME)


def configure_default_handler(level=logging.INFO) -> None:
    """Attach a stderr handler emitting one JSON object per line.

    Development convenience. A deployment should configure its own.
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


def new_prediction_id() -> str:
    """A correlation id, generated before the prediction is attempted.

    Generated up front so a refusal and a success are equally traceable.
    """
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _record(**fields) -> dict:
    record = {
        "event": "prediction",
        "schema_version": LOG_SCHEMA_VERSION,
        "timestamp": _now(),
    }
    record.update(fields)
    return record


def _emit(record: dict) -> dict:
    logger.info(json.dumps(record, ensure_ascii=True, default=str, sort_keys=True))
    return record


def log_prediction(
    request_id,
    panel,
    predictions,
    feature_records,
    model_versions,
    flags=None,
    user_id=None,
    report_id=None,
    stored_prediction_id=None,
) -> dict:
    """Record a completed prediction.

    ``panel`` is the CBCPanel the prediction was made from; its ``as_dict``
    carries every input with both its canonical and its as-reported unit.
    ``feature_records`` is the per-model mapping returned by
    ``model_contract.features_for``, which is what each estimator actually
    saw and in which unit.
    """
    return _emit(
        _record(
            outcome="predicted",
            request_id=request_id,
            user_id=user_id,
            report_id=report_id,
            stored_prediction_id=stored_prediction_id,
            inputs=panel.as_dict(),
            model_inputs=feature_records,
            model_versions=model_versions,
            outputs=predictions,
            flags=[f.as_dict() for f in (flags or [])],
        )
    )


def log_refusal(
    request_id,
    reason,
    detail,
    raw_input=None,
    user_id=None,
) -> dict:
    """Record a prediction that was refused.

    ``raw_input`` is echoed verbatim -- including the bare number that had no
    unit -- because the point of the record is to show what arrived, not a
    cleaned-up version of it.
    """
    return _emit(
        _record(
            outcome="refused",
            request_id=request_id,
            user_id=user_id,
            refusal={"reason": reason, "detail": detail},
            raw_input=raw_input,
        )
    )
