"""The backend-to-model event translation in scripts/export_interactions.py.

This mapping is the seam between two teams' vocabularies, and it fails quietly:
a wrong field name produces `event_type: null` for every row, the exporter still
writes a file, and retraining runs on nothing without raising. These tests pin
the shape of a real record from the live table so that failure mode cannot come
back unnoticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
SCRIPTS = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from export_interactions import _as_int, translate_event  # noqa: E402
from src.models.interaction_weights import SUPPORTED_EVENT_TYPES  # noqa: E402

THRESHOLD = 0.95


def backend_record(**overrides):
    """A record shaped like the ones the live UserInteractions table holds."""
    record = {
        "user_id": "99",
        "movie_id": "602",
        "interaction_type": "rating",
        "interaction_action": "set",
        "interaction_value": 3,
        "timestamp": "1997-06-18T03:52:21.000Z",
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    ("interaction_type", "action", "value", "expected"),
    [
        ("click", "record", 1, ("click", 1.0)),
        ("share", "record", 1, ("share", 1.0)),
        ("rating", "set", 4.5, ("rating", 4.5)),
        ("reaction", "set", 1, ("like", 1.0)),
        ("reaction", "set", -1, ("dislike", -1.0)),
        ("watch", "record", 0.5, ("watch", 0.5)),
        ("watch", "record", 0.94, ("watch", 0.94)),
        ("watch", "record", 0.95, ("complete", 1.0)),
        ("watch", "record", 1.0, ("complete", 1.0)),
    ],
)
def test_backend_triples_map_to_model_events(interaction_type, action, value, expected):
    record = backend_record(
        interaction_type=interaction_type,
        interaction_action=action,
        interaction_value=value,
    )
    assert translate_event(record, THRESHOLD) == expected


@pytest.mark.parametrize("interaction_type", ["rating", "reaction"])
def test_clearing_produces_no_event(interaction_type):
    """Undoing a rating is neither a positive nor a negative signal.

    A zero-valued row would reach build_training_matrix as a very low rating and
    be read as dislike, which is the opposite of what the user did.
    """
    record = backend_record(
        interaction_type=interaction_type,
        interaction_action="clear",
        interaction_value=0,
    )
    assert translate_event(record, THRESHOLD) == (None, None)


def test_every_produced_event_type_is_one_the_model_scores():
    produced = {
        translate_event(
            backend_record(
                interaction_type=interaction_type,
                interaction_action=action,
                interaction_value=value,
            ),
            THRESHOLD,
        )[0]
        for interaction_type, action, value in (
            ("click", "record", 1),
            ("share", "record", 1),
            ("rating", "set", 4.0),
            ("reaction", "set", 1),
            ("reaction", "set", -1),
            ("watch", "record", 0.6),
            ("watch", "record", 1.0),
        )
    }
    supported = set(SUPPORTED_EVENT_TYPES)
    assert produced <= supported
    # Seven of the eight. `comment` needs a backend feature that does not exist.
    assert supported - produced == {"comment"}


def test_records_already_in_model_vocabulary_pass_through():
    """An exporter pointed at a table written by another producer still works."""
    record = {"event_type": "share", "value": None}
    assert translate_event(record, THRESHOLD) == ("share", None)


def test_unknown_type_is_surfaced_rather_than_silently_dropped():
    """The caller counts and reports these; hiding them would hide a mismatch."""
    record = backend_record(interaction_type="teleport", interaction_action="record")
    event_type, _ = translate_event(record, THRESHOLD)
    assert event_type == "teleport"
    assert event_type not in SUPPORTED_EVENT_TYPES


@pytest.mark.parametrize(
    ("raw", "expected"), [("99", 99), (99, 99), (" 602 ", 602), (None, None), ("", None)]
)
def test_ids_are_coerced_to_int(raw, expected):
    """DynamoDB holds them as strings; the model indexes users and movies by int."""
    assert _as_int(raw) == expected
