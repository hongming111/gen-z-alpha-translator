import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from teacher import extract_json


def test_extract_json_plain():
    assert extract_json('{"slang": "delulu", "english": "delusional"}') == {
        "slang": "delulu", "english": "delusional"}


def test_extract_json_with_prose_around():
    text = 'Sure! Here is the pair:\n{"slang": "rizz", "english": "charm"}\nHope that helps.'
    assert extract_json(text) == {"slang": "rizz", "english": "charm"}


def test_extract_json_strips_think_wrapper():
    text = '<think>let me reason</think>{"a": 1}'
    assert extract_json(text) == {"a": 1}


def test_extract_json_none_on_garbage():
    assert extract_json("no json here") is None
