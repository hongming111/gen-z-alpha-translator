import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdg.validate import check_row, dedupe_rows

BANNED = {"she has such beta af energy today.", "extremely beta"}


def ok_row(**kw):
    base = dict(slang="that fit is bussin fr", english="that outfit is excellent",
                term="bussin", is_hard_negative=False)
    base.update(kw)
    return base


def test_valid_row_passes():
    assert check_row(ok_row(), BANNED) is None


def test_empty_field_rejected():
    assert check_row(ok_row(english=""), BANNED) is not None


def test_missing_term_rejected():
    assert check_row(ok_row(slang="totally normal sentence"), BANNED) == "term-missing"


def test_hard_negative_skips_term_check():
    assert check_row(ok_row(slang="the road was closed", term="road",
                            is_hard_negative=True), BANNED) is None


def test_eval_leak_rejected():
    r = ok_row(slang="She has such beta af energy today.")
    assert check_row(r, BANNED) == "eval-leak"


def test_dedupe():
    rows = [ok_row(), ok_row(), ok_row(slang="different one", term="fr")]
    assert len(dedupe_rows(rows)) == 2
