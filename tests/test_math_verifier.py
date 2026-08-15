from concurrent.futures import ThreadPoolExecutor

import pytest

from src.utils import verify_answer


def test_verifier_preserves_ordered_pair_structure():
    assert not verify_answer(r"(12)", r"(1,2)")


def test_verifier_accepts_display_fraction_as_plain_fraction():
    assert verify_answer(r"\dfrac{1}{2}", r"\frac{1}{2}")


@pytest.mark.parametrize(
    "compact, explicit",
    [
        (r"\frac23", r"\frac{2}{3}"),
        (r"-\frac12", r"-\frac{1}{2}"),
        (r"\frac\pi 2", r"\frac{\pi}{2}"),
        (r"\sqrt3", r"\sqrt{3}"),
    ],
)
def test_verifier_accepts_compact_latex_atoms(compact, explicit):
    assert verify_answer(compact, explicit)



def test_verifier_removes_only_real_thousands_separators():
    assert verify_answer(r"\frac{945}{16,384}", r"\frac{945}{16384}")
    assert not verify_answer(r"\{12\}", r"\{1,2\}")


def test_verifier_preserves_numeric_base_annotation():
    assert not verify_answer(r"10001", r"10001_2")
    assert not verify_answer(r"10001_3", r"10001_2")


def test_verifier_works_from_reward_worker_thread():
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(verify_answer, r"\dfrac{1}{2}", r"\frac{1}{2}").result()
    assert result
