#!/usr/bin/env python3
"""Test verify_answer from src.utils with known problem cases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import verify_answer, normalize_latex

print("Testing normalize_latex + verify_answer from src.utils\n")

tests = [
    (r"\frac{14}{3}",    r"\dfrac{14}{3}",    True,  "dfrac vs frac"),
    (r"\frac{7}{9}",     r"\dfrac{7}{9}",     True,  "dfrac vs frac"),
    (r"\frac{1}{2}",     r"\dfrac{1}{2}",     True,  "dfrac vs frac"),
    (r"-\frac{35}{9}",   r"-\dfrac{35}{9}",   True,  "negative dfrac"),
    (r"-\frac{49}{12}",  r"-\dfrac{49}{12}",  True,  "negative dfrac"),
    (r"\frac{h^2}{m}",   r"\dfrac{h^2}{m}",   True,  "symbolic dfrac"),
    (r"\frac{5\sqrt{42}}{27}", r"\dfrac{5\sqrt{42}}{27}", True, "sqrt + dfrac"),
    (r"90^\circ",        r"90",               True,  "degrees symbol"),
    (r"\left( 3, \frac{\pi}{2} \right)", r"(3, \frac{\pi}{2})", True, "left/right parens"),
    (r"(-\infty,-5]\cup[5,\infty)", r"(-\infty, -5] \cup [5, \infty)", True, "set with spaces"),
    (r"\sqrt{51}",       r"\sqrt{51}",        True,  "identical sqrt"),
    (r"p - q",           r"p - q",            True,  "symbolic"),
    (r"42",              r"42",               True,  "plain number"),
    (r"42",              r"43",               False, "different numbers"),
    (r"\frac{3}{2}",     r"\dfrac{3}{2}",     True,  "simple dfrac"),
    (r"\text{Evelyn}",   r"Evelyn",           True,  "text wrapper"),
    (r"6 - 5i",          r"6 - 5i",           True,  "complex number"),
    (r"3\sqrt{13}",      r"3\sqrt{13}",       True,  "coeff sqrt"),
]

passed = 0
failed = 0
for gt, pred, expected, desc in tests:
    result = verify_answer(pred, gt)
    ok = result == expected
    if ok:
        passed += 1
    else:
        failed += 1
        # Show normalization for debugging
        print(f"  Norm GT:   '{normalize_latex(gt)}'")
        print(f"  Norm Pred: '{normalize_latex(pred)}'")
    status = "✓" if ok else "✗"
    print(f"  {status} [{desc:20s}] GT={gt:40s} Pred={pred:40s} -> {result} (expected {expected})")

print(f"\nResults: {passed}/{passed+failed} passed, {failed} failed")
