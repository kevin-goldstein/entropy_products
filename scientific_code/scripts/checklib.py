r"""
checklib.py -- enforced checks for the scripts that accompany the paper.

Every check prints one line, '[PASS] label (details)' or '[FAIL] label (details)', and is recorded.  A script ends
with finish(), which prints a summary and exits with status 1 if any check failed, or if none was run.  run.py
collects these exit codes.  Numerical values must be finite; each check states its tolerance.  Negative controls,
which feed in a deliberately wrong input, use large(): they pass only if the residual is clearly nonzero.
"""
import sys
import mpmath as mp

_results = []


def _mag(x):
    """|x| without first narrowing high-precision numbers to a Python float."""
    try:
        return abs(mp.mpmathify(x))
    except (TypeError, ValueError):
        # In particular, SymPy nan/oo/zoo need this conversion. Their non-finite
        # result is rejected by the checks below, not replaced with zero.
        return abs(mp.mpmathify(complex(x)))


def _components(values):
    """Flatten numeric collections, including generators and NumPy/SymPy/mpmath matrices."""
    if isinstance(values, (str, bytes)):
        raise TypeError('residual collections must contain numbers, not text')
    try:
        items = iter(values)
    except TypeError:
        yield values
    else:
        for item in items:
            yield from _components(item)


def max_abs(values):
    """Maximum magnitude after checking EVERY component is finite.

    Unlike Python max(), this cannot silently discard a later NaN. An invalid
    or empty collection raises ValueError so positive and negative controls both
    fail closed. The returned mpmath number retains arbitrary-precision values.
    """
    largest = mp.mpf(0)
    count = 0
    for index, value in enumerate(_components(values)):
        magnitude = _mag(value)
        if not mp.isfinite(magnitude):
            raise ValueError('non-finite residual at component %d: %s' % (index, value))
        if magnitude > largest:
            largest = magnitude
        count += 1
    if count == 0:
        raise ValueError('empty residual collection')
    return largest


def _record(label, ok, detail=''):
    ok = bool(ok)
    _results.append((label, ok))
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', label, '  (%s)' % detail if detail else ''))
    return ok


def true(label, cond, detail=''):
    """a boolean condition."""
    return _record(label, cond, detail)


def small(label, value, tol):
    """|value| <= tol."""
    v = _mag(value)
    return _record(label, mp.isfinite(v) and v <= tol, '|value| = %.2e, tolerance %.0e' % (v, tol))


def close(label, value, target, tol, rel=False):
    """|value - target| <= tol, or <= tol |target| if rel."""
    d = _mag(value - target)
    bound = tol * _mag(target) if rel else tol
    return _record(label, mp.isfinite(d) and d <= bound,
                   'difference %.2e, tolerance %.0e%s' % (d, tol, ' (relative)' if rel else ''))


def large(label, value, lower):
    """negative control: |value| >= lower, i.e. the check is able to fail."""
    v = _mag(value)
    return _record(label, mp.isfinite(v) and v >= lower, '|value| = %.2e, must exceed %.0e' % (v, lower))


def small_all(label, values, tol):
    """Every residual is finite and its magnitude is at most tol."""
    try:
        value = max_abs(values)
    except (TypeError, ValueError, OverflowError) as error:
        return _record(label, False, str(error))
    return small(label, value, tol)


def large_any(label, values, lower):
    """Negative control: all residuals are finite and at least one is large."""
    try:
        value = max_abs(values)
    except (TypeError, ValueError, OverflowError) as error:
        return _record(label, False, str(error))
    return large(label, value, lower)


def zero(label, expr):
    """exact symbolic zero after simplification."""
    import sympy as sp
    e = sp.simplify(expr)
    return _record(label, e == 0, 'simplifies to 0' if e == 0 else 'simplifies to %s' % e)


def finish():
    n = len(_results)
    bad = [lab for lab, ok in _results if not ok]
    print('checks: %d passed, %d failed%s' % (n - len(bad), len(bad), (': ' + '; '.join(bad)) if bad else ''))
    sys.exit(1 if (bad or n == 0) else 0)
