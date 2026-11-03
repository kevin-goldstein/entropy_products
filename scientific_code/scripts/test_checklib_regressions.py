r"""Failure-detection regressions: non-finite components must never become a passing residual.

Expected failures run in child processes with captured output, so their [FAIL]
lines cannot be mistaken for failures of this regression script. Temporary
copies exercise the actual Kerr-Newman and Kerr-Sen grid entrypoints.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import mpmath as mp
import numpy as np
import sympy as sp
import checklib as ck

HERE = Path(__file__).resolve().parent


def rejects(values):
    try:
        ck.max_abs(values)
    except (TypeError, ValueError, OverflowError):
        return True
    return False


def child(code):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(HERE) + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run([sys.executable, '-c', code], cwd=HERE, env=env,
                          text=True, capture_output=True, timeout=60)


def rejected_script(label, source, anchor, insertion):
    text = (HERE / source).read_text()
    ck.true(label + ': injection location exists exactly once', text.count(anchor) == 1)
    if text.count(anchor) != 1:
        return
    text = text.replace(anchor, anchor + insertion, 1)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(HERE) + os.pathsep + env.get('PYTHONPATH', '')
    with tempfile.TemporaryDirectory(prefix='hot-attractor-check-') as directory:
        path = Path(directory) / source
        path.write_text(text)
        result = subprocess.run([sys.executable, str(path)], cwd=HERE, env=env,
                                text=True, capture_output=True, timeout=60)
    output = result.stdout + result.stderr
    ck.true(label + ': nonzero exit and explicit non-finite diagnostic',
            result.returncode != 0 and 'non-finite residual' in output,
            'captured child exit %d' % result.returncode)


ck.close('nested collections, NumPy and mpmath matrices reduce to the correct finite maximum',
         ck.max_abs(([1, -2], np.array([[3, -4]]), mp.matrix([[5j, 0]]), (x for x in [2, 3]))), 5, 0)
ck.small_all('finite componentwise positive check', [[0, mp.mpf('1e-12')], [0]], mp.mpf('1e-10'))
ck.large_any('finite componentwise negative control', [[0, mp.mpf('0.02')], [0]], mp.mpf('0.01'))

for label, bad in [('NaN', mp.nan), ('+Inf', mp.inf), ('-Inf', -mp.inf)]:
    for position in range(3):
        values = [mp.mpf(0), mp.mpf('1e-30'), mp.mpf(0)]
        values[position] = bad
        ck.true('%s at component %d is rejected, including after a finite maximum' % (label, position), rejects(values))
ck.true('an all-NaN collection is rejected', rejects([mp.nan, mp.nan]))
ck.true('NumPy matrix containing NaN is rejected', rejects(np.array([[0, 1], [np.nan, 0]])))
ck.true('mpmath matrix containing NaN is rejected', rejects(mp.matrix([[0, 1], [mp.nan, 0]])))
ck.true('SymPy matrix containing NaN is rejected', rejects(sp.Matrix([[0, 1], [sp.nan, 0]])))
ck.true('complex non-finite component is rejected', rejects([0, complex(0, float('inf'))]))
ck.true('an empty collection cannot pass vacuously', rejects([[], []]))

with mp.workdps(90):
    target = mp.mpf(1) + mp.mpf('1e-60')
    ck.close('arbitrary precision is preserved beyond binary64', ck.max_abs([1, target]), target, mp.mpf('1e-80'))
    ck.true('subnormal-to-binary64 magnitude is not silently rounded to zero',
            ck.max_abs([mp.mpf('1e-400')]) == mp.mpf('1e-400') and ck.max_abs([mp.mpf('1e-400')]) > 0)

for name, call in [
    ('positive array check', 'ck.small_all("invalid positive", [0, float("nan")], 1)'),
    ('negative array control', 'ck.large_any("invalid negative", [2, float("nan")], 1)'),
    ('all-NaN array check', 'ck.small_all("all nan", [float("nan"), float("nan")], 1)'),
    ('direct NaN', 'ck.small("direct nan", float("nan"), 1)'),
    ('direct infinity', 'ck.large("direct inf", float("inf"), 1)'),
    ('empty check set', ''),
    ('precision-sensitive failed check', 'mp.mp.dps=90; ck.small("tiny but too large", mp.mpf("1e-400"), mp.mpf("1e-500"))'),
]:
    result = child('import checklib as ck; import mpmath as mp; ' + call + '; ck.finish()' if call
                   else 'import checklib as ck; ck.finish()')
    ck.true(name + ': finish returns failure', result.returncode == 1,
            'captured child exit %d' % result.returncode)

kn_anchor = "E, Mx, S = res(mp.mpf('3.1'), mp.mpf('0.7'), (mp.mpf('1.3'), mp.mpf('0.6'), mp.mpf('0.7')))"
for name, injection in [('one Einstein component is NaN', '\nE[1, 1] = mp.nan'),
                        ('one Maxwell component is infinity', '\nMx[1] = mp.inf')]:
    rejected_script('actual Kerr-Newman test: ' + name, 'test_eom_num.py', kn_anchor, injection)
rejected_script('actual Kerr-Sen grid: every tensor component is NaN', 'kerr_sen_eom_grid.py',
                '            E, Mx, S = res(rr, tt, pv)',
                '\n            E = mp.matrix([[mp.nan]*4 for _ in range(4)]); Mx = [mp.nan]*4; S = [mp.nan]*len(S)')
rejected_script('actual negative control: a later Einstein component is NaN', 'test_eom_num.py',
                "E11, Mx11, S11 = res11(mp.mpf('3.1'), mp.mpf('0.7'), (mp.mpf('1.3'), mp.mpf('0.6'), mp.mpf('0.7')))",
                '\nE11[1, 1] = mp.nan')

ck.finish()
