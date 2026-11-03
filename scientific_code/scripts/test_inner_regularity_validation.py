"""Regression checks for failed, incomplete and non-finite inward integrations.

The early-stop cases use real partial solutions of the supplied equations; only
solver status metadata is altered when simulating an interrupted integration.
Incomplete physical integrations are rejected before diagnosing regularity.
"""
import copy
from unittest.mock import Mock, patch

import numpy as np
import checklib as ck
import inner_regularity_physical as ip


def controlled_error(label, action, expected_text):
    try:
        action()
    except ip.IntegrationError as exc:
        ck.true(label, expected_text in str(exc), str(exc))
    else:
        ck.true(label, False, 'expected an IntegrationError')


def unresolved(label, result, expected_text):
    ck.true(label, result['outcome'] == 'unresolved' and result['product'] is None
            and expected_text in result.get('reason', ''), result.get('reason', 'missing reason'))


if __name__ == '__main__':
    a1, a2, eps = 2.0, -6.0, 0.2
    ps, bde, _ = ip.doubly_extremal(a1, a2)
    bp, php = 2 * bde, ps + eps
    args = (a1, a2, bp, php)
    original_inward = ip.inward
    complete = original_inward(*args)
    partial = original_inward(*args, rend=0.01)
    ck.true('early-stop regression uses a real physical solution with an asymptotically flat exterior',
            ip.horizon_data(*args)[0] > 0 and ip.outward(*args)['flat'])
    baseline = ip.diagnose(*args)
    ck.true('the complete integration resolves the known singular logarithmic term', baseline['outcome'] == 'nonzero')
    ck.close('complete singular result retains its numerical value', baseline['c'], -0.025870427674918223, 1e-7)

    interrupted = copy.copy(partial)
    interrupted.success, interrupted.status = False, -1
    interrupted.message = 'controlled interruption after a real partial integration'
    with patch.object(ip, 'inward', return_value=interrupted):
        failed = ip.diagnose(*args)
        unresolved('failed main integration cannot masquerade as a regular horizon', failed, 'main integration')
        ck.true('the diagnostic exposes the solver failure reason', 'controlled interruption' in ip.fmt_c(failed))
        controlled_error('run rejects a failed inward solution', lambda: ip.run(*args, exterior=False), 'did not complete')
        controlled_error('shooting coefficient rejects a failed solution', lambda: ip.c_fine(*args), 'did not complete')
        controlled_error('coupling root finder aborts on a failed solve', lambda: ip.tuned_alpha(0.2), 'did not complete')

    with patch.object(ip, 'inward', return_value=partial):
        unresolved('successful termination at an early radius still lacks coverage', ip.diagnose(*args), 'does not cover')
        controlled_error('run also checks its intended endpoint', lambda: ip.run(*args, exterior=False), 'does not cover')

    for stage, choose_failure in (
            ('launch-refined', lambda kw: kw.get('d') == 1e-7),
            ('tolerance-refined', lambda kw: kw.get('rtol') == 1e-13)):
        def refined(*call_args, **kw):
            return interrupted if choose_failure(kw) else original_inward(*call_args, **kw)
        with patch.object(ip, 'inward', side_effect=refined):
            unresolved('%s failure invalidates the diagnosis and product' % stage, ip.diagnose(*args), stage)

    nonfinite_stored = copy.copy(complete)
    nonfinite_stored.y = complete.y.copy()
    nonfinite_stored.y[2, 3] = np.nan
    with patch.object(ip, 'inward', return_value=nonfinite_stored):
        unresolved('a NaN anywhere in the stored trajectory is rejected', ip.diagnose(*args), 'non-finite stored trajectory')

    for invalid in (np.nan, np.inf):
        nonfinite_dense = copy.copy(complete)
        def dense_with_invalid(radius, value=invalid):
            values = np.array(complete.sol(radius), copy=True)
            values[3] = value
            return values
        nonfinite_dense.sol = dense_with_invalid
        with patch.object(ip, 'inward', return_value=nonfinite_dense):
            unresolved('non-finite dense scalar samples (%s) invalidate the diagnosis' % invalid,
                       ip.diagnose(*args), 'non-finite dense values')
        controlled_error('non-finite vector samples (%s), used by the figure, are rejected' % invalid,
                         lambda: ip.sample_inward(nonfinite_dense, ip.RM + np.array([1e-5, 1e-8])),
                         'non-finite dense values')

    guarded = copy.copy(complete)
    guarded.sol = Mock(wraps=complete.sol)
    controlled_error('a request before the integration domain is refused',
                     lambda: ip.sample_inward(guarded, ip.RM + ip.REND / 2), 'does not cover')
    controlled_error('a request after the integration domain is refused',
                     lambda: ip.sample_inward(guarded, ip.RP), 'does not cover')
    controlled_error('non-finite query coordinates are refused',
                     lambda: ip.sample_inward(guarded, np.nan), 'non-finite sample radius')
    ck.true('invalid requests never call the extrapolating interpolant', not guarded.sol.called)
    values = ip.sample_inward(complete, ip.RM + np.array([1e-4, 1e-6, 1e-8]), 'valid vector profile')
    ck.true('valid vector sampling retains all four finite fields', values.shape == (4, 3) and np.all(np.isfinite(values)))

    # Finite fit values can still overflow during differencing or extrapolation.
    # These focused arithmetic probes bypass the physical solver intentionally.
    for label, estimates in (
            ('differences', [np.finfo(float).max, -np.finfo(float).max, 0.0, 0.0]),
            ('geometric tail', [-1e308, -2e307, 4.4e307, 9.52e307]),
            ('convergence ratio', [1.0, 1.0, 2.0, 2.0])):
        fits = [(np.float64(v), 1.0) for v in estimates + [estimates[-1], estimates[-1]]]
        with patch.object(ip, 'inward', return_value=complete), patch.object(ip, 'two_radius', side_effect=fits):
            unresolved('non-finite derived %s cannot produce a classification' % label,
                       ip.diagnose(*args), 'fit extrapolation')
    try:
        ip.finite_spread([0.0, np.nan, 1.0])
    except ValueError:
        ck.true('a later NaN cannot hide inside an area-product or coefficient spread', True)
    else:
        ck.true('a later NaN cannot hide inside an area-product or coefficient spread', False)

    for a1, a2, small_eps in ((2.0, -6.0, 0.001), (3.0, -8.0, 0.005)):
        ps, bde, _ = ip.doubly_extremal(a1, a2)
        dg = ip.diagnose(a1, a2, 2 * bde, ps + small_eps)
        ck.true('small genuine signal (%g,%g), eps=%g remains nonzero' % (a1, a2, small_eps),
                dg['outcome'] == 'nonzero' and np.isfinite(dg['c']) and dg['product'] is None)
    ps, bde, _ = ip.doubly_extremal(2.0, -2.0)
    regular = ip.diagnose(2.0, -2.0, 1.3 * bde, ps + 0.2)
    ck.true('a valid Toda solution remains consistent with zero', regular['outcome'] == 'zero')
    ck.close('the valid Toda area product is unchanged', regular['product'], 1.0, 1e-10)
    ck.finish()
