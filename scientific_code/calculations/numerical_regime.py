"""Summarize sampled amplitudes, horizon-area bounds and cross-method overlap.

This task reads existing scientific outputs; it does not integrate an ODE or
interpret matching refinements as independent methods. With a custom output
root it uses fresh input files there if present, otherwise the packaged inputs.
Every input path and SHA-256 checksum is recorded.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
from output_paths import path_label

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('HOT_ATTRACTORS_OUTPUT', ROOT / 'results/data'))


def source(relative):
    fresh = OUT / relative
    return fresh if fresh.is_file() else ROOT / 'data/generated' / relative


def main():
    branch_file = source('branches/branch_trace.csv')
    radial_file = source('branch/amplitude_scan.csv')
    with branch_file.open(newline='') as fp:
        rows = list(csv.DictReader(fp))
    with radial_file.open(newline='') as fp:
        radial = list(csv.DictReader(fp))
    selected = []
    for n in range(3, 8):
        for eps in [.2, .5, 1., 2., 3., 6.]:
            row = next(r for r in rows if int(r['n']) == n and abs(float(r['epsilon']) - eps) < 1e-10)
            nu, alpha, eta = map(float, [row['nu'], row['alpha'], row['eta']])
            k = math.cosh(alpha * eps)
            selected.append(dict(n=n, epsilon=eps, nu=nu, alpha=alpha, eta=eta,
                                 deficit_percent=100 * (1 - eta), k=k,
                                 outer_entropy_lower_bound_over_Sext=k,
                                 inner_entropy_upper_bound_over_Sext=eta / k,
                                 gap_bound_over_Sext=k - eta / k))
    comparisons = []
    for row in radial:
        eps = float(row['epsilon'])
        matching = [r for r in rows if int(r['n']) == 3 and abs(float(r['epsilon']) - eps) < 1e-10]
        if not matching:
            continue
        other = matching[0]
        comparisons.append(dict(epsilon=eps,
                                abs_alpha_difference=abs(float(row['alpha']) - float(other['alpha'])),
                                abs_eta_difference=abs(float(row['eta']) - float(other['eta']))))
    assert len(comparisons) >= 7
    max_alpha = max(x['abs_alpha_difference'] for x in comparisons)
    max_eta = max(x['abs_eta_difference'] for x in comparisons)
    assert max_alpha < 1e-10 and max_eta < 1e-10
    assert all(r['k'] >= 1 and r['gap_bound_over_Sext'] > 1 - r['eta'] for r in selected)
    endpoint = next(r for r in selected if r['n'] == 3 and r['epsilon'] == 6.)
    slope = endpoint['alpha'] * math.sinh(endpoint['alpha'] * 6) / (2 * math.sqrt(endpoint['eta']))
    metadata = {
        'new_ODE_solves': False,
        'sources': [{'path': path_label(p, ROOT), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in (branch_file, radial_file)],
        'samples': selected,
        'cross_method_overlap': {
            'samples': comparisons, 'max_alpha_difference': max_alpha, 'max_eta_difference': max_eta,
            'scope': 'Comparison of stored independent radial and scaled autonomous implementations only at overlapping amplitudes.'},
        'balanced_horizon_layer': {
            'dphi_d_fractional_radial_distance_magnitude': slope,
            'fraction_for_linear_scalar_change_0p1': .1 / slope,
            'fraction_for_linear_potential_exponent_change_1': 1 / (endpoint['alpha'] * slope),
            'scope': 'Leading horizon Taylor derivative: a scale diagnostic, not a resolved nonlinear transition location.'},
        'gap_bound': {'expression': '(k-eta/k)*Sext', 'k': 'cosh(alpha*epsilon)',
                      'three_node_leading_term': '24*epsilon**2+O(epsilon**4)',
                      'assumptions': 'Positive outer radius slope; Veff_min = Sext/pi; fixed outer horizon scalar on an orbit.'},
        'scope': 'Finite sampled values only. Independent radial constructions are not claimed at large amplitudes.'}
    dest = OUT / 'branches'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'numerical_regime.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with (dest / 'numerical_regime.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=selected[0].keys())
        writer.writeheader(); writer.writerows(selected)
    print(f'[PASS] {len(selected)} sampled amplitude summaries and {len(comparisons)} cross-method comparisons')
    print(f'Maximum alpha difference {max_alpha:.3g}; eta difference {max_eta:.3g}')


if __name__ == '__main__':
    main()
