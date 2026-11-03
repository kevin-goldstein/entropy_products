#!/usr/bin/env python3
"""Portable entry point for the scientific calculations and validation suite."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
BASE = '''test_checklib_regressions test_inner_regularity_validation
test_exact_dilaton_integration cold_from_hot_check static_K check_K_conservation
check_K_dyonic4d reduced_equations_check ddim_identity dilaton_exact dilaton_thermo
agm_checks smooth_representative_check complex_horizons btz_check subtracted_rn
test_eom_num kerr_newman_check kerr_sen_fields kerr_sen_check kerr_sen_eom_grid
region2_rotating rotating_attractor_limit c3_exact inner_regularity_physical
fig_inner_regularity tuned_branch_bvp test_tuned_branch_bvp
complex_horizons_symbolic toda_exact klr_continuation continued_bvp'''.split()
CALCULATIONS = ['quartic', 'half_interval_scan', 'orbit_scan', 'validate_core', 'plot_orbit',
                'branch_trace', 'plot_branches', 'higher_modes', 'plot_deficit', 'plot_families', 'numerical_regime', 'plot_excess']
CONTROLS = ['check_static_virial', 'rotating_exact_controls', 'complex_validation', 'second_variation_control',
            'coupling_series_control', 'series_fraction_comparison', 'exact_obstruction_control',
            'toda_b2_control', 'general_translation_check', 'fixed_coupling_control', 'asymmetric_excess', 'asymmetric_branch_check', 'asymmetric_collocation']
SLOW = set('''kerr_sen_fields kerr_sen_check region2_rotating rotating_attractor_limit
inner_regularity_physical fig_inner_regularity klr_continuation continued_bvp'''.split())
SLOW.add('higher_modes')
QUICK = ['quartic', 'tuned_branch_bvp', 'test_tuned_branch_bvp', 'half_interval_scan',
         'orbit_scan', 'validate_core', 'plot_orbit', 'c3_exact',
         'check_static_virial', 'rotating_exact_controls', 'complex_validation', 'second_variation_control']
QUICK += ['coupling_series_control', 'exact_obstruction_control', 'toda_b2_control', 'general_translation_check',
          'branch_trace', 'fixed_coupling_control', 'plot_branches', 'plot_deficit', 'plot_families', 'numerical_regime']
TASKS = {name: ('scripts', name+'.py') for name in BASE}
TASKS.update({name: ('calculations', name+'.py') for name in CALCULATIONS})
TASKS.update({name: ('controls', name+'.py') for name in CONTROLS})


def environment():
    versions = {}
    for name in ('numpy', 'scipy', 'sympy', 'mpmath', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'platform': platform.platform(), 'machine': platform.machine(), 'packages': versions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['quick', 'standard', 'full'], default='quick',
                        help='quick: central results; standard: omit costly tensor/IVP checks; full: all tasks')
    parser.add_argument('--only', nargs='+', choices=sorted(TASKS),
                        help='run only these named tasks, in the given order')
    parser.add_argument('--list', action='store_true', help='list selected tasks without executing')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/data',
                        help='scientific outputs (default: package results/data)')
    parser.add_argument('--log-dir', type=Path,
                        help='logs and machine-readable summary (default: results/validation/MODE)')
    parser.add_argument('--paper-figures-dir', type=Path,
                        help='also copy freshly generated manuscript figure PDFs into this manuscript figure directory')
    parser.add_argument('--timeout', type=float, default=None,
                        help='optional limit in seconds per task; timeout is reported as failure')
    args = parser.parse_args()
    if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
        parser.error('Run without -O/PYTHONOPTIMIZE: exact and numerical assertions must remain active.')
    if args.timeout is not None and args.timeout <= 0:
        parser.error('--timeout must be positive')
    selected = args.only or (QUICK if args.mode == 'quick' else
                            [n for n in TASKS if args.mode == 'full' or n not in SLOW])
    if args.paper_figures_dir is not None and not {'plot_orbit', 'plot_branches', 'plot_deficit', 'plot_families', 'plot_excess'}.intersection(selected):
        parser.error('--paper-figures-dir requires at least one plot task in the selected tasks')
    if args.list:
        for name in selected:
            print('/'.join(TASKS[name])+('  [potentially long]' if name in SLOW else ''))
        return 0
    out = args.output_dir.resolve()
    logs = (args.log_dir or ROOT/'results/validation'/('selected' if args.only else args.mode)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    report = {'mode': 'selected' if args.only else args.mode,
              'utc_started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'environment': environment(), 'tasks': [],
              'scope': 'Only listed tasks were executed; no claim is made for omitted tasks.'}
    missing = [k for k,v in report['environment']['packages'].items() if v is None]
    if missing:
        print('Missing dependencies: '+', '.join(missing)+'. Install requirements.txt with this Python.', file=sys.stderr)
        return 2
    (logs/'environment.json').write_text(json.dumps(report['environment'],indent=2)+'\n')
    env = dict(os.environ, HOT_ATTRACTORS_OUTPUT=str(out), PYTHONDONTWRITEBYTECODE='1')
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='hot-attractors-mpl-') as mpldir:
        env['MPLCONFIGDIR'] = mpldir
        env['XDG_CACHE_HOME'] = mpldir
        for i,name in enumerate(selected,1):
            script = ROOT.joinpath(*TASKS[name])
            print(f'[{i}/{len(selected)}] {name}', flush=True)
            tick = time.perf_counter()
            logpath = logs/(name+'.log')
            timed_out = False
            with logpath.open('w',encoding='utf-8') as fp:
                try:
                    result = subprocess.run([sys.executable,str(script)],cwd=ROOT,env=env,
                                            stdout=fp,stderr=subprocess.STDOUT,timeout=args.timeout)
                    rc = result.returncode
                except subprocess.TimeoutExpired:
                    fp.write('\nRUNNER: per-task timeout exceeded.\n')
                    timed_out, rc = True, 124
            text = logpath.read_text(encoding='utf-8',errors='replace')
            marked_pass = len(re.findall(r'\[PASS\]',text))
            marked_fail = len(re.findall(r'\[FAIL\]',text))
            paper_figure = None
            if args.paper_figures_dir is not None and name in {'plot_orbit', 'plot_branches', 'plot_deficit', 'plot_families', 'plot_excess'} and rc == 0 and marked_fail == 0:
                try:
                    figure_name = {'plot_orbit': 'orbit.pdf', 'plot_branches': 'branches.pdf', 'plot_deficit': 'deficit.pdf', 'plot_families': 'families.pdf', 'plot_excess': 'excess_example.pdf'}[name]
                    target = args.paper_figures_dir.resolve()/figure_name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(out/'figures'/figure_name, target)
                    paper_figure = str(target)
                except OSError as error:
                    with logpath.open('a', encoding='utf-8') as fp:
                        fp.write(f'\nRUNNER: could not copy manuscript figure: {error}\n')
                    rc = 1
            # Some exact calculations use Python assertions and emit JSON. Do not
            # manufacture a number of checks when the script does not count them.
            row = {'task': name, 'script': '/'.join(TASKS[name]), 'exit_code': rc,
                   'seconds': round(time.perf_counter()-tick,3), 'pass_markers': marked_pass,
                   'fail_markers': marked_fail, 'timed_out': timed_out,
                   'passed': rc == 0 and marked_fail == 0, 'log': logpath.name}
            if paper_figure is not None:
                row['paper_figure'] = paper_figure
            report['tasks'].append(row)
            print(f"  {'PASS' if row['passed'] else 'FAIL'} ({row['seconds']:.2f}s)",flush=True)
            (logs/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    report['total_seconds'] = round(time.perf_counter()-started,3)
    report['tasks_passed'] = sum(r['passed'] for r in report['tasks'])
    report['tasks_failed'] = len(report['tasks'])-report['tasks_passed']
    report['pass_markers'] = sum(r['pass_markers'] for r in report['tasks'])
    report['fail_markers'] = sum(r['fail_markers'] for r in report['tasks'])
    (logs/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['task                                status      seconds   PASS   FAIL']
    lines += [f"{r['task']:<35} {'PASS' if r['passed'] else 'FAIL':<8} {r['seconds']:>9.3f} {r['pass_markers']:>6} {r['fail_markers']:>6}" for r in report['tasks']]
    lines.append(f"{report['tasks_passed']} tasks passed, {report['tasks_failed']} failed; {report['total_seconds']:.3f} seconds.")
    lines.append('PASS-marker totals exclude uncounted assertions and are not a count of independent methods.')
    (logs/'summary.txt').write_text('\n'.join(lines)+'\n')
    print(lines[-2],flush=True)
    return int(report['tasks_failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
