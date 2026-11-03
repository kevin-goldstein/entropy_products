"""Exercise runner failure handling with disposable child programs.

Run separately: python3 tests/test_runner.py. No scientific reference data are
changed by these tests; all files and intentionally failing programs are temporary.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as runner


class RunnerTests(unittest.TestCase):
    def invoke(self, program, options=(), task='sample', defaults=False):
        with tempfile.TemporaryDirectory(prefix='hot attractor runner test ') as dirname:
            root = Path(dirname)
            (root/'scripts').mkdir()
            (root/'scripts'/f'{task}.py').write_text(program)
            args = ['run.py', *options] if defaults else ['run.py', '--output-dir', str(root/'results'), '--log-dir', str(root/'logs'), *options]
            with patch.object(runner,'ROOT',root), patch.object(runner,'TASKS',{task:('scripts',f'{task}.py')}), \
                    patch.object(runner,'QUICK',[task]), patch.object(sys,'argv',args), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = runner.main()
            report_path = root/('results/validation/quick/summary.json' if defaults else 'logs/summary.json')
            probe_path = root/('results/data/probe.txt' if defaults else 'results/probe.txt')
            report = json.loads(report_path.read_text()) if report_path.exists() else None
            output = probe_path.read_text() if probe_path.exists() else None
            self.assertFalse((root/'data/generated').exists())
            self.assertFalse((root/'validation/reference').exists())
            return code, report, output

    def test_defaults_preserve_reference_directories(self):
        code,report,output = self.invoke("import os\nfrom pathlib import Path\nPath(os.environ['HOT_ATTRACTORS_OUTPUT'],'probe.txt').write_text('local results')\n", defaults=True)
        self.assertEqual((code,output),(0,'local results'))
        self.assertEqual(report['tasks_passed'],1)

    def test_success_and_output_routing(self):
        code,report,output = self.invoke("import os\nfrom pathlib import Path\nPath(os.environ['HOT_ATTRACTORS_OUTPUT'],'probe.txt').write_text('routed')\nprint('[PASS] probe')\n")
        self.assertEqual((code,output),(0,'routed'))
        self.assertEqual(report['tasks_passed'],1)
        self.assertEqual(report['pass_markers'],1)

    def test_nonzero_exit_is_failure(self):
        code,report,_=self.invoke('raise SystemExit(7)\n')
        self.assertEqual(code,1)
        self.assertEqual(report['tasks'][0]['exit_code'],7)

    def test_fail_marker_is_failure_even_with_zero_exit(self):
        code,report,_=self.invoke("print('[FAIL] intentional negative case')\n")
        self.assertEqual(code,1)
        self.assertEqual(report['fail_markers'],1)

    def test_assertions_remain_active(self):
        code,report,_=self.invoke("assert False, 'intentional assertion'\n")
        self.assertEqual(code,1)
        self.assertNotEqual(report['tasks'][0]['exit_code'],0)

    def test_timeout_is_failure(self):
        code,report,_=self.invoke('import time\ntime.sleep(10)\n',['--timeout','0.1'])
        self.assertEqual(code,1)
        self.assertTrue(report['tasks'][0]['timed_out'])
        self.assertEqual(report['tasks'][0]['exit_code'],124)

    def test_list_does_not_execute_program(self):
        code,report,_=self.invoke("raise AssertionError('must not execute')\n",['--list'])
        self.assertEqual(code,0)
        self.assertIsNone(report)

    def test_manuscript_figure_copies_successful_output(self):
        with tempfile.TemporaryDirectory() as dirname:
            code, report, _ = self.invoke(
                "import os\nfrom pathlib import Path\np=Path(os.environ['HOT_ATTRACTORS_OUTPUT'])/'figures'\np.mkdir()\n(p/'orbit.pdf').write_bytes(b'fresh figure')\n",
                ['--paper-figures-dir', dirname], task='plot_orbit')
            self.assertEqual(code, 0)
            self.assertEqual((Path(dirname)/'orbit.pdf').read_bytes(), b'fresh figure')
            self.assertEqual(report['tasks'][0]['paper_figure'], str(Path(dirname).resolve()/'orbit.pdf'))

    def test_failed_figure_task_preserves_manuscript_figure(self):
        with tempfile.TemporaryDirectory() as dirname:
            target=Path(dirname)/'orbit.pdf'
            target.write_bytes(b'previous figure')
            code, report, _ = self.invoke(
                "import os\nfrom pathlib import Path\np=Path(os.environ['HOT_ATTRACTORS_OUTPUT'])/'figures'\np.mkdir()\n(p/'orbit.pdf').write_bytes(b'incomplete figure')\nprint('[FAIL] intentional')\n",
                ['--paper-figures-dir', dirname], task='plot_orbit')
            self.assertEqual(code, 1)
            self.assertEqual(target.read_bytes(), b'previous figure')
            self.assertNotIn('paper_figure', report['tasks'][0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
