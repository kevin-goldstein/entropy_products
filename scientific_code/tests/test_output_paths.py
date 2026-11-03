"""Check portable provenance and separation of reference and runtime files."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'calculations'))
import manifest
from output_paths import path_label


class OutputPathTests(unittest.TestCase):
    def test_in_repository_paths_are_relative(self):
        with tempfile.TemporaryDirectory(prefix='hot attractors ') as dirname:
            root = Path(dirname)
            self.assertEqual(path_label(root/'data/generated/branch.csv', root),
                             'data/generated/branch.csv')
            self.assertEqual(path_label(root/'results/data/branch.csv', root),
                             'results/data/branch.csv')

    def test_external_paths_remain_unambiguous(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            self.assertEqual(path_label(root/'external/branch.csv', root/'package'),
                             str(root.resolve()/'external/branch.csv'))

    def test_manifest_covers_references_but_ignores_new_results(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            for name in ['data/generated/branch.csv', 'validation/reference/summary.json',
                         'results/data/branch.csv', 'results/validation/summary.json',
                         '__pycache__/module.pyc']:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            self.assertEqual({p.relative_to(root).as_posix() for p in manifest.files(root)},
                             {'data/generated/branch.csv', 'validation/reference/summary.json'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
