import os
import unittest

from fast5_research import Fast5, iterate_fast5


class IterateFiles(unittest.TestCase):
    def setUp(self):
        self.path = (os.path.join(
            os.path.dirname(__file__), 'data', 'recursive'
        ))

    def test_000_single_layer(self):
        fnames = list(iterate_fast5(self.path, paths=True))
        self.assertEqual(len(fnames), 3)

    def test_001_recursive(self):
        fnames = list(iterate_fast5(self.path, paths=True, recursive=True))
        self.assertEqual(len(fnames), 5)

if __name__ == "__main__":
    unittest.main()
