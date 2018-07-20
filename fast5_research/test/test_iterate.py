import os
import tempfile
import unittest
from uuid import uuid4

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

    def test_002_from_file(self):
        tmp_file = os.path.join(tempfile.gettempdir(), str(uuid4()))
        with open(tmp_file, 'w') as fh:
            fh.write('filename\tjunk\n')
            for i, fname in enumerate(iterate_fast5(self.path, paths=True)):
                fh.write('{}\t{}\n'.format(os.path.basename(fname), i))
        fnames = list(iterate_fast5(self.path, paths=True, strand_list=tmp_file))
        self.assertEqual(len(fnames), 3)


if __name__ == "__main__":
    unittest.main()
