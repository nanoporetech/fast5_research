from collections import namedtuple
import os
import tempfile
import unittest

import numpy as np
import numpy.testing as nptest

from fast5_research import Fast5
from fast5_research.util import _sanitize_data_for_reading

class Fast5BasecallerAndMapper(unittest.TestCase):

    @classmethod
    def get_file_path(self,filename):
        file_path = os.path.join(os.path.dirname(__file__), 'data', filename)
        return file_path

    @classmethod
    def setUpClass(self):
        """Create a read fast5 from scratch with previously simulated mapping and basecall 1D data"""
        print('* Fast5 Basecaller and Mapper')

        self.seq = 'CATTACGCATTTACCGAAACCTGGGCAAA'
        self.qstring = '!'*len(self.seq)
        self.model_file = 'example_template.model'
        self.events_file = 'example_template.events'
        self.model_file = 'example_template.model'
        self.bc_scale_file = 'example_template.bc_scale'
        self.bc_path_file = 'example_template.bc_path'
        self.map_scale_file = 'example_template.map_scale'
        self.map_path_file = 'example_template.map_path'
        self.map_post_file = 'example_template.map_post'
        self.ref_name = 'test_seq'

        # Open new file
        header = ['channel_number', 'offset', 'range', 'digitisation', 'sampling_rate']
        channel_id = {x:0 for x in header}
        tracking_id = tracking_id = {
            'exp_start_time': '1970-01-00T00:00:00Z',
            'run_id': 'a'*32,
            'flow_cell_id': 'FAH00000',
        }
        fakefile = tempfile.NamedTemporaryFile()
        self.fh = Fast5.New(fakefile.name, channel_id=channel_id, tracking_id=tracking_id, read='a')

        # load data to set within fast5 file
        self.model = np.genfromtxt(self.get_file_path(self.model_file), dtype=None, delimiter='\t', names=True, encoding='utf8')

        self.model['kmer'] = self.model['kmer'].astype(str)

        self.events = np.genfromtxt(self.get_file_path(self.events_file), dtype=None, delimiter='\t', names=True)

        # use namedtuple to imitate a Scale object
        Scale = namedtuple('Scale', ['shift', 'scale', 'drift', 'var', 'scale_sd', 'var_sd'])

        bc_scale = Scale(*np.genfromtxt(self.get_file_path(self.bc_scale_file), dtype=None, delimiter='\t'))
        bc_path = np.genfromtxt(self.get_file_path(self.bc_path_file), dtype=np.int32, delimiter='\t')

        self.fh.set_basecall_data(self.events, bc_scale, bc_path, self.model, self.seq)

        map_scale = Scale(*np.genfromtxt(self.get_file_path(self.map_scale_file), dtype=None, delimiter='\t'))
        map_path = np.genfromtxt(self.get_file_path(self.map_path_file), dtype=np.int32, delimiter='\t')
        map_post = np.genfromtxt(self.get_file_path(self.map_post_file), delimiter='\t')

        n_states = len(self.seq) - len(self.model['kmer'][0]) + 1
        self.fh.set_mapping_data(self.events, map_scale, map_path, self.model, self.seq, self.ref_name)
        self.fh.set_mapping_data(self.events, map_scale, map_path, self.model, self.seq, self.ref_name, post=map_post)

    @classmethod
    def tearDownClass(self):
        self.fh.close()

    def test_000_basic_folder_structure(self):
        """Test root folder structure creation"""

        self.assertEqual(list(self.fh.keys()), ['Analyses', 'UniqueGlobalKey'])
        self.assertEqual(list(self.fh['/Analyses'].keys()), ['Basecall_1D_000', 'Squiggle_Map_000', 'Squiggle_Map_001'])

    def test_005_basecall_1d_folder_structure(self):
        """Test basecall 1d folder structure creation"""

        self.assertEqual(list(self.fh['/Analyses/Basecall_1D_000'].keys()), ['BaseCalled_template', 'Summary'])
        self.assertEqual(list(self.fh['/Analyses/Basecall_1D_000/BaseCalled_template'].keys()), ['Events', 'Fastq', 'Model'])

    def test_010_mapping_folder_structure(self):
        """Test mapping structure creation"""

        self.assertEqual(list(self.fh['/Analyses/Squiggle_Map_000'].keys()), ['SquiggleMapped_template', 'Summary'])
        self.assertEqual(list(self.fh['/Analyses/Squiggle_Map_000/SquiggleMapped_template'].keys()), ['Events', 'Model'])
        self.assertEqual(list(self.fh['/Analyses/Squiggle_Map_000/Summary'].keys()), ['squiggle_map_template'])

        self.assertEqual(list(self.fh['/Analyses/Squiggle_Map_001'].keys()), ['SquiggleMapped_template', 'Summary'])
        self.assertEqual(list(self.fh['/Analyses/Squiggle_Map_001/SquiggleMapped_template'].keys()), ['Events',  'Model'])
        self.assertEqual(list(self.fh['/Analyses/Squiggle_Map_001/Summary'].keys()), ['squiggle_map_template'])

    def test_015_fastq(self):
        """ Test fastq assembly and writing """

        fastq = '@unknown\n{}\n+\n{}\n'.format(self.seq, self.qstring)
        self.assertEqual(_sanitize_data_for_reading(self.fh['/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'][()]), fastq)

    def test_020_basecall_1d_event_writing(self):
        """Test basecall event writing"""

        input_events = self.events['mean']
        output_events = self.fh['/Analyses/Basecall_1D_000/BaseCalled_template/Events']['mean'][()]
        nptest.assert_array_equal(input_events, output_events)

    def test_025_basecall_1d_event_reading(self):
        """Test basecall event reading with the getter function"""

        input_events = self.events['mean']
        output_events = self.fh.get_basecall_data()['mean']
        nptest.assert_array_equal(input_events, output_events)

    def test_030_mapping_event_writing(self):
        """Test mapping event writing"""

        input_events = self.events['mean']
        output_events = self.fh['/Analyses/Squiggle_Map_000/SquiggleMapped_template/Events']['mean'][()]
        output_events_with_post = self.fh['/Analyses/Squiggle_Map_001/SquiggleMapped_template/Events']['mean'][()]

        nptest.assert_array_equal(input_events, output_events)
        nptest.assert_array_equal(input_events, output_events_with_post)

    def test_035_mapping_event_reading(self):
        """Test mapping event reading with the getter function"""

        input_means = self.events['mean']
        events = self.fh.get_mapping_data()
        nptest.assert_array_equal(input_means, events['mean'])
        self.assertEqual(events['kmer'].dtype, np.dtype('|U5'))

    def test_036_mapping_event_reading_any(self):
        """Test mapping event reading with the I don't care function"""

        input_means = self.events['mean']
        events = self.fh.get_mapping_data()
        nptest.assert_array_equal(input_means, events['mean'])
        self.assertEqual(events['kmer'].dtype, np.dtype('|U5'))


if __name__ == '__main__':
    unittest.main()
