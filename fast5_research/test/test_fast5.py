import os
import tempfile
import types
import unittest
from uuid import uuid4

import h5py
import numpy as np

from fast5_research import Fast5

class Fast5API(unittest.TestCase):
    test_file = 'example_basecall_squiggle_mapping.fast5'

    def setUp(self):
        self.h = Fast5(os.path.join(
            os.path.dirname(__file__), 'data', self.test_file
        ))

        # Use to create new temp files
        self.tmp_events_float = np.array(
            [(0.0, 1.0, 10.0, 2.0)],
            dtype=[(x, 'float') for x in ['start','length', 'mean', 'stdv']]
        )
        self.tmp_events_int = np.array(
            [(0, 5000, 10.0, 2.0)],
            dtype=[
                ('start', 'uint32'), ('length', 'uint32'),
                ('mean', 'float'), ('stdv', 'float')
            ]
        )
        self.tmp_raw = np.ones(15, dtype=np.int16)

        self.tmp_channel_id = {
            'channel_number': 1,
            'range': 1.0,
            'digitisation': 1.0,
            'offset': 0.0,
            'sample_rate': 5000.0,
            'sampling_rate': 5000.0
        }
        self.tmp_read_id = {
            'start_time': 0.0,
            'duration': 1.0,
            'read_number': 1,
            'start_mux': 1,
            'read_id': str(uuid4()),
            'scaling_used': 1,
            'median_before': 0
        }
        self.tmp_tracking_id = {
            'exp_start_time': '1970-01-01T00:00:00Z',
            'run_id': str(uuid4()).replace('-',''),
            'flow_cell_id': 'FAH00000',
        }


    def tearDown(self):
        self.h.close()

    @classmethod
    def setUpClass(self):
        print('* Fast5 API')


    def test_000_basic_functions(self):
        # Just test an inherited member
        self.assertEqual(
            os.path.basename(self.h.filename), self.test_file,
            'Inherited member attribute not correct.'
        )

        # We shouldn't be writable by default
        self.assertFalse(self.h.writable, 'File is not non-writable by default.')

    def test_010_get_meta(self):
        self.assertSetEqual(
            set(self.h.attributes.keys()),
            {
             'scaling_used', 'median_before',
             'start_time', 'read_number',
             'abasic_found', 'duration', 'start_mux'
             },
            '.attributes does not contain expected fields.'
        )

        self.assertSetEqual(
            set(self.h.channel_meta.keys()),
            {
             'channel_number', 'range', 'offset',
             'digitisation', 'sampling_rate',
             },
            '.channel_meta does not contain expected fields.'
        )

        self.assertTrue(
            {
             'strand_duration', 'pore_before', 'abasic',
             'start_time', 'mux', 'channel', 'filename'
            }.issubset(self.h.summary().keys()),
            '.summary does not contain expected fields.'
        )

        # Duration and start_time should be int, not float (samples, not times)
        for key in ['duration', 'start_time']:
            self.assertIsInstance(
                self.h.attributes[key], int
            )

    def test_020_get_reads_et_al(self):
        reads = self.h.get_reads()
        try:
            read = reads.next()
        except AttributeError:
            read = next(reads)
        self.assertIsInstance(
            reads, types.GeneratorType,
            '.get_reads() does not give generator.'
        )
        self.assertIsInstance(
            read, np.ndarray,
            '.get_reads().next() does not give numpy array by default.'
        )
        self.assertSequenceEqual(
            read.dtype.names, ['start', 'length', 'mean', 'stdv'],
            '.get_reads().next() does not give "event data".'
        )
        reads = self.h.get_reads(group=True)
        try:
            read = reads.next()
        except AttributeError:
            read = next(reads)
        self.assertIsInstance(
            read, h5py._hl.group.Group,
            '.get_reads().next() does not give h5py group when asked.'
        )

    def test_030_analysis_locations(self):
        path = self.h.get_analysis_latest('Basecall_1D')
        self.assertEqual(
            '/Analyses/Basecall_1D_000', path,
            '.get_analysis_latest() does not return correct.'
        )

        path = self.h.get_analysis_new('Basecall_1D')
        self.assertEqual(
            '/Analyses/Basecall_1D_001', path,
            '.get_analysis_new() does not return correct.'
        )

    def test_040_split_data(self):
        indices = self.h.get_section_indices()
        self.assertIsInstance(
            indices, tuple,
            '.get_section_indices() does not give tuple'
        )

        for i in range(2):
            self.assertIsInstance(
                indices[i], tuple,
                '.get_section_indices() does not give tuple of tuple, item {}'.format(i)
            )

    def test_045_split_data_events(self):
        for section in ('template', 'complement'):
            read = self.h.get_section_events(section)
            self.assertIsInstance(
                read, np.ndarray,
                '.get_section_events({}) does not give numpy array by default.'.format(section)
            )


    def test_050_sequence_data(self):
        for section in ('template', 'complement'):
            call = self.h.get_fastq(
                'Basecall_1D', section
            )
            self.assertIsInstance(call, str, '{} call is not str.'.format(section))

        # Check ValueError raised when requesting absent data
        self.assertRaises(
            ValueError, self.h.get_fastq, 'Basecall_1D', '2D'
        )


    def test_060_construct_new_file_checks(self):
        tmp_file = os.path.join(tempfile.gettempdir(), str(uuid4()))

        with self.assertRaises(IOError):
            fh = Fast5.New(tmp_file, 'r')
            fh = Fast5.New(tmp_file, 'a', channel_id = self.tmp_channel_id)
            fh = Fast5.New(tmp_file, 'a', tracking_id=self.tmp_tracking_id)

        # This should be fine
        with Fast5.New(tmp_file, 'a', channel_id = self.tmp_channel_id, tracking_id=self.tmp_tracking_id) as h:
            h.set_read(self.tmp_events_float, self.tmp_read_id)


    def test_061_write_read_float_data(self):
        tmp_file = os.path.join(tempfile.gettempdir(), str(uuid4()))

        with Fast5.New(tmp_file, 'a', channel_id = self.tmp_channel_id, tracking_id=self.tmp_tracking_id) as h:
            h.set_read(self.tmp_events_float, self.tmp_read_id)

        # Metadata duration and start_time should be integers, not floats
        with Fast5(tmp_file, 'r') as h:
            for key in ['duration', 'start_time']:
                self.assertIsInstance(
                   h.attributes[key], int
            )


        with Fast5(tmp_file) as h:
            events = h.get_read()
            self.assertEqual(events['start'].dtype.descr[0][1], '<f8',
                'Writing float data did not give float data on read.'
            )
            actual = events['start'][0]
            expected = self.tmp_events_float['start'][0]
            self.assertEqual(actual, expected,
                'Write float, data on read not scaled correctly, got {} not {}'.format(
                    actual, expected
                )
            )

        os.unlink(tmp_file)

    def test_065_write_int_read_float_data(self):
        tmp_file = os.path.join(tempfile.gettempdir(), str(uuid4()))

        with Fast5.New(tmp_file, 'a', channel_id = self.tmp_channel_id, tracking_id=self.tmp_tracking_id) as h:
            h.set_read(self.tmp_events_int, self.tmp_read_id)

        with Fast5(tmp_file) as h:
            events = h.get_read()
            self.assertEqual(events['start'].dtype.descr[0][1], '<f8',
                'Writing uint data did not give float data on read.'
            )
            actual = events['start'][0]
            expected = self.tmp_events_float['start'][0]
            self.assertEqual(actual, expected,
                'Write unit, data on read not scaled correctly, got {} not {}'.format(
                    actual, expected
                )
            )

        os.unlink(tmp_file)

    def test_067_write_raw_data(self):
        tmp_file = os.path.join(tempfile.gettempdir(), str(uuid4()))
        with Fast5.New(tmp_file, 'a', channel_id = self.tmp_channel_id, tracking_id=self.tmp_tracking_id) as h:
            h.set_raw(self.tmp_raw, meta=self.tmp_read_id, read_number=1)

        with self.assertRaises(TypeError):
            with Fast5.New(tmp_file, 'a', channel_id = self.tmp_channel_id, tracking_id=self.tmp_tracking_id) as h:
                h.set_raw(self.tmp_raw.astype(float), meta=self.tmp_read_id, read_number=1)

    def test_070_reference_fasta(self):
        for section in ('template', 'complement'):
            call = self.h.get_reference_fasta('Alignment', section)
            self.assertIsInstance(call, str, '{} call is not str.'.format(section))

    def test_080_parse_temperature(self):
        temps = self.h.get_temperature()
        expected = np.array([(1.0, 32.75), (8.0, 32.88), (9.0, 32.75)],
            dtype=[('time', '<f8'), ('minion_heatsink_temperature', '<f8')]
        )
        self.assertTupleEqual(temps.dtype.names, expected.dtype.names)
        for field in expected.dtype.names:
            np.testing.assert_allclose(
                expected[field], temps[0:3][field])


if __name__ == "__main__":
    unittest.main()
