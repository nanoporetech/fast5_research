import numpy as np
import os
import unittest

from fast5_research import BulkFast5


class BulkFast5Test(unittest.TestCase):

    example_file = 'elec3_example.fast5'

    def setUp(self):
        self.filepath = os.path.join(
            os.path.dirname(__file__), 'data', self.example_file
        )
        self.fh = BulkFast5(self.filepath)

    def tearDown(self):
        self.fh.close()

    @classmethod
    def setUpClass(self):
        print('\n* Bulk Fast5')

    # tests specific to this file

    def test_parse_experimental_metadata(self):
        """Test if the experiment metadata is parsed correctly."""

        expected = {'asic_id': '3503485095',
                    'asic_id_17': '61607',
                    'asic_id_eeprom': '0',
                    'asic_temp': '34.09',
                    'department': 'research',
                    'device_id': 'MN15971',
                    'exp_script_hash': 'e8b64319a2a8c0238d924a48d27b3ed433fa6e74',
                    'exp_script_name': './python/recipes/EXPERIMENT_RESEARCH_Static_Flips_Script_CP1_increase.py',
                    'exp_script_purpose': 'platform_qc',
                    'exp_start_time': '1465229980',
                    'experiment_type': 'chip_res',
                    'filename': 'minicol095_20160606_fnfad16824_mn15971_chip_res_jun03f02_elec0',
                    'flow_cell_id': 'FAD16824',
                    'heatsink_temp': '37.02',
                    'hostname': 'MINICOL095',
                    'protocol_run_id': '7ce62553-1ed5-42eb-aaf4-984efa212945',
                    'protocols_version_name': '0.51.1.69',
                    'rc_wiggle_test': 'true',
                    'read_classifier': 'platform_qc',
                    'read_classifier_hash': '82be2305a7bc4a86ad34cd734e05350a4f347d2c',
                    'read_classifier_is_ont_standard': '1',
                    'read_classifier_reference_hash': '82be2305a7bc4a86ad34cd734e05350a4f347d2c',
                    'run_id': '3aada3b43b81da733010adef6c89d18357fbd682',
                    'user_filename_input': 'jun03f02',
                    'version': '0.51.1.62 b201602101407',
                    'version_name': '0.51.1.62 b201602101407'}

        found = self.fh.exp_metadata
        del found['script_options']

        self.assertDictEqual(found, expected)

    def test_parse_temperature(self):
        temps = self.fh.get_temperature()
        expected = np.array([(2.0, 36.97), (6.0, 36.98), (8.0, 37.0)],
            dtype=[('time', '<f8'), ('minion_heatsink_temperature', '<f8')]
        )
        self.assertTupleEqual(temps.dtype.names, expected.dtype.names)
        for field in expected.dtype.names:
            np.testing.assert_allclose(
                expected[field], temps[0:3][field])

    def test_parse_waveform_timings(self):
        timings = self.fh.get_waveform_timings()
        self.assertEqual(len(timings), 8)
        expected_first = np.array((313.23200000000003, 607.77099999999996))
        np.testing.assert_allclose(expected_first, np.array(timings[0]))

    def test_raw_data_raises_exception_if_absent(self):
        """Test parsing raw data from a channel without raw data raises an exception."""

        with self.assertRaises(KeyError):
            self.fh.get_raw(1)

    def test_parse_raw_data(self):
        """Test parsing the whole raw dataset"""

        raw = self.fh.get_raw(self.fh.channels[0])
        self.assertEqual(len(raw), 1212525)

    def test_parse_event_data_len(self):
        """Test parsing the whole event dataset"""

        events = self.fh.get_events(self.fh.channels[0])
        self.assertEqual(len(events), 30710)

    def test_get_mux_changes(self):
        """Test parsing of mux changes"""
        mux_changes = list(self.fh.get_mux_changes(self.fh.channels[0]))
        self.assertEqual(len(mux_changes), 6)
        self.assertTupleEqual((3030000, 2), tuple(mux_changes[2]))
        # now test another channel - this might fail if caching has gone wrong
        mux_changes = list(self.fh.get_mux_changes(self.fh.channels[1]))
        self.assertTupleEqual((50000, 0), tuple(mux_changes[2]))

    # tests which have been designed to work for our elec3 example and converted
    # ABF file

    def test_parse_mux_by_time(self):
        """Test getting the mux for a time point"""

        mux = self.fh.get_mux(self.fh.channels[0], time=400)
        self.assertEqual(mux, 1)

    def test_parse_mux_by_raw_index(self):
        """Test getting the mux for a raw index"""

        mux = self.fh.get_mux(self.fh.channels[0], raw_index=400*self.fh.sample_rate)
        self.assertEqual(mux, 1)

    # general tests which should work for any file

    def test_parse_raw_data_by_index(self):
        """Test parsing the raw dataset sliced by indices"""

        raw = self.fh.get_raw(self.fh.channels[0], raw_indices=[100, 900])
        self.assertEqual(len(raw), 800)

    def test_parse_raw_data_by_time(self):
        """Test parsing the raw dataset sliced by times"""

        raw = self.fh.get_raw(self.fh.channels[0], times=[0, 0.05])
        self.assertEqual(len(raw), int(0.05 * self.fh.sample_rate))

    def test_parse_event_data_names(self):
        """Test parsing event data column names."""

        events = self.fh.get_events(self.fh.channels[0])
        for name in ['start', 'length', 'mean', 'stdv']:
            self.assertIn(name, list(events.dtype.names))

    def test_parse_event_data_by_raw_index(self):
        """Test parsing the event dataset sliced by raw indices"""

        events = self.fh.get_events(self.fh.channels[0])
        start = events[1]['start']
        end = events[4]['start']
        events = self.fh.get_events(self.fh.channels[0], raw_indices=[start, end])
        self.assertEqual(len(events), 3)

    def test_parse_event_data_by_raw_index(self):
        """Test parsing the event dataset sliced by event indices"""

        events = self.fh.get_events(self.fh.channels[0], event_indices=[1, 3])
        self.assertEqual(len(events), 2)

    def test_parse_event_data_by_time(self):
        """Test parsing the event dataset sliced by times"""

        events = self.fh.get_events(self.fh.channels[0])
        start = float(events[1]['start'] / self.fh.sample_rate)
        end = float(events[4]['start'] / self.fh.sample_rate)
        events = self.fh.get_events(self.fh.channels[0], times=[start, end])
        self.assertEqual(len(events), 3)

    def test_parse_read_data(self):
        """Test parsing the reads dataset """

        reads = list(self.fh.get_reads(self.fh.channels[0], transitions=False))
        self.assertEqual(len(reads), 953)
        reads = list(self.fh.get_reads(self.fh.channels[0], transitions=True, penultimate_class=False))
        self.assertEqual(len(reads), 967)

        # check a single-row read
        self.assertEqual(reads[1]['event_index_start'], 16)
        self.assertEqual(reads[1]['event_index_end'], 18)
        self.assertEqual(reads[1]['read_start'], 45060)
        self.assertEqual(reads[1]['read_length'], 1247)

        # check the more complicated case of a multi-row read which was stitched
        expected = {
            'classification': 'zero',
            'drift': 0.76819804090134713,
            'event_index_end': 5939,
            'event_index_start': 18,
            'median': -0.74220257568359393,
            'median_dwell': 78.0,
            'median_sd': 4.6702038236038623,
            'range': 36.423590332031246,
            'read_id': 'cac35512-2520-4a89-ac92-23db908ba45f',
            'read_length': 871039,
            'read_start': 46307
        }
        for key in ['event_index_start', 'event_index_end', 'classification',
                    'read_length', 'read_id']:
            self.assertEqual(reads[2][key], expected[key])
        for key in ['drift', 'median', 'median_dwell', 'range', 'read_start',
                    'median_sd']:
            self.assertAlmostEqual(reads[2][key], expected[key])

        # test the penultimate_class option
        reads = list(self.fh.get_reads(self.fh.channels[0], penultimate_class=True))
        self.assertEqual(reads[2]['classification'], 'user1')
        reads = list(self.fh.get_reads(self.fh.channels[0], penultimate_class=False))
        self.assertEqual(reads[2]['classification'], 'zero')

    def test_parse_voltage_by_index(self):
        """Test parsing the voltage dataset sliced by indices"""
        voltage = self.fh.get_voltage(raw_indices=[100, 900])
        self.assertEqual(len(voltage), 800)

    def test_parse_voltage_by_time(self):
        """Test parsing the voltage dataset sliced by times"""

        raw = self.fh.get_voltage(times=[0, 0.05])
        self.assertEqual(len(raw), int(0.05 * self.fh.sample_rate))

    def test_voltage_scaling(self):
        """Test scaling of the voltage"""
        voltage = self.fh.get_voltage()
        # find index of 1st non-zero voltage
        index = np.where(np.abs(voltage) > 0)[0][0]
        unscaled_voltage = self.fh.get_voltage(use_scaling=False)
        self.assertNotEqual(voltage[index], unscaled_voltage[index])

    def test_parse_state_data(self):
        """Test parsing of state data"""
        states = self.fh.get_state_changes(self.fh.channels[0])
        self.assertEqual(len(states), 43)

    def test_get_state_by_raw_index(self):
        """Test channel state at a give raw index"""

        state = self.fh.get_state(self.fh.channels[0], raw_index=100)
        self.assertEqual(state, 'unclassified')

        state = self.fh.get_state(self.fh.channels[0], raw_index=61000)
        self.assertEqual(state, 'inrange')

        # now test another channel - this might fail if caching has gone wrong
        state = self.fh.get_state(self.fh.channels[1], raw_index=61000)
        self.assertEqual(state, 'saturated')

    def test_get_state_by_time(self):
        """Test channel state at a give raw index"""
        state = self.fh.get_state(self.fh.channels[0], time=100/self.fh.sample_rate)
        self.assertEqual(state, 'unclassified')

        state = self.fh.get_state(self.fh.channels[0], time=61000/self.fh.sample_rate)
        self.assertEqual(state, 'inrange')

        state = self.fh.get_state(self.fh.channels[1], time=61000/self.fh.sample_rate)
        self.assertEqual(state, 'saturated')

    def test_get_states_in_window_by_raw_index(self):
        """Test get_states_in_window using a window specified in raw indices"""
        inds = (3045000, 3930001)
        states = self.fh.get_states_in_window(self.fh.channels[0], raw_indices=inds)
        expected = np.array(['above', 'inrange', 'unclassified_following_reset', 'unusable_pore'], dtype='U28')

        assert np.all(states == expected)
        states = self.fh.get_states_in_window(self.fh.channels[1], raw_indices=inds)
        expected = np.array(['above', 'inrange', 'unclassified_following_reset'], dtype='U28')
        assert np.all(states == expected)

    def test_get_states_in_window_by_times(self):
        """Test get_states_in_window using a window specified in times"""
        times = (3045000.0 / self.fh.sample_rate, 3930001.0 / self.fh.sample_rate)
        states = self.fh.get_states_in_window(self.fh.channels[0], times=times)
        expected = np.array(['above', 'inrange', 'unclassified_following_reset', 'unusable_pore'], dtype='U28')
        assert np.all(states == expected)
        states = self.fh.get_states_in_window(self.fh.channels[1], times=times)
        expected = np.array(['above', 'inrange', 'unclassified_following_reset'], dtype='U28')
        assert np.all(states == expected)


class BulkABFFast5Test(BulkFast5Test):

    example_file = 'abf2bulkfast5.fast5'

    def setUp(self):
        self.filepath = os.path.join(
            os.path.dirname(__file__), 'data', self.example_file
        )
        self.fh = BulkFast5(self.filepath)

    def tearDown(self):
        self.fh.close()

    @classmethod
    def setUpClass(self):
        print('\n* Bulk ABF Fast5')

    # tests to skip
    @unittest.skip("Skipping test_parse_experimental_metadata")
    def test_parse_experimental_metadata(self):
        pass

    @unittest.skip("Skipping test_parse_temperature")
    def test_parse_temperature(self):
        pass

    @unittest.skip("Skipping test_parse_waveform_timings")
    def test_parse_waveform_timings(self):
        pass

    @unittest.skip("Skipping test_parse_read_data")
    def test_parse_read_data(self):
        pass

    @unittest.skip("Skipping test_parse_state_data")
    def test_parse_state_data(self):
        """Test parsing of state data"""
        pass

    @unittest.skip("Skipping test_get_state_by_raw_index")
    def test_get_state_by_raw_index(self):
        """Test channel state at a give raw index"""
        pass

    @unittest.skip("Skipping test_get_state_by_time")
    def test_get_state_by_time(self):
        """Test channel state at a give raw index"""
        pass

    @unittest.skip("Skipping test_get_states_in_window_by_raw_index")
    def test_get_states_in_window_by_raw_index(self):
        """Test get_states_in_window using a window specified in raw indices"""
        pass

    @unittest.skip("Skipping test_get_states_in_window_by_times")
    def test_get_states_in_window_by_times(self):
        """Test get_states_in_window using a window specified in times"""
        pass

    def test_raw_data_raises_exception_if_absent(self):
        """Test parsing raw data from a channel without raw data raises an exception."""
        with self.assertRaises(KeyError):
            self.fh.get_raw(2)

    def test_parse_raw_data(self):
        """Test parsing the whole raw dataset"""
        raw = self.fh.get_raw(self.fh.channels[0])
        self.assertEqual(len(raw), 10000)

    def test_parse_event_data_len(self):
        """Test parsing the whole event dataset"""
        events = self.fh.get_events(self.fh.channels[0])
        self.assertEqual(len(events), 5)

    def test_get_mux_changes(self):
        """Test parsing of mux changes"""
        mux_changes = list(self.fh.get_mux_changes(self.fh.channels[0]))
        self.assertEqual(len(mux_changes), 1)
        self.assertTupleEqual((0, 1), tuple(mux_changes[0]))


if __name__ == "__main__":
    unittest.main()
