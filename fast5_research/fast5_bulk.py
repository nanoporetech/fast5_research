import ast
from collections import defaultdict
from fast5_research.util import dtype_descr
import itertools
import re
from sys import version_info
from xml.dom import minidom
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py

import numpy as np
from numpy.lib.recfunctions import append_fields


from fast5_research.util import get_changes, _clean_attrs, _sanitize_data_for_writing, _sanitize_data_for_reading

if version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


class BulkFast5(h5py.File):
    """Class for reading data from a bulk fast5 file"""

    __tracking_path__ = '/UniqueGlobalKey/tracking_id'
    __pore_model_old__ = 'Meta/User/pore_model'
    __pore_model_new__ = 'Meta/User/analysis_conf'
    __context_path__ = '/UniqueGlobalKey/context_tags/'
    __intermediate_data__ = '/IntermediateData/'
    __voltage_meta__ = '/Device/VoltageMeta'
    __voltage_data__ = '/Device/MetaData'
    __channel_meta__ = '/IntermediateData/Channel_{}/Meta'
    __multiplex_data__ = '/MultiplexData/Channel_{}/Multiplex'

    __raw_data__ = "Raw/Channel_{}/Signal"
    __raw_meta__ = "Raw/Channel_{}/Meta"
    __event_data__ = "/IntermediateData/Channel_{}/Events"
    __read_data__ = "/IntermediateData/Channel_{}/Reads"
    __state_data__ = "/StateData/Channel_{}/States"

    # The below refers to MinION Mk1 ASIC, may change in future
    __mk1_asic_mux_states__ = {
        'common_voltage_1': 1,
        'common_voltage_2': 2,
        'common_voltage_3': 3,
        'common_voltage_4': 4,
        'gnd': 15,
        'gnd_through_resistor': 14,
        'open_pore': 0,
        'test_current_1': 10,
        'test_current_2': 11,
        'test_current_3': 12,
        'test_current_4': 13,
        'test_current_open_pore': 5,
        'unblock_voltage_1': 6,
        'unblock_voltage_2': 7,
        'unblock_voltage_3': 8,
        'unblock_voltage_4': 9
    }

    def __init__(self, filename, mode='r'):
        """Create an BulkFast5 instance.

        :param filename: path to a bulk fast5 file.
        :param mode: h5py opening mode.
        """

        super(BulkFast5, self).__init__(filename, mode)
        if mode == 'r':
            data = self[self.__intermediate_data__]
            self.channels = sorted([int(name.strip('Channel_')) for name in data.keys()])
            self.parsed_exp_history = None # we parse the history lazily

            # Parse experimental metadata
            self.exp_metadata = dict()
            for path in (self.__tracking_path__, self.__context_path__):
                try:
                    self.exp_metadata.update(_clean_attrs(self[path].attrs))
                except KeyError:
                    raise KeyError('Cannot read summary from {}'.format(path))

            # This should be safe
            try:
                self.sample_rate = float(self['Meta'].attrs['sample_rate'])
            except:
                self.sample_rate = float(self.get_metadata(self.channels[0])['sample_rate'])


    def get_metadata(self, channel):
        """Get the metadata for the specified channel.

        Look for first for events metadata, and fall-back on raw metadata, returning an empty dict if neither could be found."""
        if hasattr(self, '_cached_metadata'):
            if channel in self._cached_metadata:
                return self._cached_metadata[channel]
        else:
            self._cached_metadata = {}

        if self.__channel_meta__.format(channel) in self:
            meta = _clean_attrs(self[self.__channel_meta__.format(channel)].attrs)
        elif self.has_raw(channel): # use raw meta data
            meta = _clean_attrs(self[self.__raw_meta__.format(channel)].attrs)
        else:
            meta = {}

        self._cached_metadata[channel] = meta
        return meta


    def get_event_detection_parameters(self):
        """Get the full set of parameters related to event detection """
        if self.__pore_model_old__ in self:   # Old Minknow file
            xmldoc = minidom.parseString("".join(self[self.__pore_model_old__].value))
            return dict(xmldoc.getElementsByTagName('event_detection')[0].attributes.items())
        elif self.__pore_model_new__ in self:  # New Minknow file
            result = "".join(self[self.__pore_model_new__].value)
            result = result.replace('true', 'True').replace('false', 'False')
            return ast.literal_eval(result)['event_detection']


    def get_tracking_meta(self):
        """Get tracking meta data"""
        return _clean_attrs(self[self.__tracking_path__].attrs)


    def get_context_meta(self):
        """Get context meta"""
        return _clean_attrs(self[self.__context_path__].attrs)


    def has_raw(self, channel):
        """Return True if there is raw data for this channel."""
        raw_location = self.__raw_data__.format(channel)
        return self._has_data(raw_location)


    def has_reads(self, channel):
        """Return True if there is read data for this channel."""
        read_location = self.__read_data__.format(channel)
        return self._has_data(read_location)


    def has_states(self, channel):
        """Return True if there is State data for this channel."""
        state_location = self.__state_data__.format(channel)
        return self._has_data(state_location)


    def _has_data(self, location):
        """Return true if the given data path exists

        :param location: str, path with fast5.
        """
        if hasattr(self, '_cached_paths'):
            if location in self._cached_paths:
                return self._cached_paths[location]
        else:
            self._cached_paths = {}

        location_split = location.split('/')
        folder = '/'.join(location_split[:-1])
        name = location_split[-1]
        present = folder in self and name in self[folder].keys()
        self._cached_paths[location] = present
        return present


    def _time_interval_to_index(self, channel, times):
        """Translate a tuple of (start_sec, end_sec) to an index."""
        start_sec, end_sec = times
        start = self._seconds_to_index(channel, start_sec)
        end = self._seconds_to_index(channel, end_sec)
        return (start, end)


    def _seconds_to_index(self, channel, time):
        """Translate a point in time to an index."""
        return int(time * float(self.sample_rate))


    def _scale(self, channel, data):
        """Scale event data if necessary, else return unchanged.

        If event metadata can't be found, assume events don't need scaling."""

        meta_data = self.get_metadata(channel)

        if 'scaling_used' not in meta_data or meta_data.get('scaling_used'):
            return data
        else:
            channel_scale = meta_data['range'] / meta_data['digitisation']
            channel_offset = meta_data['offset']
            data['mean'] = (data['mean'] + channel_offset) * channel_scale
            return data


    def get_raw(self, channel, times=None, raw_indices=(None, None), use_scaling=True):
        """If available, parse channel raw data.

        :param channel: channel number int
        :param times: tuple of floats (start_second, end_second)
        :param raw_indices: tuple of ints (start_index, end_index)
        :param use_scaling: if True, scale the current level

        .. note::
            Exactly one of the slice keyword arguments needs to be specified,
            as the method will override them in the order of times
            > raw_indices.
        """

        if not self.has_raw(channel):
            raise KeyError('Channel {} does not contain raw data.'.format(channel))

        if times is not None:
            raw_indices = self._time_interval_to_index(channel, times)

        raw_data = self.__raw_data__.format(channel)
        data = self[raw_data][raw_indices[0]:raw_indices[1]]

        if use_scaling:
            meta_data = self.get_metadata(channel)
            raw_unit = meta_data['range'] / meta_data['digitisation']
            data = (data + meta_data['offset']) * raw_unit

        return data


    def get_events(self, channel, times=None, raw_indices=None, event_indices=(None, None),
                   use_scaling=True):
        """Parse channel event data.

        :param channel: channel number int
        :param times: tuple of floats (start_second, end_second)
        :param raw_indices: tuple of ints (start_index, end_index)
        :param event_indices: tuple of ints (start_index, end_index)
        :param use_scaling: if True, scale the current level

        .. note::
            Exactly one of the slice keyword arguments needs to be specified,
            as the method will override them in the order of times
            > raw_indices > event_indices.
        """

        event_data = self.__event_data__.format(channel)
        ev = self[event_data]

        if times is not None:
            raw_indices = self._time_interval_to_index(channel, times)
        if raw_indices is not None:
            event_indices = np.searchsorted(ev['start'], raw_indices)
        data = _sanitize_data_for_reading(ev[event_indices[0]:event_indices[1]])

        # Change variance to stdv column
        data['variance'] = np.sqrt(data['variance'])
        data.dtype.names = ['stdv' if n == 'variance' else n for n in data.dtype.names]

        if use_scaling:
            return self._scale(channel, data)
        else:
            return data


    def _get_reads_data(self, channel):
        """Parse channel read data exactly as it is in the bulk fast5 file.

        :param channel: channel number int

        .. note::
            No processing is done - reads might span several rows.
        """
        if not self.has_reads(channel):
            raise KeyError('Channel {} does not contain read data.'.format(channel))

        return self[self.__read_data__.format(channel)]


    def get_reads(self, channel, transitions=False, penultimate_class=True):
        """Parse channel read data to yield details of reads.

        :param channel: channel number int
        :param transitions: if True, include transition reads
        :param penultimate_class: if True, for reads which span multiple rows,
                                 use the classification from the penultimate row.
         """

        read_data = self._get_reads_data(channel)

        return_keys = [
            'read_start', 'read_length',
            'event_index_start', 'event_index_end', 'classification', 'read_id',
            'median', 'median_sd', 'median_dwell', 'range', 'drift'
        ]
        additional_keys = ['flags']
        required_keys = return_keys + additional_keys

        for key in required_keys:
            if key not in read_data.dtype.names:
                raise KeyError('The read data did not contain the required key {}'.format(key))

        # classification is enumerated
        enum_map = h5py.check_dtype(enum=read_data.dtype['classification'])
        classes = _clean_attrs({v:k for k, v in enum_map.items()})
        # read dataset into memory, lest we return h5py objects
        read_data = read_data[()]

        # we need to combine 'event_index_start', 'read_start' from first row in
        # the read with sum 'read_length' over all rows and all other cols from
        # final row. If penultimate_class is True, use classification from penultimate row.
        accum_stats = None
        accum_names = ('event_index_start', 'read_start', 'read_length', 'classification')
        for n, row in enumerate(read_data):
            if accum_stats is None:
                accum_stats = {k:row[k] for k in accum_names}
            else:
                accum_stats['read_length'] += row['read_length']
                if penultimate_class:  # use classification from previous row
                    accum_stats['classification'] = read_data[n - 1]['classification']
                else:  # use classification from current row
                    accum_stats['classification'] = row['classification']

            # pick out only the columns we want
            row_details = {k:row[k] for k in return_keys}

            if row['flags'] & 0x1 == 0:
                # read has ended
                if classes[row['classification']] == 'transition' and not transitions:
                    accum_stats = None  # prepare for next read
                else:
                    for k in accum_stats:  # replace
                        row_details[k] = accum_stats[k]
                    row_details['classification'] = classes[row_details['classification']]
                    yield _clean_attrs(row_details)
                    accum_stats = None


    def get_state_changes(self, channel):
        """Parse channel state changes.

        :param channel: channel number int
        """
        if not self.has_states(channel):
            raise KeyError('Channel {} does not contain state data.'.format(channel))

        if hasattr(self, '_cached_state_changes'):
            if channel in self._cached_state_changes:
                return self._cached_state_changes[channel]
        else:
            self._cached_state_changes = {}

        # state data is enumerated
        col = 'summary_state'
        data = self[self.__state_data__.format(channel)]
        enum_map = h5py.check_dtype(enum=data.dtype[col])
        enum_to_state = _clean_attrs({v:k for k, v in enum_map.items()})

        # translate ints into strings
        states = np.array([enum_to_state[key] for key in data[col]])

        try:
            data = np.array(data['approx_raw_index'])
        except ValueError: #not a KeyError: see h5py/_hl/dataset.pyc in readtime_dtype(basetype, names)
            data = np.array(data['acquisition_raw_index'])
        if len(data) > 0:
            data = data.astype([('approx_raw_index', data.dtype)], copy=False)
            data = append_fields(data,
                                ['approx_raw_index_end','summary_state'],
                                [np.roll(data['approx_raw_index'], -1), states], usemask=False)
            # set end of last state to something approximating infinity (the largest u64 int).
            data['approx_raw_index_end'][-1] = -1
        else:  # some channels don't contain channel state data, just create a dummy array
            data =  np.array([],  dtype=[('approx_raw_index', '<u8'),
                                         ('approx_raw_index_end', '<u8'),
                                         ('summary_state', 'S28')])
        self._cached_state_changes[channel] = data
        return _sanitize_data_for_reading(data)


    def get_state(self, channel, raw_index=None, time=None):
        """Find the channel state at a given time

        :param channel: channel number int
        :param raw_index: sample index
        :param time: time in seconds

        .. note::
            Exactly one of the slice keyword arguments needs to be specified,
            as the method will override them in the order of times
            > raw_indices.
        """
        assert (time is not None) or (raw_index is not None), 'Need either a time or a raw_index argument'
        if time is not None:
            raw_index = self._seconds_to_index(channel, time)

        data = self.get_state_changes(channel)

        # Check if the requested index is before the first state entry
        if raw_index < data['approx_raw_index'][0]:
            msg = 'No state data at index {}, which is before first state at {}'
            raise RuntimeError(msg.format(raw_index, data['approx_raw_index'][0]))

        # Now get last record before requested sample, handling the special case
        # where there is no last record (i.e. if raw_index == 0)
        if raw_index == 0:
            i = 0
        else:
            i = np.searchsorted(data['approx_raw_index'], raw_index) - 1

        state = data['summary_state'][i]

        return state


    def get_states_in_window(self, channel, times=None, raw_indices=None):
        """Find all channel states within a time window.

        :param channel: channel number int
        :param times: tuple of floats (start_second, end_second)
        :param raw_indices: tuple of ints (start_index, end_index)

        .. note::
            Exactly one of the slice keyword arguments needs to be specified,
            as the method will override them in the order of times
            > raw_indices.
        """

        assert (times is not None) or (raw_indices is not None), 'Need either a time or a raw_index argument'
        if times is not None:
            raw_indices = self._seconds_to_index(channel, times[0]), self._seconds_to_index(channel, times[1])
        states = self.get_state_changes(channel)
        first_state, last_state = np.searchsorted(states['approx_raw_index'], raw_indices, side='right')
        return np.unique(states['summary_state'][first_state-1:last_state])


    def get_mux(self, channel, raw_index=None, time=None, wells_only=False, return_raw_index=False):
        """Find the multiplex well_id ("the mux") at a given time

        :param channel: channel number int
        :param raw_index: sample index
        :param time: time in seconds
        :wells_only: bool, if True, ignore changes to mux states not in [1,2,3,4]
                     and hence return the last well mux.
        :return_raw_index: bool, if True, return tuple (mux, raw_index), raw_index being
                     raw index when the mux was set.

        .. note::
            There are multiple mux states associated with each well (e.g. common_voltage_1 and unblock_volage_1).
            Here, we return the well_id associated with the mux state (using self.enum_to_mux), i.e. 1 in both these cases.

            Exactly one of the slice keyword arguments needs to be specified,
            as the method will override them in the order of times
            > raw_indices.
        """
        assert (time is not None) or (raw_index is not None), 'Need either a time or a raw_index argument'
        if time is not None:
            raw_index = self._seconds_to_index(channel, time)

        data = self.get_mux_changes(channel, wells_only=wells_only)

        # Check if the requested index is before the first mux entry
        if raw_index < data['approx_raw_index'][0]:
            msg = 'No mux data at index {}, which is before first mux at {}'
            raise RuntimeError(msg.format(raw_index, data['approx_raw_index'][0]))

        # Now get last record before requested sample, handling the special case
        # where there is no last record (i.e. if raw_index == 0)
        if raw_index == 0:
            i = 0
        else:
            i = np.searchsorted(data['approx_raw_index'], raw_index) - 1

        mux = self.enum_to_mux[data[i]['well_id']]

        if return_raw_index:
            raw_index = data[i]['approx_raw_index']  # when the mux was set
            return mux, raw_index
        else:
            return mux


    @staticmethod
    def _strip_metadata(data):
        """Strip dtype.metadata dicts from enumerated arrays.

        :param data: structured np.array
        :returns: view of the same data with the metadata removed.

        .. note::
            since h5py v 2.3, enumerated dtypes come with a dtype.metadata dict
            see https://github.com/numpy/numpy/issues/6771 and
            https://github.com/h5py/h5py/pull/355/commits/5da2e96942218ffb1c9b614be9be8409bea219f8
            This can stop functions like recfunctions.append_fields working on
            these arrays, so strip out this dict. as it's not writeable, just
            create a view with the appropriate data type
        """
        d = []
        for col, str_type in dtype_descr(data):
            if not isinstance(str_type, str) and isinstance(str_type[1], dict) and 'enum' in str_type[1]:
                str_type = str_type[0]
            d.append((col, str_type))
        return data.view(np.dtype(d))


    def get_mux_changes(self, channel, wells_only=False):
        """Get changes in multiplex settings for given channel.

        :param channel: channel for which to fetch data
        :wells_only: bool, if True, ignore changes to mux states not in [1,2,3,4]

        .. note::
            There are multiple mux states associated with each well (e.g. 1:common_voltage_1 and 6:unblock_voltage_1).
            Here, we return mux state numbers, e.g. 1 and 6, which can be linked to the well_id using self.enum_to_mux
        """
        if hasattr(self, '_cached_mux_changes'):
            if channel in self._cached_mux_changes[wells_only]:
                return self._cached_mux_changes[wells_only][channel]
        else:
            # cache mux changes separately for well_only True and False
            self._cached_mux_changes = {True: {}, False: {}}

        enum_col = 'well_id'
        multiplex_data = self.__multiplex_data__.format(channel)
        data = self[multiplex_data]
        enum = _clean_attrs(h5py.check_dtype(enum=data.dtype[enum_col]))
        assert enum == self.__mk1_asic_mux_states__, 'Got unexpected multiplex states'

        if not hasattr(self, "enum_to_mux"):
            # Build a dict which relates enum values to mux.
            self.enum_to_mux = {}
            for k, v in enum.items():
                mux = 0
                mo = re.search(r'(\d)$', k)
                if mo is not None:
                    mux = int(mo.group(0))
                self.enum_to_mux[v] = mux
        data = data[()]  # load into memory
        data = self._strip_metadata(data)  # remove dtype.metadata dict present with h5py>=2.3.0

        # remove any rows where the mux state has not changed
        data = get_changes(data, ignore_cols=('approx_raw_index',))

        if wells_only:  # only consider changes to wells in [1,2,3,4]
            wells = [1, 2, 3, 4]
            mask = np.in1d(data['well_id'], wells)
            mask[0] = True  # keep first mux, whatever it is
            data = data[mask]
        self._cached_mux_changes[wells_only][channel] = data
        return data


    def get_mux_changes_in_window(self, channel, times=None, raw_indices=None):
        """Find all mux changes within a time window.

        :param channel: channel number int
        :param times: tuple of floats (start_second, end_second)
        :param raw_indices: tuple of ints (start_index, end_index)

        .. note::
            There are multiple mux values associated with each well (e.g. 1:common_voltage_1 and 6:unblock_voltage_1).
            Here, we return mux values, e.g. 1 and 6, which can be linked to the well_id using self.enum_to_mux.

            Exactly one of the slice keyword arguments needs to be specified,
            as the method will override them in the order of times
            > raw_indices.
        """

        assert (times is not None) or (raw_indices is not None), 'Need either a time or a raw_index argument'
        if times is not None:
            raw_indices = self._seconds_to_index(channel, times[0]), self._seconds_to_index(channel, times[1])
        muxes = self.get_mux_changes(channel)
        first_mux, last_mux = np.searchsorted(muxes['approx_raw_index'], raw_indices, side='right')
        return muxes[first_mux-1:last_mux]


    def get_waveform_timings(self):
        """Extract the timings of the waveforms (if any).

         :returns: list of tuples of start and end times
        """
        mux_timings = []
        on_index = None
        for i in range(0, len(self["Device"]["AsicCommands"])):
            if self._waveform_enabled(i):
                on_index = self["Device"]["AsicCommands"][i]["frame_number"]
            elif on_index is not None:
                # when _waveform_enabled(i) returns to False, save on and off
                # timings
                off_index = self["Device"]["AsicCommands"][i]["frame_number"]
                on_time = on_index / self.sample_rate
                off_time = off_index / self.sample_rate
                mux_timings.append((on_time, off_time))
                on_index = None
        return mux_timings


    def _waveform_enabled(self, cmd_index):
        """Checks AsicCommand history to see if the waveform command was issued.

        .. note::
        Here is the relevant section of the engineering documentation.
        engineering documentation (July 2015 version)

        Settings from PC: 512 bytes
        1. Equals 17 otherwise FPGA drops the parcel
        2. Command for FPGA:
            =1 load configuration data in ASIC
            =2 begin reading data from ASIC
            =3 reset ASIC chip
            =5 load configuration and begin/continue reading - used for
               real-time re-loading ASIC configuration
        3.  4bit: enable zero supply voltage for Fan ('1'- Fan can be switched
                  off completely, '0'- Fan is always On)
            3bit: temperature control On/Off ('1' - On, '0' - Off)
            2-1 bits: Fan speed control ('00' - Off, '11' - On
                      (only when temperature control is off))
            0 bit: soft temperature control ('1' - On, '0' - Off)
        4.  0bit: On/Off ASIC analogue supply voltage ('0' - off, '1' - on)
        5.  ASIC clock: '000' - 64MHz, '001' - 128MHz, '010' - 32MHz,
                            '100' - 16MHz, '110' - 8MHz
        6.  3 bit: Enable ('1' - on, '0' - off) channel mapping (channel
                   sequence 0,1...510,511) for 512 channels mode
            2 bit: Enable ('1' - on, '0' - off) ASIC configuration update every
                   1ms with values for bias voltage from LUT
            1-0 bits: Number of channels from ASIC: '00' - 128ch,
                      '01'-256ch, '10' - 512ch
        """

        waveform_flag = self["Device"]["AsicCommands"][cmd_index]["command"].tostring()[5]
        # if cmd is not a bytestring, convert waveform flag to an integer. Needed for python2.x compatibility
        if not isinstance(waveform_flag, int):
            waveform_flag = ord(waveform_flag)
        waveform_enabled = waveform_flag & 4 != 0
        return waveform_enabled


    def get_voltage(self, times=None, raw_indices=(None, None), use_scaling=True):
        """Extracts raw common electrode trace

        :raw_indices: tuple of ints to limit section of voltage data loaded.
        :use_scaling: bool, whether to scale voltage data. If no scaling meta is found,
                      scale by -5 (as appropriate for MinION).
        :return: voltage as array (including 5x multiplyer for MinKnow)
        """
        if times is not None:
            raw_indices = self._time_interval_to_index(self.channels[0], times)

        voltages = self[self.__voltage_data__
                        ][raw_indices[0]:raw_indices[1]]['bias_voltage']
        if use_scaling:
            # fast5 converted from ABF files have a voltage meta section
            # containing scaling parameters
            if self.__voltage_meta__ in self:
                voltage_meta = _clean_attrs(self[self.__voltage_meta__].attrs)
                unit = voltage_meta['range'] / voltage_meta['digitisation']
                offset = voltage_meta['offset']
            else:
                # Assume MinION scaling of 5
                unit = -5
                offset = 0
            voltages = (voltages + offset) * unit

        return voltages


    def get_bias_voltage_changes(self):
        """Get changes in the bias voltage.

        .. note::
            For a long (-long-long) time the only logging of the common
            electrode voltage was the experimental history (accurate to one
            second). The addition of the voltage trace changed this, but this
            dataset is cumbersome. MinKnow 1.x(.3?) added the asic command
            history which is typically much shorter and therefore quicker to
            query. The bias voltage is numerously record. For MinION asics
            there is typically a -5X multiplier to convert the data into
            correct units with the sign people are used to.
        """
        if hasattr(self, '_cached_voltage_changes'):
            return self._cached_voltage_changes

        # First try the asic command, fallback to the experimental history,
        # and finally the voltage trace.
        try:
            self._cached_voltage_changes = self._bias_from_asic_commands()
        except:
            try:
                self._cached_voltage_changes = self._bias_from_exp_hist()
            except:
                try:
                    self._cached_voltage_changes = self._bias_from_voltages()
                except:
                    raise RuntimeError('Cannot parse voltage changes.')

        return self._cached_voltage_changes


    def _bias_from_voltages(self):
        """Extract voltage changes from the voltage trace data."""

        voltages = self.get_voltage()
        changes = np.where(voltages[:-1] != voltages[1:])[0]

        voltage_changes = np.empty(
            len(changes) + 1,
            dtype=[('time', float), ('set_bias_voltage', int)]
        )
        voltage_changes['time'][0] = voltages[0]
        voltage_changes['time'][1:] = changes
        voltage_changes['time'] /= self.sample_rate
        voltage_changes['set_bias_voltage'] = voltages[0]
        voltage_changes['set_bias_voltage'][1:] = voltages[changes]
        return voltage_changes


    def _bias_from_asic_commands(self):
        """Extract voltages in Asic commands, filtering to only changes."""

        all_voltages = [AsicBCommand(cmd).configuration.bias_voltage
            for cmd in self['/Device/AsicCommands']['command']
        ]
        all_frames = self['/Device/AsicCommands']['frame_number']

        prev_voltage = all_voltages[0]
        changes = [(all_frames[0], prev_voltage)]
        for frame, voltage in itertools.izip(all_frames[1:], all_voltages[1:]):
            if voltage != prev_voltage:
                changes.append((frame, voltage))

        voltage_changes = np.array(
            changes,
            dtype=[('time', float), ('set_bias_voltage', int)]
        )
        voltage_changes['time'] /= self.sample_rate
        voltage_changes['set_bias_voltage'] *= -5
        return voltage_changes


    def _bias_from_exp_hist(self):
        """Extract voltage changes from experimental history.

        ..note:: The experimental history is deprecated in MinKnow 1.3
        """
        if self.parsed_exp_history is None:
            self.parse_history()
        voltage_changes = self.parsed_exp_history['set_bias_voltage']
        voltage_changes['set_bias_voltage'] *= -1
        return voltage_changes


    def get_bias_voltage_changes_in_window(self, times=None, raw_indices=None):
        """Find all mux voltage changes within a time window.

        :param times: tuple of floats (start_second, end_second)
        :param raw_indices: tuple of ints (start_index, end_index)

        .. note::
            This is the bias voltage from the expt history (accurate to 1
            second), and will not include any changes in voltage related to
            waveforms. For the full voltage trace, use get_voltage.

            Exactly one of the slice keyword arguments needs to be specified,
            as the method will override them in the order of times
            > raw_indices.
        """

        assert (times is not None) or (raw_indices is not None), 'Need either a time or a raw_index argument'
        if times is None:
            times = float(raw_indices[0]) / self.sample_rate, float(raw_indices[1]) / self.sample_rate
        bias_voltage_changes = self.get_bias_voltage_changes()
        first_index, last_index = np.searchsorted(bias_voltage_changes['time'], times, side='right')
        return bias_voltage_changes[first_index:last_index]


    __engine_states__ = {
        'minion_asic_temperature': float,
        'minion_heatsink_temperature': float,
        'set_bias_voltage': float,
        'fan_speed': int
    }
    __temp_fields__ = ('heatsink', 'asic')


    def parse_history(self):
        """Parse the experimental history to pull out various environmental factors.
        The functions below are quite nasty, don't enquire too hard.
        """
        try:
            exph_fh = StringIO(str(self['Meta/User']['experimental_history'][:].tostring().decode()))
        except Exception:
            raise RuntimeError('Cannot read experimental_history from fast5')

        data = defaultdict(list)
        for item in self._iter_records(exph_fh):
            #item should contain 'time' and something else
            time = item['time']
            field, value = next((k, v) for k, v in item.items() if k != 'time')
            data[field].append((time, value))

        self.parsed_exp_history = {
            k:np.array(data[k], dtype=[('time', float), (k, self.__engine_states__[k])])
            for k in data.keys()
        }
        return self


    def get_engine_state(self, state, time=None):
        """Get changes in an engine state or the value of an engine
        state at a given time.

        :param state: the engine state to retrieve.
        :param time: the time at which to grab engine state.
        """
        if state not in self.__engine_states__:
            raise RuntimeError("'field' argument must be one of {}.".format(self.__engine_states__.keys()))

        if self.parsed_exp_history is None:
            self.parse_history()

        states = self.parsed_exp_history[state]
        if time is None:
            return states
        else:
            i = np.searchsorted(states['time'], time) - 1
            return states[state][i]


    def get_temperature(self, time=None, field=__temp_fields__[0]):
        if field not in self.__temp_fields__:
            raise RuntimeError("'field' argument must be one of {}.".format(self.__temp_fields__))

        return self.get_engine_state('minion_{}_temperature'.format(field), time)


    def _iter_records(self, exph_fh):
        """Parse an iterator over file-like object representing
        an experimental history.
        """
        for line in exph_fh:
            mo = re.match(r'.*:\s+Expt time: (\d+)s:? (.*)', line)
            if mo:
                time, msg = mo.groups()
                rec = self._parse_line(msg)
                if rec:
                    key, value = rec
                    yield {'time': int(time), key:value}


    def _parse_line(self, msg):
        """Check if a line of experimental history records
        a change in the engine state.
        """
        mo = re.match(r'Experimental EngineState: (.*)', msg)
        if mo:
            msg2 = mo.group(1)
            return self._parse_engine_state(msg2)


    def _parse_engine_state(self, msg):
        """Extract engine state and value from a line of
        experimental history.
        """
        mo = re.match(r'(\w+) is now (.*)', msg)
        if mo:
            key, value = mo.group(1), mo.group(2)
            if key in self.__engine_states__:
                return key, value


    def _add_attrs(self, data, location, convert=None):
        """Convenience method for adding attrs to a possibly new group.
        :param data: dict of attrs to add
        :param location: hdf path
        :param convert: function to apply to all dictionary values
        """
        self.__add_attrs(self, data, location, convert=None)


    @staticmethod
    def __add_attrs(self, data, location, convert=None):
        """Implementation of _add_attrs as staticmethod. This allows
        functionality to be used in .New() constructor but is otherwise nasty!
        """
        if location not in self:
            self.create_group(location)
        attrs = self[location].attrs
        for k, v in data.items():
            if convert is not None:
                attrs[_sanitize_data_for_writing(k)] = _sanitize_data_for_writing(convert(v))
            else:
                attrs[_sanitize_data_for_writing(k)] = _sanitize_data_for_writing(v)


    def _add_numpy_table(self, data, location):
        data = _sanitize_data_for_writing(data)
        self.create_dataset(location, data=data, compression=True)


    @classmethod
    def New(cls, fname, read='a', tracking_id={}, context_tags={}, channel_id={}):
        """Construct a fresh bulk file, with meta data written to
        standard locations. There is currently no checking this meta data.
        TODO: Add meta data checking.

        """

        # Start a new file, populate it with meta
        with h5py.File(fname, 'w') as h:
            h.attrs[_sanitize_data_for_writing('file_version')] = _sanitize_data_for_writing(1.0)
            for data, location in zip(
                [tracking_id, context_tags],
                [cls.__tracking_path__, cls.__context_path__]
            ):
                # see cjw's comment in fast5.py:
                # 'no idea why these must be str, just following ossetra'
                cls.__add_attrs(h, data, location, convert=str)

        # return instance from new file
        return cls(fname, read)


    def set_raw(self, raw, channel, meta=None):
        """Set the raw data in file.

        :param raw: raw data to add
        :param channel: channel number
        """
        req_keys = ['description', 'digitisation', 'offset', 'range',
                    'sample_rate']

        meta = {k:v for k,v in meta.items() if k in req_keys}
        if len(meta.keys()) != len(req_keys):
            raise KeyError(
                'Raw meta data must contain keys: {}.'.format(req_keys)
            )

        raw_folder = '/'.join(self.__raw_data__.format(channel).split('/')[:-1])
        raw_data_path = self.__raw_data__.format(channel)
        self._add_attrs(meta, raw_folder)
        self[raw_data_path] = raw


    def set_events(self, data, meta, channel):
        """Write event data to file

        :param data: event data
        :param meta: meta data to attach to read
        :param read_number: per-channel read counter
        """
        req_meta_keys = ['description', 'digitisation', 'offset', 'range',
                    'sample_rate']
        if not set(req_meta_keys).issubset(meta.keys()):
            raise KeyError(
                'Read meta does not contain required fields: {}, got {}'.format(
                    req_fields, meta.keys()
                )
            )
        req_event_fields = [
            'start', 'length', 'mean', 'variance'
        ]
        if not isinstance(data, np.ndarray):
            raise TypeError('Data is not ndarray.')

        # if data contains 'stdv', square this to get the variance
        # seemingly bulk fast5 files contain variance and not stdv, as
        # taking the sqrt would be slow on minknow.
        names = list(data.dtype.names)
        for i, name in enumerate(names):
            if name == 'stdv':
                names[i] = 'variance'
                data['stdv'] = np.square(data['stdv'])
        data.dtype.names = names

        if not set(req_event_fields).issubset(data.dtype.names):
            raise KeyError(
                'Read data does not contain required fields: {}, got {}.'.format(
                    req_event_fields, data.dtype.names
                )
            )

        event_meta_path = self.__channel_meta__.format(channel)
        self._add_attrs(meta, event_meta_path)

        uint_fields = ('start', 'length')
        dtype = np.dtype([(
            d[0], 'uint32') if d[0] in uint_fields else d
            for d in dtype_descr(data)
        ])

        # If the data is not an int or uint we assume it is in seconds and scale
        # appropriately
        if data['start'].dtype.kind not in ['i', 'u']:
            data['start'] *= meta['sample_rate']
            data['length'] *= meta['sample_rate']

        events_path = self.__event_data__.format(channel)
        self._add_numpy_table(
            data.astype(dtype), events_path
        )


    def set_voltage(self, data, meta):
        req_keys = ['description', 'digitisation', 'offset', 'range',
                    'sample_rate']
        meta = {k:v for k,v in meta.items() if k in req_keys}
        if len(meta.keys()) != len(req_keys):
            raise KeyError(
                'Raw meta data must contain keys: {}.'.format(req_keys)
            )

        self._add_attrs(meta, self.__voltage_meta__)
        dtype = np.dtype([('bias_voltage', np.int16)])
        self._add_numpy_table(
            data.astype(dtype, copy=False), self.__voltage_data__

        )


#
# Taken from minknow/asicb_command/__init__.py
#
class AsicBConfiguration(object):
    """Wrapper around the asicb configuration struct passed to the asicb over usb"""
    def __init__(self, config):
        self.data = str(config)
        # Interpret as bytes...
        self.bytes = np.frombuffer(self.data, dtype="u1")
        # ...with reverse bit order
        self.bits = np.unpackbits(self.bytes[::-1])[::-1].copy()


    @property
    def bias_voltage(self):
        val = self.int_at(129, 121)
        if val > 256:
            return 256 - val
        return val


    def active_mux(self, channel):
        """
        Gets the active mux for the specified channel
        :param channel: 0 based
        """
        first_bit_channel_0 = 211     # bit of mux state for channel 0
        mux_state_size = 4
        requested_channel_first_bit = first_bit_channel_0 + mux_state_size * channel
        return self.int_at(requested_channel_first_bit + mux_state_size - 1, requested_channel_first_bit)


    def int_at(self, start, end):
        bits = self.bits_at(start, end)
        num = 0
        for on in reversed(bits):
            num = num << 1
            if on:
                num |= 1
        return num


    def bits_at(self, start, end):
        return self.bits[end:start+1]


class AsicBCommand(object):
    """Wrapper around the asicb command structure"""
    def __init__(self, command):
        self.data = str(command)
        self._configuration = AsicBConfiguration(self.data[10:])
        self.bytes = np.frombuffer(self.data, dtype="u1")

        if self.bytes[0] != 17:
            raise Exception("Invalid command - magic byte was '{}', expected '17'"
                            .format(self.bytes[0]))


    @property
    def min_temperature(self):
        return self._bytes[7]


    @property
    def min_temperature(self):
        return self._bytes[8]


    @property
    def configuration(self):
        return self._configuration
