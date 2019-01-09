from copy import deepcopy
import fnmatch
from glob import iglob
import itertools
import os
import re
import random
import shutil
import subprocess
import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py

import numpy as np
import numpy.lib.recfunctions as nprf
import progressbar

from fast5_research.util import dtype_descr
from fast5_research.util import mad
from fast5_research.util import docstring_parameter
from fast5_research.util import readtsv

from fast5_research.util import validate_event_table, validate_model_table, validate_scale_object, _clean_attrs
from fast5_research.util import create_basecall_1d_output, create_mapping_output, mean_qscore, qstring_to_phred, _sanitize_data_for_writing, _sanitize_data_for_reading

warnings.simplefilter("always", DeprecationWarning)


class Fast5(h5py.File):
    """Class for grabbing data from single read fast5 files. Many attributes/
    groups are assumed to exist currently (we're concerned mainly with reading).
    Needs some development to make robust and for writing.

    """

    __base_analysis__ = '/Analyses'
    __event_detect_name__ = 'EventDetection'
    __raw_path__ = '/Raw/Reads'
    __raw_name_old__ = 'RawData'
    __raw_path_old__ = '{}/{}/'.format(__base_analysis__, __raw_name_old__)
    __raw_signal_path_old__ = '{}/Signal'.format(__raw_path_old__)
    __raw_meta_path_old__ = '{}/Meta'.format(__raw_path_old__)
    __channel_meta_path__ = '/UniqueGlobalKey/channel_id'
    __tracking_id_path__ = 'UniqueGlobalKey/tracking_id'
    __context_tags_path__ = 'UniqueGlobalKey/context_tags'

    __default_event_path__ = 'Reads'

    __default_basecall_1d_analysis__ = 'Basecall_1D'

    __default_seq_section__ = 'template'
    __default_basecall_fastq__ = 'BaseCalled_{}/Fastq'
    __default_basecall_1d_events__ = 'BaseCalled_{}/Events'
    __default_basecall_1d_model__ = 'BaseCalled_{}/Model'
    __default_basecall_1d_summary__ = 'Summary/basecall_1d_{}'

    __default_alignment_analysis__ = 'Alignment'

    __default_section__ = 'template'

    __default_mapping_analysis__ = 'Squiggle_Map'
    __default_mapping_events__ = 'SquiggleMapped_{}/Events'
    __default_mapping_model__ = 'SquiggleMapped_{}/Model'
    __default_mapping_summary__ = 'Summary/squiggle_map_{}'

    __default_substep_mapping_analysis__ = 'Substate_Map'
    __default_substep_mapping_events__ = '/Events'

    __default_basecall_mapping_analysis__ = 'AlignToRef'
    __default_basecall_mapping_events__ = 'CurrentSpaceMapped_{}/Events/'
    __default_basecall_mapping_model__ = 'CurrentSpaceMapped_{}/Model/'
    __default_basecall_mapping_summary__ = '/Summary/current_space_map_{}/' # under AlignToRef analysis
    __default_basecall_alignment_summary__ = '/Summary/genome_mapping_{}/' # under Alignment analysis

    __default_engine_state_path__ = '/EngineStates/'
    __temp_fields__ = ('heatsink', 'asic')


    def __init__(self, fname, read='r'):

        super(Fast5, self).__init__(fname, read)

        # Attach channel_meta as attributes, slightly redundant
        for k, v in _clean_attrs(self[self.__channel_meta_path__].attrs).items():
            setattr(self, k, v)
        # Backward compat.
        self.sample_rate = self.sampling_rate

        self.filename_short = os.path.splitext(os.path.basename(self.filename))[0]
        short_name_match = re.search(re.compile(r'ch\d+_file\d+'), self.filename_short)
        self.name_short = self.filename_short
        if short_name_match:
            self.name_short = short_name_match.group()


    @classmethod
    def New(cls, fname, read='w', tracking_id={}, context_tags={}, channel_id={}):
        """Construct a fresh single-read file, with meta data written to
        standard locations.

        """
        if read not in ('w', 'a'):
            raise IOError("New file can only be opened with 'a' or 'w' intent.")

        channel_id = cls.convert_channel_id(channel_id)
        tracking_id = cls.convert_tracking_id(tracking_id)


        # Start a new file, populate it with meta
        with h5py.File(fname, read) as h:
            h.attrs[_sanitize_data_for_writing('file_version')] = _sanitize_data_for_writing(1.0)
            for data, location in zip(
                [tracking_id, context_tags],
                [cls.__tracking_id_path__, cls.__context_tags_path__]
            ):
                # cjw: no idea why these must be str, just following ossetra
                cls._add_attrs_to_fh(h, data, location, convert=str)
            # These aren't forced to be str
            cls._add_attrs_to_fh(h, channel_id, cls.__channel_meta_path__)

        # return instance from new file
        return cls(fname, 'a')


    def _add_attrs(self, data, location, convert=None):
        """Convenience method for adding attrs to a possibly new group.
        :param data: dict of attrs to add
        :param location: hdf path
        :param convert: function to apply to all dictionary values
        """
        self._add_attrs_to_fh(self, data, location, convert=None)


    @staticmethod
    def _add_attrs_to_fh(fh, data, location, convert=None):
        """Implementation of _add_attrs as staticmethod. This allows
        functionality to be used in .New() constructor but is otherwise nasty!
        """
        if location not in fh:
            fh.create_group(location)
        attrs = fh[location].attrs
        for k, v in data.items():
            if convert is not None:
                attrs[_sanitize_data_for_writing(k)] = _sanitize_data_for_writing(convert(v))
            else:
                attrs[_sanitize_data_for_writing(k)] = _sanitize_data_for_writing(v)


    @staticmethod
    def convert_channel_id(channel_id):
        # channel_id: spec. requires these
        req_fields = {
                      'digitisation': np.dtype('f8'),
                      'offset': np.dtype('f8'),
                      'range': np.dtype('f8'),
                      'sampling_rate': np.dtype('f8'),
                      'channel_number': str,
        }
        if not set(req_fields).issubset(set(channel_id.keys())):
            raise KeyError(
                'channel_id does not contain required fields: {},\ngot {}.'.format(req_fields.keys(), channel_id.keys())
            )
        channel_id = _type_meta(channel_id, req_fields)
        return channel_id


    @staticmethod
    def convert_tracking_id(tracking_id):
        # tracking_id: spec. says this group should be present as string, says
        #   nothing about required keys, but certain software require the
        #   following minimal set
        req_fields = {
            # guppy
            'exp_start_time': np.dtype('U20'),  # e.g. '1970-01-01T00:00:00Z'
            'run_id': np.dtype('U32'),          #      '118ea414303a46d892603141b9cbd7b0'
            'flow_cell_id': np.dtype('U8'),     #      'FAH12345'
            # ...add others
        }
        if not set(req_fields).issubset(set(tracking_id.keys())):
            raise KeyError(
                'tracking_id does not contain required fields: {},\ngot {}.'.format(req_fields.keys(), tracking_id.keys())
            )
        tracking_id = _type_meta(tracking_id, req_fields)
        return tracking_id


    @staticmethod
    def convert_raw_meta(meta):
        req_keys = {'start_time': np.dtype('u8'),
                    'duration': np.dtype('u4'),
                    'read_number': np.dtype('i4'),
                    'start_mux': np.dtype('u1'),
                    'read_id': str,
                    'median_before': np.dtype('f8'),
        }
        meta = {k:v for k,v in meta.items() if k in req_keys}
        if len(meta.keys()) != len(req_keys):
            raise KeyError(
                'Raw meta data must contain keys: {}.'.format(req_keys.keys())
            )
        meta = _type_meta(meta, req_keys)
        return meta
    

    def _add_string_dataset(self, data, location):
        assert type(data) == str, 'Need to supply a string'
        self.create_dataset(location, data=_sanitize_data_for_writing(data))


    def _add_numpy_table(self, data, location):
        self.create_dataset(location, data=_sanitize_data_for_writing(data), compression=True)


    def _add_event_table(self, data, location):
        validate_event_table(data)
        self._add_numpy_table(data, location)


    def _join_path(self, *args):
        return '/'.join(args)


    @property
    def writable(self):
        """Can we write to the file."""
        if self.mode is 'r':
            return False
        else:
            return True


    def assert_writable(self):
        assert self.writable, "File not writable, opened with {}.".format(self.mode)


    @property
    def channel_meta(self):
        """Channel meta information as python dict"""
        return _clean_attrs(self[self.__channel_meta_path__].attrs)


    @property
    def tracking_id(self):
        """Tracking id meta information as python dict"""
        return _clean_attrs(self[self.__tracking_id_path__].attrs)


    @property
    def context_tags(self):
        """Context tags meta information as python dict"""
        return _clean_attrs(self[self.__context_tags_path__].attrs)


    @property
    def attributes(self):
        """Attributes for a read, assumes one read in file"""
        try:
            return _clean_attrs(self.get_read(group=True).attrs)
        except IndexError:
            return _clean_attrs(self.get_read(group=True, raw=True).attrs)


    def summary(self, rename=True, delete=True, scale=True):
        """A read summary, assumes one read in file"""
        to_rename = zip(
            ('start_mux', 'abasic_found', 'duration', 'median_before'),
            ('mux', 'abasic', 'strand_duration', 'pore_before')
        )
        to_delete = ('read_number', 'scaling_used')

        data = deepcopy(self.attributes)
        data['filename'] = os.path.basename(self.filename)
        data['run_id'] = self.tracking_id['run_id']
        data['channel'] = self.channel_meta['channel_number']
        if scale:
            data['duration'] /= self.channel_meta['sampling_rate']
            data['start_time'] /= self.channel_meta['sampling_rate']

        if rename:
            for i,j in to_rename:
                try:
                    data[j] = data[i]
                    del data[i]
                except KeyError:
                    pass
        if delete:
            for i in to_delete:
                try:
                    del data[i]
                except KeyError:
                    pass

        for key in data:
            if isinstance(data[key], float):
                data[key] = np.around(data[key], 4)

        return data


    def strip_analyses(self, keep=('{}_000'.format(__event_detect_name__), __raw_name_old__)):
        """Remove all analyses from file

        :param keep: whitelist of analysis groups to keep

        """
        analyses = self[self.__base_analysis__]
        for name in analyses.keys():
            if name not in keep:
                del analyses[name]


    def repack(self, pack_opts=''):
        """Run h5repack on the current file. Returns a fresh object."""
        path = os.path.abspath(self.filename)
        path_tmp = '{}.tmp'.format(path)
        mode = self.mode
        self.close()
        pack_opts = pack_opts.split()
        subprocess.call(['h5repack'] + pack_opts + [path, path_tmp])
        shutil.move(path_tmp, path)
        return Fast5(path, mode)


    ###
    # Extracting read event data

    def get_reads(self, group=False, raw=False, read_numbers=None):
        """Iterator across event data for all reads in file

        :param group: return hdf group rather than event data
        """
        if not raw:
            event_group = self.get_analysis_latest(self.__event_detect_name__)
            event_path = self._join_path(event_group, self.__default_event_path__)
            reads = self[event_path]
        else:
            try:
                reads = self[self.__raw_path__]
            except:
                yield self.get_raw()[0]

        if read_numbers is None:
            it = reads.keys()
        else:
            it = (k for k in reads.keys()
                  if _clean_attrs(reads[k].attrs)['read_number'] in read_numbers)

        if group == 'all':
            for read in it:
                yield reads[read], read
        elif group:
            for read in it:
                yield reads[read]
        else:
            for read in it:
                if not raw:
                    yield self._get_read_data(reads[read])
                else:
                    yield self._get_read_data_raw(reads[read])


    def get_read(self, group=False, raw=False, read_number=None):
        """Like get_reads, but only the first read in the file

        :param group: return hdf group rather than event/raw data
        """
        gen = None
        if read_number is None:
            gen = self.get_reads(group, raw)
        else:
            gen = self.get_reads(group, raw, read_numbers=[read_number])
        try:
            return(gen.next())
        except AttributeError:
            return(next(gen))


    def _get_read_data(self, read, indices=None):
        """Private accessor to read event data"""
        # We choose the following to always be floats
        float_fields = ('start', 'length', 'mean', 'stdv')

        events = read['Events']

        # We assume that if start is an int or uint the data is in samples
        #    else it is in seconds already.
        needs_scaling = False
        if events['start'].dtype.kind in ['i', 'u']:
            needs_scaling = True

        dtype = np.dtype([(
            d[0], 'float') if d[0] in float_fields else d
            for d in dtype_descr(events)
        ])
        data = None
        with events.astype(dtype):
            if indices is None:
                data = events[()]
            else:
                try:
                    data = events[indices[0]:indices[1]]
                except:
                    raise ValueError(
                        'Cannot retrieve events using {} as indices'.format(indices)
                    )

        # File spec mentions a read.attrs['scaling_used'] attribute,
        #    its not clear what this is. We'll ignore it and hope for
        #    the best.
        if needs_scaling:
            data['start'] /= self.sample_rate
            data['length'] /= self.sample_rate
        return _sanitize_data_for_reading(data)


    def _get_read_data_raw(self, read, indices=None, scale=True):
        """Private accessor to read raw data"""
        raw = read['Signal']
        dtype = float if scale else int

        data = None
        with raw.astype(dtype):
            if indices is None:
                data = raw[()]
            else:
                try:
                    data = raw[indices[0]:indices[1]]
                except:
                    raise ValueError(
                        'Cannot retrieve events using {} as indices'.format(indices)
                    )

        # Scale data to pA
        if scale:
            meta = self.channel_meta
            raw_unit = meta['range'] / meta['digitisation']
            data = (data + meta['offset']) * raw_unit
        return data


    @staticmethod
    def _convert_meta_times(meta, sample_rate):
        # Metadata should be written in samples (int), not seconds (float)
        if isinstance(meta['start_time'], float):
            meta['start_time'] = int(round(meta['start_time'] * sample_rate))
            meta['duration'] = int(round(meta['duration'] * sample_rate))
        return meta


    def set_read(self, data, meta):
        """Write event data to file

        :param data: event data
        :param meta: meta data to attach to read
        :param read_number: per-channel read counter
        """
        req_fields = [
            'start_time', 'duration', 'read_number',
            'start_mux', 'read_id', 'scaling_used', 'median_before'
        ]
        if not set(req_fields).issubset(meta.keys()):
            raise KeyError(
                'Read meta does not contain required fields: {}, got {}'.format(
                    req_fields, meta.keys()
                )
            )
        req_fields = ['start', 'length', 'mean', 'stdv']
        if not isinstance(data, np.ndarray):
            raise TypeError('Data is not ndarray.')
        if not set(req_fields).issubset(data.dtype.names):
            raise KeyError(
                'Read data does not contain required fields: {}, got {}.'.format(
                    req_fields, data.dtype.names
                )
            )

        event_group = self.get_analysis_new(self.__event_detect_name__)
        event_path = self._join_path(event_group, self.__default_event_path__)
        path = self._join_path(
            event_path, 'Read_{}'.format(meta['read_number'])
        )

        # Metadata should be written in samples (int), not seconds (float)
        meta = self._convert_meta_times(meta, self.sample_rate)
        self._add_attrs(meta, path)


        uint_fields = ('start', 'length')
        dtype = np.dtype([(
            d[0], 'uint32') if d[0] in uint_fields else d
            for d in dtype_descr(data)
        ])

        # (see _get_read_data()). If the data is not an int or uint
        #    we assume it is in seconds and scale appropriately
        if data['start'].dtype.kind not in ['i', 'u']:
            data['start'] *= self.sample_rate
            data['length'] *= self.sample_rate
        self._add_event_table(
            data.astype(dtype), self._join_path(path, 'Events')
        )


    def get_read_stats(self):
        """Combines stats based on events with output of .summary, assumes a
        one read file.

        """
        data = deepcopy(self.summary())
        read = self.get_read()
        n_events = len(read)
        q = np.percentile(read['mean'], [10, 50, 90])
        data['range_current'] = q[2] - q[0]
        data['median_current'] = q[1]
        data['num_events'] = n_events
        data['median_sd'] = np.median(read['stdv'])
        data['median_dwell'] = np.median(read['length'])
        data['sd_current'] = np.std(read['mean'])
        data['mad_current'] = mad(read['mean'])
        data['eps'] = data['num_events'] / data['strand_duration']
        return data

    ###
    # Raw Data

    @docstring_parameter(__raw_path_old__)
    def get_raw(self, scale=True):
        """Get raw data in file, might not be present.

        :param scale: Scale data to pA? (rather than ADC values)

        .. warning::
            This method is deprecated and should not be used, instead use
            .get_read(raw=True) to read both MinKnow conformant files
            and previous Tang files.
        """
        warnings.warn(
            "'Fast5.get_raw()' is deprecated, use 'Fast5.get_read(raw=True)'.",
            DeprecationWarning,
            stacklevel=2
        )
        try:
            raw = self[self.__raw_signal_path_old__]
            meta = _clean_attrs(self[self.__raw_meta_path_old__].attrs)
        except KeyError:
            raise KeyError('No raw data available.')

        raw_data = None
        if scale:
            raw_data = raw[()].astype('float')
            raw_unit = meta['range'] / meta['digitisation']
            raw_data = (raw_data + meta['offset']) * raw_unit
        else:
            raw_data = raw[()]
        return raw_data, meta['sample_rate']


    def set_raw(self, raw, meta=None, read_number=None):
        """Set the raw data in file.

        :param raw: raw data to add
        :param read_number: read number (as usually given in filename and
            contained within HDF paths, viz. Reads/Read_<>/). If not given
            attempts will be made to guess the number (assumes single read
            per file).
        """

        # Enforce raw is 16bit int, don't try to second guess if not
        if raw.dtype != np.int16:
            raise TypeError('Raw data must be of type int16.')

        # Attempt to guess read_number and meta from event detection group
        if read_number is None:
            exception = RuntimeError("'read_number' not given and cannot guess.")
            try:
                n_reads = sum(1 for _ in self.get_reads())
            except IndexError:
                # if no events present
                raise exception
            else:
                if n_reads == 1:
                    read_number = _clean_attrs(self.get_read(group=True).attrs)['read_number']
                    meta = _clean_attrs(self.get_read(group=True, read_number=read_number).attrs)
                else:
                    raise exception

        # Metadata should be written in samples (int), not seconds (float)
        meta = self._convert_meta_times(meta, self.sample_rate)
        # Ensure meta values are in correct type
        meta = self.convert_raw_meta(meta)

        # Check meta is same as that for event data, if any
        try:
            event_meta = _clean_attrs(self.get_read(group=True, read_number=read_number).attrs)
        except IndexError:
            pass
        else:
            if sum(meta[k] != event_meta[k] for k in meta.keys()) > 0:
                raise ValueError(
                    "Attempted to set raw meta data as {} "
                    "but event meta is {}".format(meta, event_meta)
                )
        # Good to go!
        read_path = self._join_path(self.__raw_path__, 'Read_{}'.format(read_number))
        data_path = self._join_path(read_path, 'Signal')
        self._add_attrs(meta, read_path)
        self[data_path] = raw


    def set_raw_old(self, raw, meta):
        """Set the raw data in file.

        :param raw: raw data to add
        :param meta: meta data dictionary

        .. warning::
            This method does not write raw data conforming to the Fast5
            specification. This class will currently still read data
            written by this method.
        """
        warnings.warn(
            ".set_raw() does not conform to the Fast5 spec.. Although this "
            "class will read data written by this method, other tools "
            "may fail to read the resultant file.",
            FutureWarning
        )

        # Verify meta conforms to our standard
        req_fields = ['range', 'digitisation', 'offset', 'sample_rate']
        if not set(req_fields).issubset(set(meta.keys())):
            raise KeyError(
                'raw meta data dictionary must contain {} fields.'.format(req_fields)
            )
        if abs(meta['sample_rate'] - self.sample_rate) > 1.0: # is this a sensible error?
            raise ValueError(
                'Tried to set raw data with sample rate {}, should be {}.'.format(
                    meta['sample_rate'], self.sample_rate
                )
            )

        self[self.__raw_signal_path_old__] = raw
        self._add_attrs(meta, self.__raw_meta_path_old__)


    ###
    # Analysis path resolution

    def get_analysis_latest(self, name):
        """Get group of latest (present) analysis with a given base path.

        :param name: Get the (full) path of newest analysis with a given base
            name.
        """
        try:
            return self._join_path(
                self.__base_analysis__,
                sorted(filter(
                    lambda x: name in x, self[self.__base_analysis__].keys()
                ))[-1]
            )
        except (IndexError, KeyError):
            raise IndexError('No analyses with name {} present.'.format(name))


    def get_analysis_new(self, name):
        """Get group path for new analysis with a given base name.

        :param name: desired analysis name
        """

        # Formatted as 'base/name_000'
        try:
            latest = self.get_analysis_latest(name)
            root, counter = latest.rsplit('_', 1)
            counter = int(counter) + 1
        except IndexError:
            # Nothing present
            root = self._join_path(
                self.__base_analysis__, name
            )
            counter = 0
        return '{}_{:03d}'.format(root, counter)


    def get_model(self, section=__default_section__, analysis=__default_mapping_analysis__):
        """Get model used for squiggle mapping"""
        base = self.get_analysis_latest(analysis)
        model_path = self._join_path(base, self.__default_mapping_model__.format(section))
        return self[model_path][()]

    # The remaining are methods to read and write data as chimaera produces
    #    It is necessarily all a bit nasty, but should provide a more
    #    consistent interface to the files. Paths are defaulted

    ###
    # Temperature etc.

    @docstring_parameter(__default_engine_state_path__)
    def get_engine_state(self, state, time=None):
        """Retrieve engine state from {}, either across the whole read
        (default) or at a given time.

        :param state: name of engine state
        :param time: time (in seconds) at which to retrieve temperature

        """
        location = self._join_path(
            self.__default_engine_state_path__, state
        )
        states = self[location][()]
        if time is None:
            return states
        else:
            i = np.searchsorted(states['time'], time) - 1
            return states[state][i]


    @docstring_parameter(__default_engine_state_path__, __temp_fields__)
    def get_temperature(self, time=None, field=__temp_fields__[0]):
        """Retrieve temperature data from {}, either across the whole read
        (default) or at a given time.

        :param time: time at which to get temperature
        :param field: one of {}

        """
        if field not in self.__temp_fields__:
            raise RuntimeError("'field' argument must be one of {}.".format(self.__temp_fields__))

        return self.get_engine_state('minion_{}_temperature'.format(field), time)


    def set_engine_state(self, data):
        """Set the engine state data.

        :param data: a 1D-array containing two fields, the first of which
            must be named 'time'. The name of the second field will be used
            to name the engine state and be used in the dataset path.
        """
        fields = data.dtype.names
        if fields[0] != 'time':
            raise ValueError("First field of engine state data must be 'time'.")
        if len(fields) != 2:
            raise ValueError("Engine state data must contain exactly two fields.")

        state = fields[1]
        location = self._join_path(
            self.__default_engine_state_path__, state
        )
        self[location] = data


    ###
    # Template/adapter splitting data
    __default_split_analysis__= 'Segment_Linear'
    __split_summary_location__ = '/Summary/split_adapter'

    @docstring_parameter(__base_analysis__)
    def set_split_data(self, data, analysis=__default_split_analysis__):
        """Write a dict containing split point data.

        :param data: `dict`-like object containing attrs to add
        :param analysis: Base analysis name (under {})

        .. warning::
            Not checking currently for required fields.
        """

        location = self._join_path(
            self.get_analysis_new(analysis), self.__split_summary_location__
        )
        self._add_attrs(data, location)


    @docstring_parameter(__base_analysis__)
    def get_split_data(self, analysis=__default_split_analysis__):
        """Get signal segmentation data.

        :param analysis: Base analysis name (under {})
        """

        def _inner(path=analysis):
            location = self._join_path(
                self.get_analysis_latest(path), 'Summary'
            )
            try:
                # there should be a single group under the above
                grps = list(self[location].keys())
                if len(grps) != 1:
                    raise
            except:
                raise IndexError('Cannot find location containing split point data.')
            location = self._join_path(location, grps[0])
            try:
                return _clean_attrs(self[location].attrs)
            except:
                raise ValueError(
                    'Could not retrieve signal split point data from attributes of {}'.format(location)
                )
        try:
            return _inner(analysis)
        except:
            # try a fallback location
            return _inner('Adapter_Split')


    @docstring_parameter(__base_analysis__)
    def get_section_indices(self, analysis=__default_split_analysis__):
        """Get two tuples indicating the event indices for signal
        segmentation boundaries.

        :param analysis: Base analysis path (under {})
        """

        # TODO: if the below fails, calculating the values on the fly would be
        #       a useful feature. Which brings to mind could we do such a thing
        #       in all cases of missing data? Probably not reasonble.
        attrs = self.get_split_data(analysis)
        try:
            return (
                (attrs['start_index_temp'], attrs['end_index_temp']),
                (attrs['start_index_comp'], attrs['end_index_comp'])
            )
        except:
            raise ValueError('Could not retrieve signal segmentation data.')


    @docstring_parameter(__base_analysis__)
    def get_section_events(self, section, analysis=__default_split_analysis__):
        """Get the event data for a signal section

        :param analysis: Base analysis path (under {})
        """

        indices = self.get_section_indices(analysis)
        read = self.get_read(group=True)
        events = None
        if section == 'template':
            events = self._get_read_data(read, indices[0])
        elif section == 'complement':
            events = self._get_read_data(read, indices[1])
        else:
            raise ValueError(
                '"section" parameter for fetching events must be "template" or "complement".'
            )
        return events


    ###
    # 1D Basecalling data

    @docstring_parameter(__base_analysis__)
    def set_basecall_data(self, events, scale, path, model, seq,
                          section=__default_section__, name='unknown',
                          post=None, score=None,
                          quality_data=None, qstring=None,
                          analysis=__default_basecall_1d_analysis__):
        """Create an annotated event table and 1D basecalling summary similiar
        to chimaera and add them to the fast5 file.

        :param events: Numpy record array of events. Must contain the mean,
            stdv, start and length fields.
        :param scale: Scaling object.
        :param path: Viterbi path containing model pointers (1D np.array).
        :param model: Model object.
        :param seq: Basecalled sequence string for fastq.
        :param section: String to use in paths, e.g. 'template'.
        :param name: Identifier string for fastq.
        :param post: Numpy 2D array containing the posteriors (event, state), used to annotate events.
        :param score: Quality value for the whole strand.
        :param quality_data: Numpy 2D array containing quality_data, used to annotate events.
        :param qstring: Quality string for fastq.
        :param analysis: Base analysis name (under {})
        """

        # Validate input
        self.assert_writable()
        validate_event_table(events)
        validate_scale_object(scale)
        validate_model_table(model)

        # Prepare paths
        base = self.get_analysis_new(analysis)
        fastq_path = self._join_path(base, self.__default_basecall_fastq__.format(section))
        events_path = self._join_path(base, self.__default_basecall_1d_events__.format(section))
        model_path = self._join_path(base, self.__default_basecall_1d_model__.format(section))
        summary_path = self._join_path(base, self.__default_basecall_1d_summary__.format(section))

        # Create annotated events and results
        annot_events, results = create_basecall_1d_output(events, scale, path, model, post=None)
        if seq is not None:
            results['sequence_length'] = len(seq)
        if score is not None:
            results['strand_score'] = score

        # Write fastq
        if qstring is None:
            qstring = '!' * len(seq)
        else:
            qscores = qstring_to_phred(qstring)
            results['mean_qscore'] = mean_qscore(qscores)

        fastq = '@{}\n{}\n+\n{}\n'.format(name, seq, qstring)
        self._add_string_dataset(fastq, fastq_path)

        # Write summary
        self._add_attrs(results, summary_path)

        # Write annotated events
        read_meta = {
            'start_time': annot_events[0]['start'],
            'duration': annot_events[-1]['start'] + annot_events[-1]['length'] - annot_events[0]['start']
        }
        self._add_event_table(annot_events, events_path)
        self._add_attrs(read_meta, events_path)

        # Write model
        scale_meta = {
            'shift': scale.shift,
            'scale': scale.scale,
            'drift': scale.drift,
            'var': scale.var,
            'scale_sd': scale.scale_sd,
            'var_sd': scale.var_sd
        }
        self._add_numpy_table(model, model_path)
        self._add_attrs(scale_meta, model_path)


    @docstring_parameter(__base_analysis__)
    def get_basecall_data(self, section=__default_section__, analysis=__default_basecall_1d_analysis__):
        """Read the annotated basecall_1D events from the fast5 file.

        :param section: String to use in paths, e.g. 'template'.
        :param analysis: Base analysis name (under {})
        """

        base = self.get_analysis_latest(analysis)
        events_path = self._join_path(base, self.__default_basecall_1d_events__.format(section))
        try:
            # use _get_read_data to make sure int fields are converted as needed
            return self._get_read_data({'Events': self[events_path]})
        except:
            raise ValueError('Could not retrieve basecall_1D data from {}'.format(events_path))


    @docstring_parameter(__base_analysis__)
    def get_alignment_attrs(self, section=__default_section__, analysis=__default_alignment_analysis__):
        """Read the annotated alignment meta data from the fast5 file.

        :param section: String to use in paths, e.g. 'template'.
        :param analysis: Base analysis name (under {})

        """

        attrs = None
        base = self.get_analysis_latest(analysis)
        attr_path = self._join_path(base,
            self.__default_basecall_alignment_summary__.format(section))
        try:
            attrs = _clean_attrs(self[attr_path].attrs)
        except:
            raise ValueError('Could not retrieve alignment attributes from {}'.format(attr_path))

        return attrs

    ###
    # Mapping data

    @docstring_parameter(__base_analysis__)
    def set_mapping_data(self, events, scale, path, model, seq, ref_name,
                         section=__default_section__,
                         post=None, score=None, is_reverse=False,
                         analysis=__default_mapping_analysis__):
        """Create an annotated event table and mapping summary similiar to
        chimaera and add them to the fast5 file.

        :param events: :class:`np.ndarray` of events. Must contain mean,
            stdv, start and length fields.
        :param scale: Scaling object.
        :param path: :class:`np.ndarray` containing position in reference.
            Negative values will be interpreted as "bad emissions".
        :param model: Model object to use.
        :param seq: String representation of the reference sequence.
        :param section: Section of strand, e.g. 'template'.
        :param name: Reference name.
        :param post: Two-dimensional :class:`np.ndarray` containing posteriors.
        :param score: Mapping quality score.
        :param is_reverse: Mapping refers to '-' strand (bool).
        :param analysis: Base analysis name (under {})
        """

        # Validate input
        self.assert_writable()
        validate_event_table(events)
        validate_model_table(model)
        validate_scale_object(scale)
        assert isinstance(seq, str), 'seq needs to be a str'

        # Create annotated events and results
        n_states = None
        if post is None:
            n_states = len(seq) - len(model['kmer'][0]) + 1
        annot_events, results = create_mapping_output(
            events, scale, path, model, seq, post=post, is_reverse=is_reverse, n_states=n_states
        )
        results['ref_name'] = ref_name
        if score is not None:
            results['strand_score'] = score

        # Create scale meta to be written with model
        scale_meta = {
            'shift': scale.shift,
            'scale': scale.scale,
            'drift': scale.drift,
            'var': scale.var,
            'scale_sd': scale.scale_sd,
            'var_sd': scale.var_sd
        }

        self._write_mapping_data(annot_events, results, model, scale_meta, section, analysis=analysis)


    def _write_mapping_data(self, annot_events, results, model, scale_meta, section,
                            analysis=__default_mapping_analysis__):

        # Prepare paths
        paths_dict = {
            'Squiggle_Map': (self.__default_mapping_analysis__, self.__default_mapping_events__,
                             self.__default_mapping_model__, self.__default_mapping_summary__),
            'AlignToRef': (self.__default_basecall_mapping_analysis__, self.__default_basecall_mapping_events__,
                           self.__default_basecall_mapping_model__, self.__default_basecall_mapping_summary__)
                      }

        base, event_path, model_path, summary_path = paths_dict[analysis]
        base = self.get_analysis_new(analysis)
        event_path = self._join_path(base, event_path.format(section))
        model_path = self._join_path(base, model_path.format(section))
        summary_path = self._join_path(base, summary_path.format(section))

        # Write annotated events
        read_meta = {
            'start_time': annot_events[0]['start'],
            'duration': annot_events[-1]['start'] + annot_events[-1]['length'] - annot_events[0]['start']
        }
        self._add_event_table(annot_events, event_path)
        self._add_attrs(read_meta, event_path)

        # Write model
        self._add_numpy_table(model,  model_path)
        self._add_attrs(scale_meta, model_path)

        # Write summary
        self._add_attrs(results, summary_path)


    @docstring_parameter(__base_analysis__)
    def get_mapping_data(self, section=__default_section__, analysis=__default_mapping_analysis__, get_model=False):
        """Read the annotated mapping events from the fast5 file.

        .. note::
            The seq_pos column for the events table returned from basecall_mapping is
            adjusted to be the genome position (consistent with squiggle_mapping)

        :param section: String to use in paths, e.g. 'template'.
        :param analysis: Base analysis name (under {}). For basecall mapping
            use analysis = 'AlignToRef'.
        """

        events = None
        if analysis == self.__default_mapping_analysis__:
            # squiggle_mapping
            base = self.get_analysis_latest(analysis)
            event_path = self._join_path(base, self.__default_mapping_events__.format(section))
            try:
                # use _get_read_data to make sure int fields are converted as needed
                events = self._get_read_data({'Events': self[event_path]})
            except:
                raise ValueError('Could not retrieve squiggle_mapping data from {}'.format(event_path))
            if get_model:
                model_path = self._join_path(base, self.__default_mapping_model__.format(section))
                try:
                    model = self[model_path][()]
                except:
                    raise ValueError('Could not retrieve squiggle_mapping model from {}'.format(model_path))

            attrs = self.get_mapping_attrs(section=section)

        elif analysis == self.__default_substep_mapping_analysis__:
            # substep mapping
            base = self.get_analysis_latest(analysis)
            event_path = self._join_path(base, self.__default_substep_mapping_events__.format(section))
            try:
                # use _get_read_data to make sure int fields are converted as needed
                events = self._get_read_data({'Events': self[event_path]})
            except:
                raise ValueError('Could not retrieve substep_mapping data from {}'.format(event_path))
            attrs=None
            if get_model:
                raise NotImplementedError('Retrieving substep model not implemented.')

        else:
            # basecall_mapping
            base = self.get_analysis_latest(analysis)
            event_path = self._join_path(base, self.__default_basecall_mapping_events__.format(section))
            try:
                # use _get_read_data to make sure int fields are converted as needed
                events = self._get_read_data({'Events': self[event_path]})
            except:
                raise ValueError('Could not retrieve basecall_mapping data from {}'.format(event_path))
            if get_model:
                model_path = self._join_path(base, self.__default_basecall_mapping_model__.format(section))
                try:
                    model = self[model_path][()]
                except:
                    raise ValueError('Could not retrieve squiggle_mapping model from {}'.format(model_path))

            # Modify seq_pos to be the actual genome position (consistent with squiggle_map)
            attrs = self.get_mapping_attrs(section=section, analysis=self.__default_alignment_analysis__)
            if attrs['direction'] == '+':
                events['seq_pos'] = events['seq_pos'] + attrs['ref_start']
            else:
                events['seq_pos'] = attrs['ref_stop'] - events['seq_pos']

        # add transition field
        if attrs:
            move = np.ediff1d(events['seq_pos'], to_begin=0)
            if attrs['direction'] == '-':
                move *= -1
            if 'move' not in events.dtype.names:
                events = nprf.append_fields(events, 'move', move)
            else:
                events['move'] = move

        if get_model:
            return events, model
        else:
            return events


    def get_any_mapping_data(self, section=__default_section__, attrs_only=False, get_model=False):
        """Convenience method for extracting whatever mapping data might be
        present, favouring squiggle_mapping output over basecall_mapping.

        :param section: (Probably) 'template'
        :param attrs_only: Use attrs_only=True to return mapping attributes without events

        :returns: the tuple (events, attrs) or attrs only
        """
        events = None
        attrs = None
        try:
            if not attrs_only:
                events = self.get_mapping_data(section=section, get_model=get_model)
            attrs = self.get_mapping_attrs(section=section)
        except Exception as e:
            try:
                if not attrs_only:
                    events = self.get_mapping_data(section=section,
                        analysis=self.__default_basecall_mapping_analysis__, get_model=get_model)
                attrs = self.get_mapping_attrs(section=section,
                    analysis=self.__default_alignment_analysis__)
            except Exception as e:
                raise ValueError(
                    "Cannot find any mapping data at paths I know about. "
                    "Consider using get_mapping_data() with analysis argument."
                )
        if not attrs_only:
            if get_model:
                return events[0], attrs, events[1]
            else:
                return events, attrs
        else:
            return attrs


    @docstring_parameter(__base_analysis__)
    def get_mapping_attrs(self, section=__default_section__, analysis=__default_mapping_analysis__):
        """Read the annotated mapping meta data from the fast5 file.
        Names which are inconsistent between squiggle_mapping and basecall_mapping are added to
        basecall_mapping (thus duplicating the attributes in basecall mapping).

        :param section: String to use in paths, e.g. 'template'.
        :param analysis: Base analysis name (under {})
                         For basecall mapping use analysis = 'Alignment'
        """

        attrs = None
        if analysis == self.__default_mapping_analysis__:
            # squiggle_mapping
            base = self.get_analysis_latest(analysis)
            attr_path = self._join_path(base, self.__default_mapping_summary__.format(section))
            try:
                attrs = _clean_attrs(self[attr_path].attrs)
            except:
                raise ValueError('Could not retrieve squiggle_mapping meta data from {}'.format(attr_path))
        else:
            # basecall_mapping

            # AligToRef attributes (set AlignToRef first so that Alignment attrs are not overwritten)
            base = self.get_analysis_latest(self.__default_basecall_mapping_analysis__)
            attr_path = self._join_path(base, self.__default_basecall_mapping_summary__.format(section))
            try:
                attrs = _clean_attrs(self[attr_path].attrs)
            except:
                raise ValueError('Could not retrieve basecall_mapping meta data from {}'.format(attr_path))

            # Rename some of the fields
            rename = [
                ('genome_start', 'ref_start'),
                ('genome_end', 'ref_stop'),
            ]
            for old, new in rename:
                attrs[new] = attrs.pop(old)

            # Alignment attributes
            base = self.get_analysis_latest(analysis)
            attr_path = self._join_path(
                base, self.__default_basecall_alignment_summary__.format(section))
            try:
                genome = _clean_attrs(self[attr_path].attrs)['genome']
            except:
                raise ValueError('Could not retrieve basecall_mapping genome field from {}'.format(attr_path))
            try:
                attrs['reference'] = (self.get_reference_fasta(section = section)).split('\n')[1]
            except:
                raise ValueError('Could not retrieve basecall_mapping fasta from Alignment analysis')

            # Add attributes with keys consistent with Squiggle_map
            rc = '_rc'
            is_rc = genome.endswith(rc)
            attrs['ref_name'] = genome[:-len(rc)] if is_rc else genome
            attrs['direction'] =  '-' if is_rc else '+'

        # Trim any other fields, the allowed are those produced by
        #   squiggle_mapping. We allow strand_score but do not require
        #   it since our writer does not require it.
        required = [
            'direction', 'ref_start', 'ref_stop', 'ref_name',
            'num_skips', 'num_stays', 'reference'
        ]
        additional = ['strand_score', 'shift', 'scale', 'drift', 'var', 'scale_sd', 'var_sd']
        keep = required + additional
        assert set(required).issubset(set(attrs)), 'Required mapping attributes not found'
        for key in (set(attrs) - set(keep)):
            del(attrs[key])

        return attrs

    ###
    # Sequence data

    @docstring_parameter(__base_analysis__)
    def get_fastq(self, analysis=__default_basecall_1d_analysis__, section=__default_seq_section__, custom=None):
        """Get the fastq (sequence) data.

        :param analysis: Base analysis name (under {})
        :param section: (Probably) 'template'
        :param custom: Custom hdf path overriding all of the above.
        """

        err_msg = 'Could not retrieve sequence data from {}'

        if custom is not None:
            location = custom
        else:
            location = self._join_path(
                self.get_analysis_latest(analysis), self.__default_basecall_fastq__.format(section)
            )
        try:
            fastq = self[location][()]
        except:
            # Did we get given section != template and no analysis, that's
            #    more than likely incorrect. Try alternative analysis
            if section != self.__default_seq_section__ and analysis == self.__default_basecall_1d_analysis__:
                location = self._join_path(
                    self.get_analysis_latest(self.__default_basecall_1d_analysis__),
                    self.__default_basecall_fastq__.format(section)
                )
                try:
                    fastq = self[location][()]
                except:
                    raise ValueError(err_msg.format(location))
            else:
                raise ValueError(err_msg.format(location))
        else:
            fastq = _sanitize_data_for_reading(fastq)

        return fastq

    @docstring_parameter(__base_analysis__)
    def get_sam(self, analysis=__default_alignment_analysis__, section=__default_seq_section__, custom=None):
        """Get SAM (alignment) data.

        :param analysis: Base analysis name (under {})
        :param section: (Probably) 'template'
        :param custom: Custom hdf path overriding all of the above.
        """

        if custom is not None:
            location = custom
        else:
            location = self._join_path(
                self.get_analysis_latest(analysis), 'Aligned_{}'.format(section), 'SAM'
            )
        try:
            return self[location][()]
        except:
            raise ValueError('Could not retrieve SAM data from {}'.format(location))


    @docstring_parameter(__base_analysis__)
    def get_reference_fasta(self, analysis=__default_alignment_analysis__, section=__default_seq_section__, custom=None):
        """Get fasta sequence of known DNA fragment for the read.

        :param analysis: Base analysis name (under {})
        :param section: (Probably) 'template'
        :param custom: Custom hdf path overriding all of the above.
        """

        if custom is not None:
            location = custom
        else:
            location = self._join_path(
                self.get_analysis_latest(analysis), 'Aligned_{}'.format(section), 'Fasta'
            )
        try:
            sequence = _sanitize_data_for_reading(self[location][()])
        except:
            raise ValueError('Could not retrieve sequence data from {}'.format(location))

        return sequence


def recursive_glob(treeroot, pattern):
    # Emulates e.g. glob.glob("**/*.fast5"", recursive=True) in python3
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        for f in goodfiles:
            yield os.path.join(base, f)


def iterate_fast5(path='Stream', strand_list=None, paths=False, mode='r',
                  limit=None, shuffle=False, robust=False, progress=False,
                  recursive=False):
    """Iterate over directory of fast5 files, optionally only returning those in list

    :param path: Directory in which single read fast5 are located or filename.
    :param strand_list: List of strands, can be a python list of delimited
        table. If the later and a filename field is present, this is used
        to locate files. If a file is given and a strand field is present,
        the directory index file is searched for and filenames built from that.
    :param paths: Yield file paths instead of fast5 objects.
    :param mode: Mode for opening files.
    :param limit: Limit number of files to consider.
    :param shuffle: Shuffle files to randomize yield of files.
    :param robust: Carry on with iterating over FAST5 files after an exception was raised.
    :param progress: Display progress bar.
    :param recursive: Perform a recursive search for files in subdirectories of `path`.
    """
    if strand_list is None:
        #  Could make glob more specific to filename pattern expected
        if os.path.isdir(path):
            if recursive:
                files = recursive_glob(path, '*.fast5')
            else:
                files = iglob(os.path.join(path, '*.fast5'))
        else:
            files = [path]
    else:
        if isinstance(strand_list, list):
            files = (os.path.join(path, x) for x in strand_list)
        else:
            reads = readtsv(strand_list)
            if 'filename' in reads.dtype.names:
                #  Strand list contains a filename column
                files = (os.path.join(path, x) for x in reads['filename'])
            else:
                raise KeyError("Strand file does not contain required field 'filename'.\n")

    # shuffle means we can't be lazy
    if shuffle and limit is not None:
        files = np.random.choice(list(files), limit, replace=False)
    elif shuffle:
        random.shuffle(list(files))
    elif limit is not None:
        try:
            files = files[:limit]
        except TypeError:
            files = itertools.islice(files, limit)

    if progress:
        bar = progressbar.ProgressBar()
        files = bar(files)

    for f in files:
        if not os.path.exists(f):
            sys.stderr.write('File {} does not exist, skipping\n'.format(f))
            continue
        if not paths:
            try:
                fh = Fast5(f, read=mode)
            except Exception as e:
                if robust:
                    sys.stderr.write("Could not open FAST5 file {}: {}\n".format(f, e))
                else:
                    raise e
            else:
                yield fh
                fh.close()
        else:
            yield os.path.abspath(f)


def _type_meta(meta, types):
    """Convert meta data fields into required types.
    :param meta: dict of values
    :param: types: dict of types or `np.dtype`

    :returns: dict
    """

    converted = {k:v for k,v in meta.items()}
    for k, dtype in types.items():
        if k in meta:
            if isinstance(dtype, np.dtype):
                val = np.array([meta[k]], dtype=dtype)[0]
            else:
                val = dtype(meta[k])
            converted[k] = val
    return converted
