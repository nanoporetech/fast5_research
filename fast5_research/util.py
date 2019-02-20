from copy import deepcopy
from itertools import tee
from math import pow, log10
import os
import sys

import numpy as np
import numpy.lib.recfunctions as nprf


def qstring_to_phred(quality):
    """Compute standard phred scores from a quality string."""
    qscores = [ord(q) - 33 for q in quality]
    return qscores


def mean_qscore(scores):
    """Returns the phred score corresponding to the mean of the probabilities
    associated with the phred scores provided. Taken from chimaera.common.utilities.


    :param scores: Iterable of phred scores.

    :returns: Phred score corresponding to the average error rate, as
        estimated from the input phred scores.
    """
    if len(scores) == 0:
        return 0.0
    sum_prob = 0.0
    for val in scores:
        sum_prob += pow(10, -0.1 * val)
    mean_prob = sum_prob / len(scores)
    return -10.0 * log10(mean_prob)


def kmer_overlap_gen(kmers, moves=None):
    """From a list of kmers return the character shifts between them.
    (Movement from i to i+1 entry, e.g. [AATC,ATCG] returns [0,1]).
    Allowed moves may be specified in moves argument in order of preference.
    Taken from dragonet.bio.seq_tools

    :param kmers: sequence of kmer strings.
    :param moves: allowed movements, if None all movements to length of kmer
        are allowed.
    """

    first = True
    yield 0
    for last_kmer, this_kmer in window(kmers, 2):
        if first:
            if moves is None:
                l = len(this_kmer)
                moves = range(l + 1)
            first = False

        l = len(this_kmer)
        for j in moves:
            if j < 0:
                if last_kmer[:j] == this_kmer[-j:]:
                    yield j
                    break
            elif j > 0 and j < l:
                if last_kmer[j:l] == this_kmer[0:-j]:
                    yield j
                    break
            elif j == 0:
                if last_kmer == this_kmer:
                    yield 0
                    break
            else:
                yield l
                break


def build_mapping_table(events, ref_seq, post, scale, path, model):
    """Build a mapping table based on output of a dragonet.mapper style object.
    Taken from chimaera.common.utilities.

    :param events: Numpy record array of events. Must contain the mean,
        stdv, start and length fields.
    :param ref_seq: String representation of the reference sequence.
    :param post: Numpy 2D array containing the posteriors (event, state).
    :param scale: Scaling object.
    :param path: Numpy 1D array containing position in reference. May contain
        negative values, which will be interpreted as "bad emissions".
    :param model: Model object to use.

    :returns: numpy record array containing summary fields. One record per event.

    ====================   =====================================================
    Output Field           Description
    ====================   =====================================================
    *mean*                 mean value of event samples (level)
    *scaled_mean*          *mean* scaled to the bare level emission (mean/mode)
    *stdv*                 standard deviation of event samples (noise)
    *scaled_stdv*          *stdv* scaled to the bare stdv emission (mode)
    *start*                start time of event /s
    *length*               length of event /s
    *model_level*          modelled event level, i.e. the level emission
                           associated with the kmer *kmer*, scaled to the data
    *model_scaled_level*   bare level emission
    *model_sd*             modelled event noise, i.e. the sd emission associated
                           with the kmer *kmer*, scaled  to the data
    *model_scaled_sd*      bare noise emission
    *seq_pos*              aligned sequence position, position on Viterbi path
    *p_seq_pos*            posterior probability of states on Viterbi path
    *kmer*                 kmer identity of *seq_pos*
    *mp_pos*               aligned sequence position, position with highest
                           posterioir
    *p_mp_pos*             posterior probability of most probable states
    *mp_kmer*              kmer identity of *mp_kmer*
    *good_emission*        whether or not the HMM has tagged event as fitting
                           the model
    ====================   =====================================================

    """
    kmer_len = len(model['kmer'][0])

    kmer_index = seq_to_kmers(ref_seq, kmer_len)
    label_index = dict((j,i) for i,j in enumerate(model['kmer']))
    kmer_dtype = '|S{}'.format(kmer_len)

    column_names = ['mean', 'scaled_mean', 'stdv', 'scaled_stdv', 'start', 'length',
                    'model_level', 'model_scaled_level', 'model_sd', 'model_scaled_sd',
                    'p_seq_pos', 'p_mp_pos', 'seq_pos', 'mp_pos', 'move', 'good_emission',
                    'kmer', 'mp_kmer']
    column_types = [float] * 12 + [int] * 3 + [bool] + [kmer_dtype] * 2
    table = np.zeros(events.size, dtype=list(zip(column_names, column_types)))

    zero_start = events['start'] - events['start'][0]

    # Sequence position
    seq_pos = np.where(path >= 0, path, np.abs(path) - 1)
    seq_kmer = [kmer_index[x] for x in seq_pos]
    seq_kmer_i = [label_index[i] for i in seq_kmer]

    table['seq_pos'] = seq_pos
    table['kmer'] = seq_kmer
    table['p_seq_pos'] = post[range(post.shape[0]), seq_pos]
    table['move'] = np.ediff1d(seq_pos, to_begin=[0])
    # Highest posterior positions
    mp_pos = np.argmax(post, axis=1)
    table['mp_pos'] = mp_pos
    table['mp_kmer'] = [kmer_index[x] for x in mp_pos]
    table['p_mp_pos'] = post[range(post.shape[0]), table['mp_pos']]
    # The data
    for x in ('mean', 'start','length', 'stdv'):
        table[x] = events[x]
    # scaling data to model
    table['scaled_mean'] = (table['mean'] - scale.shift - scale.drift * zero_start) / scale.scale
    table['scaled_stdv'] = table['stdv'] / scale.scale_sd
    # The model
    table['model_scaled_level'] = model['level_mean'][seq_kmer_i]
    table['model_scaled_sd']  = model['sd_mean'][seq_kmer_i]
    # The model scaled to the data
    table['model_level'] = scale.shift + scale.drift * zero_start + scale.scale * table['model_scaled_level']
    table['model_sd'] = scale.scale_sd * table['model_scaled_sd']
    # Tag ignore and outlier states
    table['good_emission'] = [x >= 0 for x in path]
    return table


def build_mapping_summary_table(mapping_summary):
    """Build a mapping summary table

    :param mapping_summary: List of curr_map dictionaries

    :returns: Numpy record array containing summary contents. One record per array element of mapping_summary

    """
    # Set memory allocation for variable length strings
    # This works, but there must be a better way
    max_len_name = 1
    max_len_direction = 1
    max_len_seq = 1
    for summary_line in mapping_summary:
        len_name = len(summary_line['name'])
        if len_name > max_len_name:
            max_len_name = len_name

        len_direction = len(summary_line['direction'])
        if len_direction > max_len_direction:
            max_len_direction = len_direction

        len_seq = len(summary_line['seq'])
        if len_seq > max_len_seq:
            max_len_seq = len_seq

    column_names = ['name', 'direction', 'is_best', 'score', 'scale', 'shift', 'drift', 'scale_sd', 'var_sd', 'var', 'seq']
    column_types = ['|S{}'.format(max_len_name)] + ['|S{}'.format(max_len_direction)] + [bool] + [float] * 7 + ['|S{}'.format(max_len_seq)]

    table = np.zeros(len(mapping_summary), dtype=list(zip(column_names, column_types)))
    for table_line, summary_line, in zip(table,mapping_summary):
        table_line['name'] = summary_line['name']
        table_line['direction'] = summary_line['direction']
        table_line['score'] = summary_line['score']
        table_line['scale'] = summary_line['scale'].scale
        table_line['shift'] = summary_line['scale'].shift
        table_line['drift'] = summary_line['scale'].drift
        table_line['scale_sd'] = summary_line['scale'].scale_sd
        table_line['var_sd'] = summary_line['scale'].var_sd
        table_line['var'] = summary_line['scale'].var
        table_line['seq'] = summary_line['seq']

    table['is_best'] = False
    is_best = np.argmin([line['score'] for line in mapping_summary])
    table[is_best]['is_best'] = True

    return table


def create_basecall_1d_output(raw_events, scale, path, model, post=None):
    """Create the annotated event table and basecalling summaries similiar to chimaera.

    :param raw_events: :class:`np.ndarray` with fields mean, stdv, start and,
        length fields.
    :param scale: :class:`dragonet.basecall.scaling.Scaler` object (or object
        with attributes `shift`, `scale`, `drift`, `var`, `scale_sd`, `var_sd`,
        and `var_sd`.
    :param path: list containing state indices with respect to `model`.
    :param model: `:class:dragonet.util.model.Model` object.
    :param post: Two-dimensional :class:`np.ndarray` containing posteriors (event, state).
    :param quality_data: :class:np.ndarray Array containing quality_data, used to annotate events.

    :returns: A tuple of:

        * the annotated input event table
        * a dict of result
    """

    events = raw_events.copy()
    model_state = np.array([model[x]['kmer'] for x in path])
    raw_model_level = np.array([model[x]['level_mean'] for x in path])
    move = np.array(list(kmer_overlap_gen(model_state)))
    counts = np.bincount(move)
    stays = counts[0]
    skips = counts[2] if len(counts) > 2 else 0

    # Extend the event table
    read_start = events[0]['start']
    model_level = scale.shift + scale.scale * raw_model_level +\
                  scale.drift * (events['start'] - read_start)
    new_columns = ['model_state', 'model_level', 'move']
    column_data = [model_state, model_level, move]

    if post is not None:
        weights = np.sum(post, axis=1)
        new_columns.append('weights')
        column_data.append(weights)

    drop_first = set(new_columns) & set(events.dtype.names)
    events = nprf.drop_fields(events, drop_first)
    table = nprf.append_fields(events, new_columns, data=column_data, asrecarray=True)

    # Compile the results
    results = {
        'num_events': events.size,
        'called_events': events.size,
        'shift': scale.shift,
        'scale': scale.scale,
        'drift': scale.drift,
        'var': scale.var,
        'scale_sd': scale.scale_sd,
        'var_sd': scale.var_sd,
        'num_stays': stays,
        'num_skips': skips
    }

    return table, results


def create_mapping_output(raw_events, scale, path, model, seq, post=None, n_states=None, is_reverse=False, substates=False):
    """Create the annotated event table and summaries similiar to chimaera

    :param raw_events: :class:`np.ndarray` with fields `mean`, `stdv`, `start`,
        and `length` fields.
    :param scale: :class:`dragonet.basecall.scaling.Scaler` object (or object
        with attributes `shift`, `scale`, `drift`, `var`, `scale_sd`, `var_sd`,
        and `var_sd`.
    :param path: list containing state indices with respect to `model`.
    :param model: `:class:dragonet.util.model.Model` object.
    :param seq: String representation of the reference sequence.
    :param post: Two-dimensional :class:`np.ndarray` containing posteriors (event, state).
    :param is_reverse: Mapping refers to '-' strand (bool).
    :param substate: Mapping contains substates?

    :returns: A tuple of:
        * the annotated input event table,
        * a dict of result.
    """

    events = raw_events.copy()
    direction = '+' if not is_reverse else '-'
    has_post = True

    # If we don't have a posterior, pass a mock object
    if post is None:
        if n_states is None:
            raise ValueError('n_states is required if post is None.')
        has_post = False
        post = MockZeroArray((len(events), n_states))
    table = build_mapping_table(events, seq, post, scale, path, model)

    # Delete mocked out columns
    if not has_post:
        to_delete = ['p_seq_pos', 'mp_pos', 'mp_kmer', 'p_mp_pos']
        table = nprf.drop_fields(table, to_delete)

    if direction == '-':
        events['seq_pos'] = len(seq) - table['seq_pos']
        ref_start = table['seq_pos'][-1]
        ref_stop = table['seq_pos'][0]
    else:
        ref_start = table['seq_pos'][0]
        ref_stop = table['seq_pos'][-1]

    # Compute movement stats.
    _, stays, skips = compute_movement_stats(path)

    results = {
        'direction': direction,
        'reference': seq,
        'ref_start': ref_start,
        'ref_stop': ref_stop,
        'shift': scale.shift,
        'scale': scale.scale,
        'drift': scale.drift,
        'var': scale.var,
        'scale_sd': scale.scale_sd,
        'var_sd': scale.var_sd,
        'num_stays': stays,
        'num_skips': skips
    }

    return table, results


class MockZeroArray(np.ndarray):
    def __init__(self, shape):
        """Mock enough of ndarray interface to be passable as a posterior matrix
        to chimaera build_mapping_table

        :param shape: tuple, shape of array
        """
        self.shape = shape

    def argmax(self, axis=0):
        return np.zeros(self.shape[1-axis], dtype=int)


def validate_event_table(table):
    """Check if an object contains all columns of a basic event array."""

    if not isinstance(table, np.ndarray):
        raise TypeError('Table is not a ndarray.')

    req_fields = ['mean', 'stdv', 'start', 'length']
    if not set(req_fields).issubset(table.dtype.names):
        raise KeyError(
            'Array does not contain fields for event array: {}, got {}.'.format(
                req_fields, table.dtype.names
            )
        )


def validate_model_table(table):
    """Check if an object contains all columns of a dragonet Model."""
    if not isinstance(table, np.ndarray):
        raise TypeError('Table is not a ndarray.')

    req_fields = ['kmer', 'level_mean', 'level_stdv', 'sd_mean', 'sd_stdv']
    if not set(req_fields).issubset(table.dtype.names):
        raise KeyError(
            'Object does not contain fields required for Model: {}, got {}.'.format(
                req_fields, table.dtype.names
            )
        )


def validate_scale_object(obj):
    """Check if an object contains all attributes of dragonet Scaler."""

    req_attributes = ['shift', 'scale', 'drift', 'var', 'scale_sd', 'var_sd']
    msg = 'Object does not contain attributes required for Scaler: {}'.format(req_attributes)
    assert all([hasattr(obj, attr) for attr in req_attributes]), msg


def compute_movement_stats(path):
    """Compute movement stats from a mapping state path

    :param path: :class:`np.ndarry` containing position in reference.
        Negative values are interpreted as "bad emissions".
    """

    vitstate_indices = np.where(path >= 0, path, np.abs(path) - 1)
    move = np.ediff1d(vitstate_indices, to_begin=0)
    counts = np.bincount(move)
    stays = counts[0]
    skips = counts[2] if len(counts) > 2 else 0

    return move, stays, skips


def seq_to_kmers(seq, length):
    """Turn a string into a list of (overlapping) kmers.

    e.g. perform the transformation:

    'ATATGCG' => ['ATA','TAT', 'ATG', 'TGC', 'GCG']

    :param seq: character string
    :param length: length of kmers in output

    :returns: A list of overlapping kmers
    """
    return [seq[x:x + length] for x in range(0, len(seq) - length + 1)]


def window(iterable, size):
    """Create an iterator returning a sliding window from another iterator

    :param iterable: Iterator
    :param size: Size of window

    :returns: an iterator returning a tuple containing the data in the window

    """
    assert size > 0, "Window size for iterator should be strictly positive, got {0}".format(size)
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return list(zip(*iters))


def readtsv(fname, fields=None, **kwargs):
    """Read a tsv file into a numpy array with required field checking

    :param fname: filename to read. If the filename extension is
        gz or bz2, the file is first decompressed.
    :param fields: list of required fields.
    """

    if not file_has_fields(fname, fields):
        raise KeyError('File {} does not contain requested required fields {}'.format(fname, fields))

    for k in ['names', 'delimiter', 'dtype']:
        kwargs.pop(k, None)
    table = np.genfromtxt(fname, names=True, delimiter='\t', dtype=None, encoding='utf8', **kwargs)
    #  Numpy tricks to force single element to be array of one row
    return table.reshape(-1)


def docstring_parameter(*sub):
    """Allow docstrings to contain parameters."""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


def med_mad(data, factor=None, axis=None, keepdims=False):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median

    :param data: A :class:`ndarray` object
    :param factor: Factor to scale MAD by. Default (None) is to be consistent
        with the standard deviation of a normal distribution
        (i.e. mad( N(0,\sigma^2) ) = \sigma).
    :param axis: For multidimensional arrays, which axis to calculate over
    :param keepdims: If True, axis is kept as dimension of length 1

    :returns: a tuple containing the median and MAD of the data

    """
    if factor is None:
        factor = 1.4826
    dmed = np.median(data, axis=axis, keepdims=True)
    dmad = factor * np.median(abs(data - dmed), axis=axis, keepdims=True)
    if axis is None:
        dmed = dmed.flatten()[0]
        dmad = dmad.flatten()[0]
    elif not keepdims:
        dmed = dmed.squeeze(axis)
        dmad = dmad.squeeze(axis)
    return dmed, dmad


def mad(data, factor=None, axis=None, keepdims=False):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and (by default)
    adjust by a factor for asymptotically normal consistency.

    :param data: A :class:`ndarray` object
    :param factor: Factor to scale MAD by. Default (None) is to be consistent
        with the standard deviation of a normal distribution
        (i.e. mad( N(0,\sigma^2) ) = \sigma).
    :param axis: For multidimensional arrays, which axis to calculate the median over.
    :param keepdims: If True, axis is kept as dimension of length 1

    :returns: the (scaled) MAD

    """
    _ , dmad = med_mad(data, factor=factor, axis=axis, keepdims=keepdims)
    return dmad


def file_has_fields(fname, fields=None):
    """Check that a tsv file has given fields

    :param fname: filename to read. If the filename extension is
        gz or bz2, the file is first decompressed.
    :param fields: list of required fields.

    :returns: boolean
    """

    # Allow a quick return
    req_fields = deepcopy(fields)
    if isinstance(req_fields, str):
        req_fields = [fields]
    if req_fields is None or len(req_fields) == 0:
        return True
    req_fields = set(req_fields)

    inspector = open
    ext = os.path.splitext(fname)[1]
    if ext == '.gz':
        inspector = gzopen
    elif ext == '.bz2':
        inspector = bzopen

    has_fields = None
    with inspector(fname, 'r') as fh:
        present_fields = set(fh.readline().rstrip('\n').split('\t'))
        has_fields = req_fields.issubset(present_fields)
    return has_fields


def get_changes(data, ignore_cols=None, use_cols=None):
    """Return only rows of a structured array which are not equal to the previous row.

    :param data: Numpy record array.
    :param ignore_cols: iterable of column names to ignore in checking for equality between rows.
    :param use_cols: iterable of column names to include in checking for equality between rows (only used if ignore_cols is None).

    :returns: Numpy record array.
    """
    cols = list(data.dtype.names)
    if ignore_cols is not None:
        for col in ignore_cols:
            cols.remove(col)
    elif use_cols is not None:
        cols = list(use_cols)
    changed_inds = np.where(data[cols][1:] != data[cols][:-1])[0] + 1
    changed_inds = [0] + [i for i in changed_inds]
    return data[(changed_inds,)]


def _clean(value):
    """Convert numpy numeric types to their python equivalents."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'S':
            return np.char.decode(value).tolist()
        else:
            return value.tolist()
    elif type(value).__module__ == np.__name__:
        conversion = value.item()
        if sys.version_info.major == 3 and isinstance(conversion, bytes):
            conversion = conversion.decode()
        return conversion
    elif sys.version_info.major == 3 and isinstance(value, bytes):
        return value.decode()
    else:
        return value


def _clean_attrs(attrs):
    return {_clean(k): _clean(v) for k, v in attrs.items()}


def _sanitize_data_for_writing(data):
    if isinstance(data, str):
        return data.encode()
    elif isinstance(data, np.ndarray) and data.dtype.kind == np.dtype(np.unicode):
        return data.astype('S')
    elif isinstance(data, np.ndarray) and len(data.dtype) > 1:
        dtypes = dtype_descr(data)
        for index, entry in enumerate(dtypes):
            type_check = entry[1]
            if isinstance(type_check, tuple):
                # an enum?
                return data
            if type_check.startswith('<U'):
                # numpy.astype can't handle empty string datafields for some
                # reason, so we'll explicitly state that.
                if len(entry[1]) <= 2 or (len(entry[1]) == 3 and
                                          entry[1][2] == '0'):
                    raise TypeError('Empty datafield {} cannot be converted'
                                    ' by np.astype.'.format(entry[0]))
                dtypes[index] = (entry[0], '|S{}'.format(entry[1][2:]))
        return data.astype(dtypes)
    return data


def _sanitize_data_for_reading(data):
    if sys.version_info.major == 3:
        if isinstance(data, bytes):
            return data.decode()
        elif isinstance(data, np.ndarray) and data.dtype.kind == 'S':
            return np.char.decode(data)
        elif isinstance(data, np.ndarray) and len(data.dtype) > 1:
            dtypes = dtype_descr(data)
            for index, entry in enumerate(dtypes):
                type_check = entry[1]
                if isinstance(type_check, tuple):
                    # an enum?
                    return data
                if entry[1].startswith('|S'):
                    # numpy.astype can't handle empty datafields for some
                    # reason, so we'll explicitly state that.
                    if len(entry[1]) <= 2 or (len(entry[1]) == 3 and
                                              entry[1][2] == '0'):
                        raise TypeError('Empty datafield {} cannot be converted'
                                        ' by np.astype.'.format(entry[0]))
                    dtypes[index] = (entry[0], '<U{}'.format(entry[1][2:]))
            return data.astype(dtypes)
    return data


def dtype_descr(arr):
    """Get arr.dtype.descr
    Views of structured arrays in which columns have been re-ordered nolonger support arr.dtype.descr
    see https://github.com/numpy/numpy/commit/dd8a2a8e29b0dc85dca4d2964c92df3604acc212
    """
    try:
        return arr.dtype.descr
    except ValueError:
        return tuple([(n, arr.dtype[n].descr[0][1]) for n in arr.dtype.names])
