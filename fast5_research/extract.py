import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import logging
import os

import h5py
import numpy as np

from fast5_research.fast5 import Fast5, iterate_fast5
from fast5_research.fast5_bulk import BulkFast5
from fast5_research.util import _sanitize_data_for_writing, readtsv


def extract_reads():
    logging.basicConfig(
        format='[%(asctime)s - %(name)s] %(message)s',
        datefmt='%H:%M:%S', level=logging.INFO
    )
    logger = logging.getLogger('Extract Reads')
    parser = argparse.ArgumentParser(description='Bulk .fast5 to read .fast5 conversion.')
    parser.add_argument('input', help='Bulk .fast5 file for input.')
    parser.add_argument('output', help='Output folder.')
    out_format = parser.add_mutually_exclusive_group()
    out_format.add_argument('--multi', action='store_true', help='Output multi-read files.')
    out_format.add_argument('--single', action='store_false', dest='multi', help='Output single-read files.')
    parser.add_argument('--flat', default=False, action='store_true',
                        help='Create all .fast5 files in one directory')
    parser.add_argument('--by_id', default=False, action='store_true',
                        help='Name single-read .fast5 files by read_id.')
    parser.add_argument('--prefix', default="", help='Read file prefix.')
    parser.add_argument('--channel_range', nargs=2, type=int, default=None, help='Channel range (inclusive).')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes.')
    parser.add_argument('--limit', type=int, default=None, help='Limit reads per channel.')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        raise IOError('The output directory must not exist.')

    worker = functools.partial(
        extract_channel_reads,
        args.input, args.output, args.prefix, args.flat, args.by_id,
        args.limit, args.multi
    )

    if args.channel_range is None:
        with BulkFast5(args.input) as src:
            channels = src.channels
    else:
        channels = range(args.channel_range[0], args.channel_range[1] + 1)

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as executor:
            futures = [executor.submit(worker, c) for c in channels]
            for future in as_completed(futures):
                try:
                    n_reads, channel = future.result()
                except Exception as e:
                    logger.warning("Error processing channel.")
                    print(e)
                else:
                    logger.info("Extracted {} reads from channel {}.".format(n_reads, channel))
    else:
        for channel in channels:
            worker(channel)
    logger.info("Finished.")


def extract_channel_reads(source, output, prefix, flat, by_id, max_files, multi, channel):
    if flat:
        out_path = output
        # give multi files a channel prefix else they will
        # conflict between channels. Singles already get
        # a "ch" component in their name
        if multi:
            extra = 'ch{}'.format(channel)
            if prefix == '':
                prefix = extra
            else:
                prefix = '{}_{}'.format(prefix, extra)
    else:
        out_path = os.path.join(output, str(channel))
        os.makedirs(out_path)

    with BulkFast5(source) as src:
        meta = src.get_metadata(channel)
        tracking_id = src.get_tracking_meta()
        context_tags = src.get_context_meta()
        channel_id = {
            'channel_number': channel,
            'range': meta['range'],
            'digitisation': meta['digitisation'],
            'offset': meta['offset'],
            'sampling_rate': meta['sample_rate']
        }

        Writer = MultiWriter if multi else SingleWriter
        with Writer(out_path, by_id, prefix=prefix) as writer:

            median_before = None
            counter = 1
            raw_data = src.get_raw(channel, use_scaling=False)
            for read_number, read in enumerate(src.get_reads(channel)):
                if median_before is None:
                    median_before = read['median']
                    continue

                if read['classification'] != 'strand':
                    median_before = read['median']
                else:
                    counter += 1
                    start, length = read['read_start'], read['read_length']
                    read_id = {
                        'start_time': read['read_start'],
                        'duration': read['read_length'],
                        'read_number': read_number,
                        'start_mux': src.get_mux(channel, raw_index=start, wells_only=True),
                        'read_id': read['read_id'],
                        'scaling_used': 1,
                        'median_before': median_before
                    }

                    raw_slice = raw_data[start:start+length]
                    read = Read(read_id, read_number, tracking_id, channel_id, context_tags, raw_slice)
                    writer.write_read(read)
                    if counter == max_files:
                        break
    return counter, channel


def filter_multi_reads():
    logging.basicConfig(
        format='[%(asctime)s - %(name)s] %(message)s',
        datefmt='%H:%M:%S', level=logging.INFO
    )
    logger = logging.getLogger('Filter')
    parser = argparse.ArgumentParser(description='Extract reads from multi-read .fast5 files.')
    parser.add_argument('input', help='Path to input multi-read .fast5 files.')
    parser.add_argument('output', help='Output folder.')
    parser.add_argument('filter', help='A .tsv file with column `read_id` defining required reads.')
    parser.add_argument('--tsv_field', default='read_id', help='Field name from `filter` file to obtain read IDs.')
    out_format = parser.add_mutually_exclusive_group()
    out_format.add_argument('--multi', action='store_true', default=True, help='Output multi-read files.')
    out_format.add_argument('--single', action='store_false', dest='multi', help='Output single-read files.')
    parser.add_argument('--prefix', default="", help='Read file prefix.')
    parser.add_argument('--recursive', action='store_true', help='Search recursively under `input` for source files.')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes.')
    #parser.add_argument('--limit', type=int, default=None, help='Limit reads per channel.')
    args = parser.parse_args()

    if not args.multi:
        raise NotImplementedError('Extraction of reads to single read files is on the TODO list.')

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        raise IOError('The output directory must not exist.')

    #TODO: could attempt to discover filenames from args.
    logger.info("Reading filter file.")
    read_table = readtsv(args.filter, fields=[args.tsv_field])
    required_reads = set(read_table[args.tsv_field])
    logger.info("Found {} reads in filter.".format(len(required_reads)))

    # grab list of source files
    logger.info("Searching for input files.")
    src_files = list(iterate_fast5(args.input, paths=True, recursive=args.recursive))
    n_files = len(src_files)

    logger.info("Finding reads within {} source files.".format(n_files))
    index_worker = functools.partial(reads_in_multi, filt=required_reads)
    read_index = dict()
    n_reads = 0
    with ProcessPoolExecutor(args.workers) as executor:
        i = 0
        for src_file, read_ids in zip(src_files, executor.map(index_worker, src_files, chunksize=10)):
            i += 1
            n_reads += len(read_ids)
            read_index[src_file] = read_ids
            if i % 10 == 0:
                logger.info("Indexed {}/{} files. {}/{} reads".format(i, n_files, n_reads, len(required_reads)))

        read_index = dict(zip(src_files, executor.map(index_worker, src_files, chunksize=10)))

    # We don't go via creating Read objects, copying the data verbatim
    # likely quicker and nothing should need the verification that the APIs
    # provide (garbage in, garbage out).
    logger.info("Extracting {} reads.".format(n_reads))
    if args.prefix != '':
        args.prefix = '{}_'.format(args.prefix)
    with MultiWriter(args.output, None, prefix=args.prefix) as writer:
        for src_file, read_ids in read_index.items():
            logger.info("Copying {} reads from {}.".format(
                len(read_ids), os.path.basename(src_file)
            ))
            with h5py.File(src_file, 'r') as src_fh:
                for read_id in read_ids:
                    writer.write_read(src_fh["read_{}".format(read_id)])
    logger.info("Done.")


def reads_in_multi(src, filt=None):
    """Get list of read IDs contained within a multi-read file.
    
    :param src: source file.
    :param filt: perform filtering by given set.
    :returns: set of read UUIDs (as string and recorded in hdf group name).
    """
    logger = logging.getLogger(os.path.splitext(os.path.basename(src))[0])
    logger.debug("Finding reads.")
    prefix = 'read_'
    with h5py.File(src, 'r') as fh:
        read_ids = set(grp[len(prefix):] for grp in fh if grp.startswith(prefix))
    logger.debug("Found {} reads.".format(len(read_ids)))
    if filt is not None:
        read_ids = read_ids.intersection(filt)
    logger.debug("Filtered to {} reads.".format(len(read_ids)))
    return read_ids


class Read(object):
    # Just a sketch to help interchange of format
    def __init__(self, read_id, read_number, tracking_id, channel_id, context_tags, raw):
        self.read_id = read_id
        self.read_number = read_number
        self.tracking_id = tracking_id
        self.channel_id = channel_id
        self.context_tags = context_tags
        self.raw = raw

        # ensure typing and required fields
        self.channel_id = Fast5.convert_channel_id(self.channel_id)
        self.tracking_id = Fast5.convert_tracking_id(self.tracking_id)


class ReadWriter(object):
    def __init__(self, out_path, by_id, prefix=""):
        self.out_path = out_path
        self.by_id = by_id
        if prefix != "":
            prefix = "{}_".format(prefix)
        self.prefix = prefix

    def write_read(self):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass


class SingleWriter(ReadWriter):
    def write_read(self, read):
        if self.by_id:
            filename = '{}.fast5'.format(read.read_id['read_id'])
        else:
            filename = '{}read_ch{}_file{}.fast5'.format(
                self.prefix, read.channel_id['channel_number'], read.read_number
            )
        filename = os.path.join(self.out_path, filename)
        with Fast5.New(filename, 'a', tracking_id=read.tracking_id, context_tags=read.context_tags, channel_id=read.channel_id) as h:
            h.set_raw(read.raw, meta=read.read_id, read_number=read.read_number)


MULTI_READ_FILE_VERSION = "2.0"

class MultiWriter(ReadWriter):
    def __init__(self, out_path, by_id, prefix="", reads_per_file=4000):
        super(MultiWriter, self).__init__(out_path, by_id, prefix=prefix)
        self.reads_per_file = reads_per_file
        self.current_reads = 0 # reads in open file, used to signal new file condition
        self.file_counter = 0
        self.current_file = None
        self.closed = False


    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


    def close(self):
        if isinstance(self.current_file, h5py.File):
            self.current_file.close()


    def write_read(self, read):
        """Write a read.

        :param read: either a `Read` object or an hdf group handle from a
            source multi-read file.
        """
        if self.closed:
            raise RuntimeError('Cannot write after closed.')

        if self.current_reads == 0:
            # start a new file
            self.close()
            filename = '{}mreads_file{}.fast5'.format(
                self.prefix, self.file_counter
            )
            filename = os.path.join(self.out_path, filename)
            self.current_file = h5py.File(filename, 'w')
            self.current_file.attrs[_sanitize_data_for_writing('file_version')] = _sanitize_data_for_writing("2.0")
            self.file_counter += 1

        # write data
        if isinstance(read, Read):
            self._write_read(read)
        elif isinstance(read, h5py.Group):
            self._copy_read_group(read)
        else:
            raise TypeError("Cannot write type {} to output file.")
        self.current_reads += 1

        # update 
        if self.current_reads == self.reads_per_file:
            self.current_reads = 0


    def _write_read(self, read):
        if read.raw.dtype != np.int16:
            raise TypeError('Raw data must be of type int16.')

        read_group = '/read_{}'.format(read.read_id['read_id'])
        Fast5._add_attrs_to_fh(self.current_file, {'run_id': read.tracking_id['run_id']}, read_group, convert=str)

        # add all attributes
        for grp_name in ('tracking_id', 'context_tags'):
            # spec has all of these as str
            data = getattr(read, grp_name)
            Fast5._add_attrs_to_fh(self.current_file, data, '{}/{}'.format(read_group, grp_name), convert=str)
        Fast5._add_attrs_to_fh(self.current_file, read.channel_id, '{}/channel_id'.format(read_group))

        # add the data (and some more attrs)
        data_path = '{}/Raw'.format(read_group)
        read_id = Fast5._convert_meta_times(read.read_id, read.channel_id['sampling_rate'])
        read_id = Fast5.convert_raw_meta(read_id)
        Fast5._add_attrs_to_fh(self.current_file, read_id, data_path)
        signal_path = '{}/Signal'.format(data_path)
        self.current_file.create_dataset(
            signal_path, data=read.raw, compression='gzip', shuffle=True, dtype='i2')


    def _copy_read_group(self, read):
        self.current_file.copy(read, read.name)
