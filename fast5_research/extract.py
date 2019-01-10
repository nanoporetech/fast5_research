import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import logging
import os

import h5py
import numpy as np

from fast5_research.fast5 import Fast5
from fast5_research.fast5_bulk import BulkFast5
from fast5_research.util import _sanitize_data_for_writing


def extract_reads():
    logging.basicConfig(
        format='[%(asctime)s - %(name)s] %(message)s',
        datefmt='%H:%M:%S', level=logging.INFO
    )
    logger = logging.getLogger('Extract Reads')
    parser = argparse.ArgumentParser(description='Bulk .fast5 to single read .fast5 conversion.')
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
        self._write_read(read)
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


