import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import os

import logging

from fast5_research.fast5 import Fast5
from fast5_research.fast5_bulk import BulkFast5


def extract_single_reads():
    logging.basicConfig(
        format='[%(asctime)s - %(name)s] %(message)s',
        datefmt='%H:%M:%S', level=logging.INFO
    )
    logger = logging.getLogger('Extract Reads')
    parser = argparse.ArgumentParser(description='Bulk .fast5 to single read .fast5 conversion.')
    parser.add_argument('input', help='Bulk .fast5 file for input.')
    parser.add_argument('output', help='Output folder.')
    parser.add_argument('--flat', default=False, action='store_true',
                        help='Create all .fast5 files in one directory')
    parser.add_argument('--by_id', default=False, action='store_true',
                        help='Name single-read .fast5 files by read_id.')
    parser.add_argument('--prefix', default='read', help='Read file prefix.')
    parser.add_argument('--channel_range', nargs=2, type=int, default=None, help='Channel range (inclusive).')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes.')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        raise IOError('The output directory must not exist.')

    worker = functools.partial(
        extract_channel_reads,
        args.input, args.output, args.prefix, args.flat, args.by_id,
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
                except Exception:
                    logger.warning("Error processing channel.")
                else:
                    logger.info("Extracted {} reads from channel {}.".format(n_reads, channel))
    else:
        for channel in channels:
            worker(channel)
    logger.info("Finished.")


def extract_channel_reads(source, output, prefix, flat, by_id, channel):

    if flat:
        out_path = output
    else:
        out_path = os.path.join(output, str(channel))
        os.makedirs(out_path)

    with BulkFast5(source) as src:
        raw_data = src.get_raw(channel, use_scaling=False)
        meta = src.get_metadata(channel)
        tracking_id = src.get_tracking_meta()
        context_tags = src.get_context_meta()
        channel_id = {
            'channel_number': channel,
            'range': meta['range'],
            'digitisation': meta['digitisation'],
            'offset': meta['offset'],
            'sample_rate': meta['sample_rate'],
            'sampling_rate': meta['sample_rate']
        }
        median_before = None
        counter = 1
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
                if by_id:
                    filename = '{}.fast5'.format(read['read_id'])
                else:
                    filename =  '{}_read_ch{}_file{}.fast5'.format(
                        prefix, channel, read_number
                    )
                filename = os.path.join(out_path, filename)
                with Fast5.New(filename, 'a', tracking_id=tracking_id, context_tags=context_tags, channel_id=channel_id) as h:
                    h.set_raw(raw_slice, meta=read_id, read_number=read_number)
    return counter, channel
