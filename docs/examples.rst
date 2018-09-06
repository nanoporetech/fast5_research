Fast5 Examples
==============

The following code snippets demonstrate basic IO using key features of the API.

Read Files
----------

The library provides the `Fast5` class which extends `h5py.File` with methods
for acquiring common datasets and attributes from files without requiring
knowledge of the file structure. To read a file and obtain a useful summary:

.. code-block:: python

    from fast5_research import Fast5

    filename='my.fast5'

    with Fast5(filename) as fh:
        raw = fh.get_read(raw=True)
        summary = fh.summary()
    print('Raw is {} samples long.'.format(len(raw)))
    print('Summary {}.'.format(summary))

Note that in this example the raw data will be provided in pA s.

The library also allows writing of files which are conformant with Oxford
Nanopore Technologies' software. Certain meta data are needed, which the
library will enforce are present:

.. code-block:: python

    import numpy as np
    from fast5_research import Fast5

    filename='my_new.fast5'
    mean, stdv, n = 40.0, 2.0, 10000
    raw_data = np.random.laplace(mean, stdv/np.sqrt(2), int(dwell))

    # example of how to digitize data 
    start, stop = int(min(raw_data - 1)), int(max(raw_data + 1))
    rng = stop - start
    digitisation = 8192.0
    bins = np.arange(start, stop, rng / digitisation)
    # np.int16 is required, the library will refuse to write anything other
    raw_data = np.digitize(raw_data, bins).astype(np.int16)
    
    # The following are required meta data
    channel_id = {
        'digitisation': digitisation,
        'offset': 0,
        'range': rng,
        'sampling_rate': 4000,
        'channel_number': 1,
        }
    read_id = {
        'start_time': 0,
        'duration': len(raw_data),
        'read_number': 1,
        'start_mux': 1,
        'read_id': str(uuid4()),
        'scaling_used': 1,
        'median_before': 0,
    }
    tracking_id = {
        'exp_start_time': '1970-01-01T00:00:00Z',
        'run_id': str(uuid4()).replace('-',''),
        'flow_cell_id': 'FAH00000',
    }
    context_tags = {}
    
    with Fast5.New(filename, 'w', tracking_id=tracking_id, context_tags=context_tags, channel_id=channel_id) as h:
        h.set_raw(raw_data, meta=read_id, read_number=1)


Bulk Files
----------

The library exposes data within bulk `.fast5` files through the `BulkFast5` class:

.. code-block:: python

    from fast5_research import BulkFast5

    filename = 'my_bulk.fast5'
    channel = 100
    samples = [1000, 100000]

    with BulkFast5(filename) as fh:
        raw = fh.get_raw(channel, raw_indices=samples)
        multiplexer_changes = get_mux_changes_in_window(
            channel, raw_indices=samples)

The `BulkFast5` class provides in-memory caching of many intermediate results,
to optimize repeated calls to the same methods.

