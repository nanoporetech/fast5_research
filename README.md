![Oxford Nanopore Technologies logo](https://github.com/nanoporetech/medaka/raw/master/images/ONT_logo_590x106.png)


fast5_research
==============

[![Build Status](https://travis-ci.org/nanoporetech/fast5_research.svg?branch=master)](https://travis-ci.org/nanoporetech/fast5_research)

Python fast5 reading and writing functionality provided by ONT Research.

© 2018 Oxford Nanopore Technologies Ltd.

Features
--------

 * Read interface bulk `.fast5` file for extracting reads, channel states, voltage, ...
 * Read/Write interface to single read files guaranteeing conformity.
 * Works on Linux, MacOS, and Windows.
 * Open source (Mozilla Public License 2.0).

Documentation can be found at https://nanoporetech.github.io/fast5_research/.

Installation
------------

`fast5_research` is available from pypi can can be installed with pip:

    pip install fast5_research


Usage
-----

Full documentation can be found at the link above, below are two simple examples.

To read a file:

    from fast5_research import Fast5
    
    filename='my.fast5'
    
    with Fast5(filename) as fh:
        raw = fh.get_read(raw=True)
        summary = fh.summary()
    print('Raw is {} samples long.'.format(len(raw)))
    print('Summary {}.'.format(summary))

Write a file, the library will check the given meta data, ensure that all required
values are present, and covert all values to their defined types.

    from uuid import uuid4
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


Help
----

**Licence and Copyright**

© 2018 Oxford Nanopore Technologies Ltd.

`medaka` is distributed under the terms of the Mozilla Public License 2.0.

**Research Release**

Research releases are provided as technology demonstrators to provide early
access to features or stimulate Community development of tools. Support for
this software will be minimal and is only provided directly by the developers.
Feature requests, improvements, and discussions are welcome and can be
implemented by forking and pull requests. However much as we would
like to rectify every issue and piece of feedback users may have, the 
developers may have limited resource for support of this software. Research
releases may be unstable and subject to rapid iteration by Oxford Nanopore
Technologies.
