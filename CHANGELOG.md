v1.2.13
-------
* Use filename if possible when extracting reads

v1.2.12
-------
* Clear up a deprecation warning: https://github.com/nanoporetech/fast5_research/issues/30


v1.2.11
* Add `filter_reads` program to extract a subset of reads from multi-reads.


v1.2.10
-------
* Minor syntax fix in `extract.py` for python2

v1.2.9
------
* Add basic support for creation of multi-read files from bulk files.

v1.2.8
------
* Small refactor of writing of mapping data.

v1.2.6
------
* Fix slow creation of mapping table

v1.2.5
------
* Ensure event structures containing text data are returned as strings rather than bytes under python3.

v1.2.3
------
* Fixes issue with numpy 1.15 on reading type of views of structured data.
* Updated documentation (https://nanoporetech.github.io/fast5_research/)

v1.2.2
------
* Conversion from bulk to reads.
* Require numpy >= 1.14.
* A bit more python3 bytes cleaning.
* Enforce types in raw, and required tracking_id attributes.

v1.1.0
------
* Python3 compatibility changes
* Add data cleaning steps for stringly types
* Unpin numpy version

v1.0.12
-------
* Enforce some typing constraints on meta data for compatibility with some basecallers.

v1.0.11
-------
* Ignore h5py warnings on import

v1.0.10
-------
* Fix bug finding attributes when EventDetection not present

v1.0.8
------
* Easy import of core classes and functions:
    `from fast5_research import Fast5, BulkFast5, iterate_fast5`
* Enable recursive (lazy) search in `iterate_fast5`.

v1.0.9
------
* Fix itertools import

v1.0.6
------
* Ensure returned events have same dtype
* fast5.py: all returned event arrays same dtype by passing them through self._get_read_data()
* requirements: use any version of numpy
* bump version to 1.0.6 
