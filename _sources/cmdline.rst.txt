Command line Programs
=====================

`fast5_research` comes with two commandline programs for conversion of sequencing
read data.

**extract_reads** - extracts reads from a bulk ``.fast5`` to either single- or multi-read
``.fast5``:

.. code-block:: bash

    usage: extract_reads [-h] [--multi | --single] [--flat] [--by_id]
                         [--prefix PREFIX]
                         [--channel_range CHANNEL_RANGE CHANNEL_RANGE]
                         [--workers WORKERS] [--limit LIMIT]
                         input output
    
    Bulk .fast5 to read .fast5 conversion.
    
    positional arguments:
      input                 Bulk .fast5 file for input.
      output                Output folder.
    
    optional arguments:
      -h, --help            show this help message and exit
      --multi               Output multi-read files.
      --single              Output single-read files.
      --flat                Create all .fast5 files in one directory
      --by_id               Name single-read .fast5 files by read_id.
      --prefix PREFIX       Read file prefix.
      --channel_range CHANNEL_RANGE CHANNEL_RANGE
                            Channel range (inclusive).
      --workers WORKERS     Number of worker processes.
      --limit LIMIT         Limit reads per channel.


**filter_reads** - extracts a subset of reads from a set of multi-read ``.fast5`` files.

.. code-block:: bash

    usage: filter_reads [-h] [--tsv_field TSV_FIELD] [--multi | --single]
                        [--prefix PREFIX] [--recursive] [--workers WORKERS]
                        input output filter
    
    Extract reads from multi-read .fast5 files.
    
    positional arguments:
      input                 Path to input multi-read .fast5 files.
      output                Output folder.
      filter                A .tsv file with column `read_id` defining required
                            reads.
    
    optional arguments:
      -h, --help            show this help message and exit
      --tsv_field TSV_FIELD
                            Field name from `filter` file to obtain read IDs.
      --multi               Output multi-read files.
      --single              Output single-read files.
      --prefix PREFIX       Read file prefix.
      --recursive           Search recursively under `input` for source files.
      --workers WORKERS     Number of worker processes.
