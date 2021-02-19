# nvbio-benchmarks

Benchmarking program for NVBio implementation of the Myers bitvector algorithm on GPUs

## Install

First, NVBio library must be downloaded and compiled. `nvcc` must be in `$PATH`. For convenience, the Makefile has a rule for that:

```
make nvbio-install
```

If NVBio compiling is successful, the benchmarking program can be compiled.

```
make nvbio-benchmarks
```

## Usage

Running the executable without arguments will print the usage:

```
Usage:
nvbio-benchmarks <file> <max_seq_len> <num_alignments> <batch_size=50000>
```

The file format that the executable accepts is in the format of:

```
>PATTERN
<TEXT
>PATTERN
<TEXT
....
```

Each pair of PATTERN and TEXT is an alignment to compute.
