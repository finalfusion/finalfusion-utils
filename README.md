[![Travis CI](https://img.shields.io/travis/finalfusion/finalfusion-utils.svg)](https://travis-ci.org/finalfusion/finalfusion-utils)

# finalfusion-utils

## Introduction

`finalfusion-utils` is a Rust crate offering various
functionalities to process and query embeddings.
`finalfusion-utils` supports conversion between different
formats, quantization of embedding matrices, similarity and
analogy queries as well as evaluation on analogy datasets.

## Installation

### Precompiled binaries

The following precompiled binaries can be found on the
[releases page](https://github.com/finalfusion/finalfusion-utils/releases):

* `x86_64-unknown-linux-gnu-mkl`: glibc Linux build, statically linked
  against Intel MKL. This is the recommended build for Intel (non-AMD)
  CPUs.
* `x86_64-unknown-linux-musl`: static Linux build using the MUSL C
  library. This binary does not link against a BLAS/LAPACK implementation
  and therefore does not support optimized product quantization.
* `universal-macos`: dynamic macOS build. Supports both the x86_64 and
  ARM64 architectures. Linked against the Accelerate framework for
  BLAS/LAPACK.


### Using `cargo`

`finalfusion-utils` can be installed using an up-to-date Rust
toolchain, which can be installed with [rustup](https://rustup.rs).

With a valid Rust toolchain, the crate is most easily installed through
`cargo`:

~~~shell
$ cargo install finalfusion-utils
~~~

Typically, you will want to enable support for a BLAS/LAPACK library
to speed up matrix multiplication and enable optimized product
quantization support. In order to do so, run

~~~shell
$ cargo install finalfusion-utils --features implementation
~~~

where `implementation` is one of the following:

* `accelerate`: the macOS Accelerate framework.
* `intel-mkl`: Intel MKL (downloaded and statically linked).
* `intel-mkl-amd`: Intel MKL, preinstalled MKL libaries expected, override
  CPU detection for AMD CPUs.
* `netlib`: any compatible system BLAS/LAPACK implementation(s).
* `openblas`: system-installed OpenBLAS. This option is discouraged,
  unless the system OpenBLAS library is a single-threaded build with
  locking. Otherwise, OpenBLAS' threading interacts badly with application
  threads.

## Building from source

`finalfusion-utils` can also be built from source,
after cloning this repository execute the following
command in the directory to find the exectuable under
`target/release/finalfusion`:

~~~shell
$ cargo build --release
~~~

## Usage

`finalfusion-utils` is built as a single binary, the
different functionality is invoked through subcommands:

### Converting embeddings

~~~shell
# Convert embeddings in fastText format to finalfusion
$ finalfusion convert -f fasttext -t finalfusion \
    embeddings.bin embeddings.fifu

# Convert embeddings in word2vec format to finalfusion
$ finalfusion convert -f word2vec -t finalfusion \
    embeddings.w2v embeddings.fifu

# Print help with all supported combinations:
$ finalfusion convert --help
~~~

### Quantizing an embedding matrix

~~~shell
# Quantize embeddings in finalfusion format with a
# single attempt through product quantization 
$ finalfusion quantize -f finalfusion -q pq  -a 1 \
    embeddings.pq
~~~

### Analogy and similarity queries

~~~ shell
# Get the 15 nearest neighbours of "Tübingen" for
# embeddings in finalfusion format.
$ finalfusion similar -f finalfusion -k 15 \
    embeddings.fifu

# Get the 5 best answers for the analogy query
# "Berlin" is to "Deutschland" as "Amsterdam" to:
$ finalfusion analogy -f finalfusion -k 5 \
    Berlin Deutschland Amsterdam embeddings.fifu
~~~

### Evaluation on analogy datasets

~~~shell
# Evaluate embeddings on some analogy dataset
$ finalfusion compute-accuracy embeddings.fifu \
    analogies.txt
~~~

### Dump metadata

~~~shell
# Dump optionally stored metadata and store in
# metadata.txt, only supported for finalfusion
# format
$ finalfusion metadata embeddings.fifu \
    > metadata.txt
~~~

### Convert Bucket Vocab to Explicit Vocab
~~~shell
# Converts a hash-bucket based subword vocab to
# one with explicitly stored n-grams.
$ finalfusion bucket-to-explicit buckets.fifu \
    explicit.fifu 
~~~

### Print completion script

~~~shell
# Print completion script for zsh
$ finalfusion completions zsh
~~~
