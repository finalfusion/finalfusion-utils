[![Travis CI](https://img.shields.io/travis/finalfusion/finalfusion-utils.svg)](https://travis-ci.org/finalfusion/finalfusion-utils)

# finalfusion-utils

## Introduction

`finalfusion-utils` is a Rust crate offering various
functionalities to process and query embeddings.
`finalfusion-utils` supports conversion between different
formats, quantization of embedding matrices, similarity and
analogy queries as well as evaluation on analogy datasets.

## Installation

Installing `finalfusion-utils` requires a Rust toolchain
with minimum version `1.32` which can be installed via
rustup.

With a valid Rust toolchain, the crate is most easily
installed through `cargo`:

~~~shell
$ cargo install finalfusion-utils
~~~

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

### Print completion script

~~~shell
# Print completion script for zsh
$ finalfusion completions zsh
~~~
