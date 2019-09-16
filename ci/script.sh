#!/bin/bash

set -ex

cargo build --verbose
cargo test --verbose
cargo build --verbose --features "netlib"

# On Rust 1.32.0, we only care about passing tests.
if rustc --version | grep -v "^rustc 1.32.0"; then
  cargo fmt --all -- --check
  cargo clippy -- -D warnings
fi


