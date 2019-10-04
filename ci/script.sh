#!/bin/bash

set -ex

# On Rust 1.32.0, we only care about passing tests.
if [ "$TRAVIS_RUST_VERSION" = "stable" ]; then
  cargo fmt --all -- --check
  cargo clippy -- -D warnings
fi

cargo build --verbose
cargo test --verbose
cargo build --verbose --features "netlib"
