#!/bin/bash

set -ex

if [ $TARGET = "x86_64-unknown-linux-musl" ]; then
  cargo build --target ${TARGET} --verbose
else
  cargo test --target ${TARGET} --verbose
  cargo build --target ${TARGET} --features netlib --verbose

  # On Rust 1.32.0, we only care about passing tests.
  if rustc --version | grep -v "^rustc 1.32.0"; then
    cargo fmt --all -- --check
    cargo clippy -- -D warnings
  fi
fi

