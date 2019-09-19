#!/bin/bash

set -ex

if [ $TARGET != "x86_64-unknown-linux-musl" ]; then
  exit 0
fi

cargo build --target "$TARGET" --release

tmpdir="$(mktemp -d)"
name="${PROJECT_NAME}-${TRAVIS_TAG}-${TARGET}"
staging="${tmpdir}/${name}"
mkdir "${staging}"
out_dir="$(pwd)/deployment"
mkdir "${out_dir}"

cp "target/${TARGET}/release/finalfusion" "${staging}/finalfusion"
strip "${staging}/finalfusion"
cp {README.md,LICENSE.md} "${staging}/"

( cd "${tmpdir}" && tar czf "${out_dir}/${name}.tar.gz" "${name}")

rm -rf "${tmpdir}"
