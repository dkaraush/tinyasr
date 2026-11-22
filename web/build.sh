#!/bin/bash
# WebAssembly builds of tinyasr for the demo page: wasm32 + SIMD128 -> web/tinyasr.wasm (one thread), and with threads
# (atomics, shared memory imported from the page: std rebuilt with them, nightly Rust with rust-src) -> web/tinyasr-mt.wasm
cd "$(dirname "$0")/wasm"
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown || exit 1
cp target/wasm32-unknown-unknown/release/tinyasr_web.wasm ../tinyasr.wasm
# relaxed SIMD (its int8 dot product for the int4 layers; the same results): -relaxed.wasm, which the page uses where
# the browser has it
RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo build --release --target wasm32-unknown-unknown \
  --target-dir target/relaxed || exit 1
cp target/relaxed/wasm32-unknown-unknown/release/tinyasr_web.wasm ../tinyasr-relaxed.wasm
# memory: 64 MiB at start, up to 1 GiB (the page creates it with the same limits); the workers set their own stack
# pointer and thread-local storage (__wasm_init_tls)
MT="-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals -C link-arg=--shared-memory -C link-arg=--import-memory"
MT="$MT -C link-arg=--initial-memory=67108864 -C link-arg=--max-memory=1073741824"
for e in __wasm_init_tls __tls_size __tls_align __stack_pointer; do MT="$MT -C link-arg=--export=$e"; done
RUSTFLAGS="$MT" cargo +nightly build -Z build-std=std,panic_abort --release --target wasm32-unknown-unknown \
  --target-dir target/mt || exit 1
cp target/mt/wasm32-unknown-unknown/release/tinyasr_web.wasm ../tinyasr-mt.wasm
RUSTFLAGS="${MT/+simd128/+simd128,+relaxed-simd}" cargo +nightly build -Z build-std=std,panic_abort --release \
  --target wasm32-unknown-unknown --target-dir target/mt-relaxed || exit 1
cp target/mt-relaxed/wasm32-unknown-unknown/release/tinyasr_web.wasm ../tinyasr-mt-relaxed.wasm
ls -la ../tinyasr*.wasm
