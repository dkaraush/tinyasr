#!/bin/bash
# WebAssembly build of tinyasr for the demo page: wasm32 + SIMD128 -> web/tinyasr.wasm
cd "$(dirname "$0")/wasm"
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown || exit 1
cp target/wasm32-unknown-unknown/release/tinyasr_web.wasm ../tinyasr.wasm
ls -la ../tinyasr.wasm
