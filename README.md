# tinyasr

**Tiny** **A**utomatic **S**peech **R**ecognition library for running new trained Zipformer2 models, aimed to bring accurate on device speech recognition with small to download models.

Demo at: https://dkaraush.github.io/tinyasr/web/

- Small multilingual [models](#models), benchmarked on FLEURS and Common Voice.
- Punctuation, capitalization and word timestamps.
- Live streaming with re-decodes of the open sentence, or a full-context pass over a finished recording.
- Built-in Silero VAD and end-of-turn head: knows when the speaker has finished a sentence.
- SIMD on ARM NEON (`dotprod`), x86_64 SSE, and WASM.
- Zero dependencies.

### Building (Native)

```bash
cargo build --release --features c-api
# library will be built at target/release/libtinyasr.a
# include headers are at include/tinyasr.h
```

### Models

WER %, second pass / streaming (560 ms).

| Model | Size (gzip) | Language | FLEURS | Common Voice |
|---|---|---|---|---|
| `en-ru-uk` (61M) | 31.2 MB | English | 8.48 / 10.75 | 14.20 / 18.18 |
|  |  | Russian | 5.52 / 7.05 | 5.12 / 6.70 |
|  |  | Ukrainian | 6.60 / 8.53 | 7.89 / 9.69 |
| `en-es-it-pt` (62M) | 32.2 MB | English | 8.89 / 11.29 | 14.30 / 17.96 |
|  |  | Spanish | 5.26 / 6.53 | 8.19 / 10.26 |
|  |  | Italian | 5.03 / 6.33 | 8.96 / 10.97 |
|  |  | Portuguese | 6.85 / 8.15 | 6.82 / 8.48 |
| `en-ru-uz` (61M) | 31.1 MB | Uzbek | 13.95 / 15.61 | 8.80 / 10.12 |
|  |  | Russian | 6.55 / 8.42 | 6.04 / 7.85 |
|  |  | English | 9.83 / 12.08 | 15.77 / 19.18 |