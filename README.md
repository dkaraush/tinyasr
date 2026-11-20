# tinyasr

**Tiny** **A**utomatic **S**peech **R**ecognition library for running new trained Zipformer2 models, aimed to bring accurate on device speech recognition with small to download models.

Demo at: https://dkaraush.github.io/tinyasr/web/

- Small multilingual [models](#models), benchmarked on FLEURS and Common Voice.
- RNN-T and TDT transducers, stateless or LSTM-hybrid decoders; greedy or beam search (width 4 by default).
- Punctuation, capitalization and word timestamps.
- Live streaming with re-decodes of the open sentence, or a full-context pass over a finished recording.
- Built-in Silero VAD and end-of-turn head: knows when the speaker has finished a sentence.
- SIMD on ARM (`i8mm`, `dotprod` or plain NEON, picked at runtime), x86_64 SSE, and WASM; int4 weights stay int4 in memory (~46 MB loaded for a 38 MB model).
- Multi-threading supported.
- Zero dependencies.

### Building (Native)

```bash
cargo build --release --features c-api
# library will be built at target/release/libtinyasr.a
# include headers are at include/tinyasr.h
```

### Models

WER %, second pass (beam 4) / streaming (greedy)

| Model | Size (gzip) | Language | FLEURS | Common Voice |
|---|---|---|---|---|
| `en-ru-uk` (63M) | 31.9 MiB | English | 7.62 / 10.46 | 13.53 / 17.54 |
|  |  | Russian | 5.47 / 7.22 | 4.89 / 7.04 |
|  |  | Ukrainian | 6.13 / 8.22 | 7.78 / 10.29 |
| `en-es-it-pt` (62M) | 30.7 MiB | English | 9.05 / 11.29 | 14.08 / 17.96 |
|  |  | Spanish | 5.15 / 6.53 | 7.82 / 10.26 |
|  |  | Italian | 4.90 / 6.33 | 8.64 / 10.97 |
|  |  | Portuguese | 6.63 / 8.15 | 6.63 / 8.48 |
| `uz` (64M) | 32.6 MiB | Uzbek (Latin) | 10.06 / 12.95 | 7.05 / 9.32 |
|  |  | Uzbek (Mixed) | 9.69 / 12.67 | 6.81 / 9.24 |
|  |  | Uzbek (Cyrillic) | 9.95 / 12.86 | 6.93 / 9.36 |
| `en-fa` (60M) | 28.7 MiB | Persian | 5.87 / 7.23 | 14.29 / 15.98 |
|  |  | English | 9.97 / 11.04 | 16.26 / 18.06 |
| `en-km` (60M) | 28.7 MiB | Khmer (CER) | 8.21 / 9.95 | — |
|  |  | English | 9.41 / 11.69 | 15.02 / 18.93 |

### License

tinyasr, including the model files in `models/`, is licensed under the [GNU General Public License v3.0 or later](LICENSE).

Models were trained primarily on [YODAS v3](https://huggingface.co/datasets/espnet/yodas3) (ESPnet, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)), a corpus of Creative Commons-licensed YouTube audio; see the dataset card for per-source attribution. Voice activity detection uses [Silero VAD](https://github.com/snakers4/silero-vad) (MIT).
