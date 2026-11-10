//! Whole on-device speech recognition with no dependencies: microphone audio in, words with times out.
//!
//! Two ways to recognize:
//!
//! - [`Session`]: live audio. A streaming pass (Zipformer2 transducer) gives text as you speak; the open
//!   sentence is re-decoded with full context at pauses; Silero VAD tells speech from silence, and the model's
//!   end-of-turn head whether the speaker has finished, which ends the sentence and makes its words final. Language:
//!   one of the languages the host allows (see [`Session::new`]).
//! - [`Transcription`]: a finished recording. Only the full-context pass, over sentence-sized pieces cut at pauses;
//!   words come out piece by piece as they are decoded (an iterator).
//!
//! Every model is a file the caller hands over as bytes ([`Files`]): the recognizer (.tzf, int4, with its end-of-turn
//! head; uncompressed - the host unzips a .tzf.gz) and Silero VAD straight from its ONNX file. Which languages to recognize
//! in is the host's choice (it may run a language ID such as tinylid first). Single-threaded: the host calls in from
//! whatever thread it likes.
//!
//! ```no_run
//! # let read = |p: &str| std::fs::read(p).unwrap();
//! let (asr, vad) = (read("model.tzf"), read("silero_vad.onnx"));
//! let models = tinyasr::Models::load(tinyasr::Files { asr: &asr, vad: &vad }).unwrap();
//! let mut s = tinyasr::Session::new(&models, &[tinyasr::Lang::En], Default::default());
//! let mic: Vec<f32> = vec![0.0; 16000];
//! for e in s.push(&mic).into_iter().chain(s.finish()) {
//!     if let tinyasr::Event::Commit(words) = e { println!("{words:?}") }
//! }
//! ```

#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
#![deny(unsafe_code)]

extern crate alloc;

mod compat;
mod fbank;
mod kernels;
mod model;
#[cfg(feature = "c-api")]
mod ffi;
mod file;
// a static library without std carries its allocator (the host's malloc) and panic handler
#[cfg(all(not(feature = "std"), not(target_arch = "wasm32"), not(test)))]
mod no_std_runtime;
mod onnx;
mod session;
mod text;
mod vad;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

pub use file::Transcription;
pub use session::Session;
pub use vad::{Vad, VadState};

/// Sample rate of all audio in and out (mono f32 in [-1, 1]).
pub const SAMPLE_RATE: u32 = 16_000;

/// Recognition languages (the order of the model's language embeddings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    En = 0,
    Ru = 1,
    Uk = 2,
    Es = 3,
    It = 4,
    Pt = 5,
    Uz = 6,
}

impl Lang {
    pub const ALL: [Lang; 7] = [Lang::En, Lang::Ru, Lang::Uk, Lang::Es, Lang::It, Lang::Pt, Lang::Uz];

    pub fn code(self) -> &'static str {
        ["en", "ru", "uk", "es", "it", "pt", "uz"][self as usize]
    }

    pub fn from_code(c: &str) -> Option<Lang> {
        Lang::ALL.into_iter().find(|l| l.code() == c)
    }

}

/// The languages a recognition may use, as the host allows them: one fixes it; with several the recognizer picks
/// (every full-context pass runs in each of them and keeps the transcript it is most sure of: that many times the
/// cost of those passes). Only the model's own count; none of them (or empty): all of the model's.
fn allowed(m: &Models, langs: &[Lang]) -> Vec<Lang> {
    let own = m.langs();
    let l: Vec<Lang> = own.iter().copied().filter(|l| langs.contains(l)).collect();
    if l.is_empty() { own.to_vec() } else { l }
}

/// A recognized word: its text (with punctuation) and times (s from the start of the audio).
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// What recognition reports, in order.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Final words, after all earlier ones; they never change.
    Commit(Vec<Word>),
    /// The words not final yet, replacing the previous `Partial`: `open` from the full-context re-decode, `tail` from
    /// the streaming pass after it (newest, least sure).
    Partial { open: Vec<Word>, tail: Vec<Word> },
    /// With several languages: the recognizer switched to this one (words from now on are in it).
    Language(Lang),
    /// Diagnostics: the end-of-turn head at a pause (time s, P(the speaker has finished)).
    Eou { at: f32, p: f32 },
}

/// The model files, as bytes.
pub struct Files<'a> {
    /// The recognizer (.tzf, uncompressed). If it has no end-of-turn head (`eou_head.*`, v5.6 on), a sentence ends
    /// only after 1.6 s of silence.
    pub asr: &'a [u8],
    /// silero_vad.onnx (v5).
    pub vad: &'a [u8],
}

/// The loaded models; read-only, so one instance can serve several sessions.
pub struct Models {
    asr: model::Model,
    vad: Vad,
}

impl Models {
    pub fn load(f: Files) -> Result<Models, &'static str> {
        Ok(Models {
            asr: crate::model::Model::from_bytes(f.asr)?,
            vad: Vad::from_onnx(f.vad)?,
        })
    }

    /// The languages of the recognizer: en, ru, uk; en, es, it, pt; or en, ru, uz.
    pub fn langs(&self) -> &[Lang] {
        &self.asr.langs
    }

    /// Streaming chunk of the recognizer (ms).
    pub fn chunk_ms(&self) -> u32 {
        (self.asr.chunk * 20) as u32
    }

    /// The full-context pass in the one of `langs` the model is most sure of (mean token log-probability): (tokens,
    /// that language; None if it wrote nothing).
    fn full(&self, x: &[f32], langs: &[Lang]) -> (Vec<crate::model::Token>, Option<Lang>) {
        if let [l] = langs {
            return (self.asr.full(x, self.asr.lang_index(*l)), Some(*l));
        }
        let mut best: Option<(f32, Vec<crate::model::Token>, Lang)> = None;
        for &l in langs {
            let t = self.asr.full(x, self.asr.lang_index(l));
            let sc = if t.is_empty() { -1e9 } else { t.iter().map(|k| k.logp).sum::<f32>() / t.len() as f32 };
            if best.as_ref().is_none_or(|b| sc > b.0) {
                best = Some((sc, t, l));
            }
        }
        let (_, t, l) = best.unwrap();
        let l = if t.is_empty() { None } else { Some(l) };
        (t, l)
    }

    /// P(the speaker has finished) at the end of `turn` (the audio since the last sentence end, with the pause); None
    /// without the head or for a turn too short to decode.
    fn eou(&self, turn: &[f32], lang: Lang) -> Option<f32> {
        self.asr.full_head(turn, self.asr.lang_index(lang)).1
    }
}

/// Session settings.
pub struct Options {
    /// With several languages: the one to assume until the first pick (e.g. the last session's, or the UI's).
    pub start_lang: Lang,
    /// A monotonic clock (s). Re-decodes are spaced by how long they took, to keep them under ~1/3 of real time on a
    /// slow device; without a clock their time is estimated as `decode_rtf` x the audio.
    pub clock: Option<Box<dyn Fn() -> f64>>,
    pub decode_rtf: f32,
}

impl Default for Options {
    fn default() -> Options {
        Options { start_lang: Lang::En, clock: None, decode_rtf: 0.1 }
    }
}
