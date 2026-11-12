//! C ABI (include/tinyasr.h). Results are JSON text owned by the object they came from, valid until the next call
//! on it.
#![allow(unsafe_code)]

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::{c_char, CStr};

use crate::{Event, Files, Lang, Models, Options, Session, Transcription, Word};

pub struct CSession {
    s: Session<'static>,
    out: Vec<u8>,
}

pub struct CFile {
    f: Transcription<'static>,
    out: Vec<u8>,
}

/// # Safety: p is null or points to n bytes.
unsafe fn bytes<'a>(p: *const u8, n: usize) -> Option<&'a [u8]> {
    if p.is_null() { None } else { Some(unsafe { core::slice::from_raw_parts(p, n) }) }
}

/// # Safety: p is null or a NUL-terminated string.
unsafe fn lang(p: *const c_char) -> Option<Lang> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }.to_str().ok().and_then(Lang::from_code)
}

/// A comma-separated list ("en,ru"); null or empty: all.
/// # Safety: p is null or a NUL-terminated string.
unsafe fn langs(p: *const c_char) -> Vec<Lang> {
    if p.is_null() {
        return Vec::new();
    }
    let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("");
    s.split(',').filter_map(|c| Lang::from_code(c.trim())).collect()
}

fn put(out: &mut Vec<u8>, s: &str) -> *const c_char {
    out.clear();
    out.extend_from_slice(s.as_bytes());
    out.push(0);
    out.as_ptr() as *const c_char
}

fn esc(s: &str, o: &mut String) {
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
}

fn words(ws: &[Word], o: &mut String) {
    o.push('[');
    for (i, w) in ws.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str("{\"text\":");
        esc(&w.text, o);
        o.push_str(&format!(",\"start\":{:.2},\"end\":{:.2}}}", w.start, w.end));
    }
    o.push(']');
}

/// Events as a JSON array: {"type": "commit", "words": [...]}, {"type": "partial", "open": [...], "tail": [...]},
/// {"type": "language", "lang": "en"}, {"type": "eou", "at": s, "p": P}; a word is {"text", "start", "end"}.
pub fn events_json(ev: &[Event]) -> String {
    let mut o = String::from("[");
    for (i, e) in ev.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        match e {
            Event::Commit(w) => {
                o.push_str("{\"type\":\"commit\",\"words\":");
                words(w, &mut o);
                o.push('}');
            }
            Event::Partial { open, tail } => {
                o.push_str("{\"type\":\"partial\",\"open\":");
                words(open, &mut o);
                o.push_str(",\"tail\":");
                words(tail, &mut o);
                o.push('}');
            }
            Event::Language(l) => o.push_str(&format!("{{\"type\":\"language\",\"lang\":\"{}\"}}", l.code())),
            Event::Eou { at, p } => o.push_str(&format!("{{\"type\":\"eou\",\"at\":{at:.2},\"p\":{p:.3}}}")),
        }
    }
    o.push(']');
    o
}

/// Load the models from memory (copied; the caller may free them after). On failure
/// returns null and writes the reason (NUL-terminated, truncated) to `err` if not null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_load(
    asr: *const u8,
    asr_len: usize,
    vad: *const u8,
    vad_len: usize,
    err: *mut c_char,
    err_len: usize,
) -> *mut Models {
    let r = (|| {
        // SAFETY: the caller passes valid (pointer, length) pairs.
        let f = unsafe {
            Files {
                asr: bytes(asr, asr_len).ok_or("no recognizer model")?,
                vad: bytes(vad, vad_len).ok_or("no VAD model")?,
            }
        };
        Models::load(f)
    })();
    match r {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(e) => {
            if !err.is_null() && err_len > 0 {
                let n = e.len().min(err_len - 1);
                // SAFETY: err has err_len bytes.
                unsafe {
                    core::ptr::copy_nonoverlapping(e.as_ptr(), err as *mut u8, n);
                    *err.add(n) = 0;
                }
            }
            core::ptr::null_mut()
        }
    }
}

/// Free models (after every session / file using them).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_free(m: *mut Models) {
    if !m.is_null() {
        // SAFETY: m came from tinyasr_load.
        drop(unsafe { Box::from_raw(m) });
    }
}

/// Streaming chunk of the recognizer (ms).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_chunk_ms(m: *const Models) -> u32 {
    // SAFETY: m came from tinyasr_load.
    unsafe { &*m }.chunk_ms()
}

/// A live session. `langs`: the languages to recognize in, comma-separated ("en,ru"; null or "": all). One fixes the
/// language; with several the recognizer picks among them, starting from `start_lang` (may be null). `clock`: a monotonic clock in seconds, or null (re-decode time is then estimated).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_session_new(
    m: *const Models,
    lang_list: *const c_char,
    start_lang: *const c_char,
    clock: Option<extern "C" fn() -> f64>,
) -> *mut CSession {
    // SAFETY: m came from tinyasr_load and outlives the session (the caller's contract).
    let m: &'static Models = unsafe { &*m };
    let mut opt = Options::default();
    // SAFETY: null or NUL-terminated strings.
    if let Some(l) = unsafe { lang(start_lang) } {
        opt.start_lang = l;
    }
    if let Some(c) = clock {
        opt.clock = Some(Box::new(move || c()));
    }
    let s = Session::new(m, &unsafe { langs(lang_list) }, opt);
    Box::into_raw(Box::new(CSession { s, out: Vec::new() }))
}

/// Feed n samples (16 kHz); returns the events (JSON array).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_session_push(s: *mut CSession, pcm: *const f32, n: usize) -> *const c_char {
    // SAFETY: s came from tinyasr_session_new; pcm points to n floats.
    let s = unsafe { &mut *s };
    let x = if n == 0 { &[][..] } else { unsafe { core::slice::from_raw_parts(pcm, n) } };
    let e = events_json(&s.s.push(x));
    put(&mut s.out, &e)
}

/// End of audio: the last events (everything final).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_session_finish(s: *mut CSession) -> *const c_char {
    // SAFETY: s came from tinyasr_session_new.
    let s = unsafe { &mut *s };
    let e = events_json(&s.s.finish());
    put(&mut s.out, &e)
}

/// The session's language code (static string).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_session_lang(s: *const CSession) -> *const c_char {
    // SAFETY: s came from tinyasr_session_new.
    lang_str(unsafe { &*s }.s.lang())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_session_free(s: *mut CSession) {
    if !s.is_null() {
        // SAFETY: s came from tinyasr_session_new.
        drop(unsafe { Box::from_raw(s) });
    }
}

fn lang_str(l: Lang) -> *const c_char {
    [c"en", c"ru", c"uk", c"es", c"it", c"pt", c"uz", c"fa", c"km"][l as usize].as_ptr()
}

/// A recorded file (n samples, 16 kHz, copied): VAD runs here. `langs` as for sessions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_file_new(m: *const Models, pcm: *const f32, n: usize, lang_list: *const c_char) -> *mut CFile {
    // SAFETY: m came from tinyasr_load and outlives the file; pcm points to n floats; lang is null or a C string.
    let m: &'static Models = unsafe { &*m };
    let x = if n == 0 { &[][..] } else { unsafe { core::slice::from_raw_parts(pcm, n) } };
    let f = Transcription::new(m, x, &unsafe { langs(lang_list) });
    Box::into_raw(Box::new(CFile { f, out: Vec::new() }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_file_lang(f: *const CFile) -> *const c_char {
    // SAFETY: f came from tinyasr_file_new.
    lang_str(unsafe { &*f }.f.lang())
}

/// The next piece's final words (JSON array of {"text", "start", "end"}), or null when the file is done.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_file_next(f: *mut CFile) -> *const c_char {
    // SAFETY: f came from tinyasr_file_new.
    let f = unsafe { &mut *f };
    match f.f.next() {
        Some(ws) => {
            let mut o = String::new();
            words(&ws, &mut o);
            put(&mut f.out, &o)
        }
        None => core::ptr::null(),
    }
}

/// How far the file is recognized (0..1).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_file_progress(f: *const CFile) -> f32 {
    // SAFETY: f came from tinyasr_file_new.
    let f = &unsafe { &*f }.f;
    if f.duration() > 0.0 { (f.position() / f.duration()).min(1.0) } else { 1.0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_file_free(f: *mut CFile) {
    if !f.is_null() {
        // SAFETY: f came from tinyasr_file_new.
        drop(unsafe { Box::from_raw(f) });
    }
}
