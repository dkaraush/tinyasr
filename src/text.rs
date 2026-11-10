//! Words out of tokens, and the demo's text touches: sentence ends, capitals, paired quotes.

#[cfg(not(feature = "std"))]
use crate::compat::F64Ext;
use alloc::string::String;
use alloc::vec::Vec;

use crate::model::Token;

/// A word with the times (s) of its first and last token; `tw`: of its last token with a letter or digit (the model
/// often writes trailing punctuation only once it has heard the pause after it); `lp`: log-probability of the last
/// token.
#[derive(Clone, Debug)]
pub struct W {
    pub w: String,
    pub t: f64,
    pub te: f64,
    pub tw: f64,
    pub lp: f32,
}

/// The model's end-of-utterance token (v5.6): not text.
pub const EOU_TOKEN: u32 = 12;

/// Pieces -> words; "▁" starts a word, punctuation sticks to the word before. `t0`: time of frame 0 (s).
pub fn words(pieces: &[String], tk: &[Token], t0: f64) -> Vec<W> {
    let mut out: Vec<W> = Vec::new();
    for k in tk.iter().filter(|k| k.id != EOU_TOKEN) {
        let p = pieces.get(k.id as usize).map_or("", |s| s.as_str());
        let t = t0 + k.frame as f64 * 0.04;
        match out.last_mut() {
            Some(last) if !p.starts_with('▁') => {
                last.w.push_str(p);
                last.te = t;
                if p.chars().any(char::is_alphanumeric) {
                    last.tw = t;
                }
                last.lp = k.logp;
            }
            _ => out.push(W { w: p.replace('▁', ""), t, te: t, tw: t, lp: k.logp }),
        }
    }
    out.retain(|w| !w.w.is_empty());
    out
}

/// Word times (s) for the output, from emission times. A transducer writes a word as its sound ends, not as it begins
/// (its first emission is ~0.2–0.4 s after the word's start): so a word ends at its last emission and begins where the
/// previous word ended (`from` for the first), with the unvoiced frames at both ends cut off — a word after a pause
/// begins where the speech does. `voiced(frame)`: the VAD flag of a 40 ms frame, None when not known (kept).
pub fn spans(ws: &[W], mut from: f64, voiced: impl Fn(i64) -> Option<bool>) -> Vec<(f32, f32)> {
    let lag = lag();
    let mut out: Vec<(i64, i64)> = Vec::with_capacity(ws.len());
    for w in ws {
        // the emissions, moved back by the model's delay; never before the previous word
        let a0 = (((w.t - lag) / 0.04).round() as i64).max((from / 0.04).round() as i64);
        let mut b = ((w.tw.max(w.t) + 0.04 - lag) / 0.04).round() as i64;
        let mut a = a0;
        b = b.max(a + 1);
        #[cfg(feature = "std")]
        if std::env::var("TINYASR_NOTRIM").is_ok() {
            out.push((a, b));
            from = b as f64 * 0.04;
            continue;
        }
        while b - a > 1 && voiced(a) == Some(false) {
            a += 1;
        }
        while b - a > 1 && voiced(b - 1) == Some(false) {
            b -= 1;
        }
        out.push((a, b));
        from = b as f64 * 0.04;
    }
    // words a short gap apart touch (no flicker between them in a highlight)
    for i in 1..out.len() {
        if out[i].0 - out[i - 1].1 <= 3 {
            out[i - 1].1 = out[i].0;
        }
    }
    out.into_iter().map(|(a, b)| ((a as f64 * 0.04) as f32, (b as f64 * 0.04) as f32)).collect()
}

#[cfg(feature = "std")]
fn lag() -> f64 {
    std::env::var("TINYASR_LAG").ok().and_then(|v| v.parse().ok()).unwrap_or(LAG)
}
#[cfg(not(feature = "std"))]
fn lag() -> f64 {
    LAG
}

/// How late the transducer writes a word after its sound (s): measured on FLEURS against the speech's own starts and ends.
const LAG: f64 = 0.32;

/// Ends a sentence: . ? ! … optionally followed by a closing quote.
pub fn is_end(w: &str) -> bool {
    let w = w.strip_suffix(['"', '»']).unwrap_or(w);
    w.ends_with(['.', '?', '!', '…'])
}

/// Without trailing , ; : — – -
pub fn trim_trail(w: &str) -> &str {
    w.trim_end_matches([',', ';', ':', '—', '–', '-'])
}

/// The first letter in upper case (not after a digit: "20th").
pub fn cap(w: &str) -> String {
    match w.char_indices().find(|(_, c)| c.is_alphanumeric()) {
        Some((i, c)) if c.is_alphabetic() => {
            let mut s = String::from(&w[..i]);
            s.extend(c.to_uppercase());
            s.push_str(&w[i + c.len_utf8()..]);
            s
        }
        _ => w.into(),
    }
}

/// Lower case, letters and digits only (to compare words).
pub fn norm(w: &str) -> String {
    w.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect()
}

/// A frozen sentence keeps only quotes that pair up. The model (v5.6) puts quote marks in the right places but often picks
/// the wrong direction, so the direction comes from the position first: a mark in front of a word opens, one after it
/// closes; then unmatched marks are dropped.
pub fn balance_quotes(ws: Vec<W>) -> Vec<W> {
    let edge = |s: &str, front: bool| -> usize {
        // bytes of the leading (front) or trailing run of non-letters / non-digits
        if front {
            s.char_indices().find(|(_, c)| c.is_alphanumeric()).map_or(s.len(), |(i, _)| i)
        } else {
            s.len() - s.char_indices().rev().find(|(_, c)| c.is_alphanumeric()).map_or(0, |(i, c)| i + c.len_utf8())
        }
    };
    let mut out: Vec<W> = ws
        .into_iter()
        .map(|mut w| {
            let n = edge(&w.w, true);
            w.w = alloc::format!("{}{}", w.w[..n].replace('»', "«"), &w.w[n..]);
            let n = w.w.len() - edge(&w.w, false);
            w.w = alloc::format!("{}{}", &w.w[..n], w.w[n..].replace('«', "»"));
            w
        })
        .collect();
    let mut open = Vec::new();
    for (i, w) in out.iter_mut().enumerate() {
        let mut t = String::new();
        for c in w.w.chars() {
            match c {
                '«' => {
                    open.push(i);
                    t.push(c);
                }
                '»' => {
                    if open.pop().is_some() {
                        t.push(c);
                    }
                }
                c => t.push(c),
            }
        }
        w.w = t;
    }
    for i in open {
        if let Some(k) = out[i].w.rfind('«') {
            out[i].w.replace_range(k..k + '«'.len_utf8(), "");
        }
    }
    out.retain(|w| !w.w.is_empty());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bq(s: &str) -> String {
        let ws = s.split(' ').map(|w| W { w: w.into(), t: 0.0, te: 0.0, tw: 0.0, lp: 0.0 }).collect();
        balance_quotes(ws).iter().map(|w| w.w.as_str()).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn quotes() {
        assert_eq!(bq("Проект «Кто сверху"), "Проект Кто сверху");
        assert_eq!(bq("Фильм Одиссея» собрал"), "Фильм Одиссея собрал");
        assert_eq!(bq("фильм »Одиссея« собрал"), "фильм «Одиссея» собрал");
        assert_eq!(bq("«Да», сказал он"), "«Да», сказал он");
        assert_eq!(bq("сказал « да"), "сказал да");
    }

    #[test]
    fn caps() {
        assert_eq!(cap("20th"), "20th");
        assert_eq!(cap("«привет»"), "«Привет»");
        assert!(is_end("конец.»") && is_end("да?") && !is_end("так,"));
    }
}
