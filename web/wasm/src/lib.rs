//! tinyasr for the browser: the C API of the crate (tinyasr_load, tinyasr_session_*, tinyasr_file_*) plus buffers that
//! JavaScript can fill (audio, model bytes, NUL-terminated strings).
#![allow(unsafe_code)]

pub use tinyasr;

/// n bytes, 16-byte aligned (f32 audio and model files fit).
#[unsafe(no_mangle)]
pub extern "C" fn tinyasr_alloc(n: usize) -> *mut u8 {
    let l = core::alloc::Layout::from_size_align(n.max(1), 16).unwrap();
    // SAFETY: the layout has a nonzero size.
    unsafe { std::alloc::alloc(l) }
}

/// # Safety: p came from tinyasr_alloc(n).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinyasr_dealloc(p: *mut u8, n: usize) {
    let l = core::alloc::Layout::from_size_align(n.max(1), 16).unwrap();
    unsafe { std::alloc::dealloc(p, l) }
}

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// performance.now() in seconds, given by the page (imports.env.now).
    fn now() -> f64;
}

extern "C" fn clock() -> f64 {
    // SAFETY: a plain JavaScript function without arguments.
    unsafe { now() }
}

/// The clock to pass to tinyasr_session_new (a function pointer: its table index in wasm).
#[unsafe(no_mangle)]
pub extern "C" fn tinyasr_clock() -> extern "C" fn() -> f64 {
    clock
}
