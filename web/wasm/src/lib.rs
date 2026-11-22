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

/// Threads (the build with atomics, `tinyasr-mt.wasm`): the page's Web Workers run the same module on the shared
/// memory, each with its own stack and thread-local storage; tinyasr's steps go to them through the host parallel-for
/// of the C API (tinyasr_set_parallel_for).
#[cfg(target_feature = "atomics")]
mod threads {
    use core::ffi::c_void;

    /// tinyasr.h: void task(void *ctx, size_t i).
    type Task = unsafe extern "C" fn(ctx: *mut c_void, i: usize);

    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        /// The page's parallel-for: task(ctx, i) for every i in 0..n on its workers and this thread (task: its table
        /// index), returning once all have run.
        fn parallel_for(n: usize, task: usize, ctx: *mut c_void);
    }

    unsafe extern "C" fn host(_user: *mut c_void, n: usize, task: Task, ctx: *mut c_void) {
        // SAFETY: the page runs every task and returns only when all are done, while ctx is alive.
        unsafe { parallel_for(n, task as usize, ctx) }
    }

    /// The parallel-for to pass to tinyasr_set_parallel_for.
    #[unsafe(no_mangle)]
    pub extern "C" fn tinyasr_web_parallel_for() -> unsafe extern "C" fn(*mut c_void, usize, Task, *mut c_void) {
        host
    }

    /// A worker's share of a step: task(ctx, i) (task, ctx: as parallel_for got them).
    ///
    /// # Safety
    /// task and ctx came from a parallel_for that has not returned yet.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn tinyasr_web_task(task: usize, ctx: *mut c_void, i: usize) {
        // SAFETY: task is the table index of a Task (above).
        let task: Task = unsafe { core::mem::transmute::<usize, Task>(task) };
        unsafe { task(ctx, i) }
    }
}
