//! Reuse of the big temporary buffers of a pass. A pass allocates and frees hundreds of buffers of 16 KiB to a few MiB;
//! allocators hand such sizes back to the system at once (Android: every use then faults its pages in again, ~15% of
//! the time of a pass on a phone), so they are kept here instead: `zeros` / `copy` take one, `give` returns it. At most
//! `CAP` values are kept, and none after the outermost call returns (`clear`): memory at rest is unchanged.
//! Which buffer a caller gets never changes a result (they are zeroed or overwritten whole).

use alloc::vec::Vec;

/// A buffer of n zeros.
pub fn zeros(n: usize) -> Vec<f32> {
    if let Some(mut v) = imp::take(n) {
        v.clear();
        v.resize(n, 0.0);
        return v;
    }
    alloc::vec![0.0; n]
}

/// A buffer of n values the caller writes whole before reading any (a reused one keeps its old contents).
pub fn any(n: usize) -> Vec<f32> {
    if let Some(mut v) = imp::take(n) {
        v.resize(n, 0.0);
        return v;
    }
    alloc::vec![0.0; n]
}

/// A copy of x.
pub fn copy(x: &[f32]) -> Vec<f32> {
    if let Some(mut v) = imp::take(x.len()) {
        v.clear();
        v.extend_from_slice(x);
        return v;
    }
    x.to_vec()
}

pub use imp::{clear, give};

#[cfg(feature = "std")]
mod imp {
    use std::sync::Mutex;

    /// Smaller buffers are the allocator's business (values).
    const MIN: usize = 4096;
    /// Values kept at most (12 MiB).
    const CAP: usize = 3 << 20;

    struct Pool {
        free: Vec<Vec<f32>>,
        kept: usize, // sum of their capacities
    }

    static POOL: Mutex<Pool> = Mutex::new(Pool { free: Vec::new(), kept: 0 });

    fn lock() -> std::sync::MutexGuard<'static, Pool> {
        POOL.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The smallest kept buffer that holds n values (and at most 2n: a big one is not spent on a small need).
    pub fn take(n: usize) -> Option<Vec<f32>> {
        if n < MIN {
            return None;
        }
        let mut p = lock();
        let i = (0..p.free.len()).filter(|&i| (n..=2 * n).contains(&p.free[i].capacity())).min_by_key(|&i| p.free[i].capacity())?;
        let v = p.free.swap_remove(i);
        p.kept -= v.capacity();
        Some(v)
    }

    /// A buffer no longer needed.
    pub fn give(v: Vec<f32>) {
        let c = v.capacity();
        if c < MIN {
            return;
        }
        let mut p = lock();
        if p.kept + c <= CAP {
            p.kept += c;
            p.free.push(v);
        }
    }

    /// Frees every kept buffer.
    pub fn clear() {
        let free = {
            let mut p = lock();
            p.kept = 0;
            core::mem::take(&mut p.free)
        };
        drop(free);
    }
}

#[cfg(not(feature = "std"))]
mod imp {
    use alloc::vec::Vec;

    pub fn take(_: usize) -> Option<Vec<f32>> {
        None
    }

    pub fn give(_: Vec<f32>) {}

    pub fn clear() {}
}
