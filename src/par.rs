//! Parallel work. One thread unless the host allows more (`crate::Models::set_threads`) or hands over its own
//! parallel-for (`crate::Models::set_parallel_for`). The model's passes split their bigger steps into independent
//! tasks (`chunks_mut`, `map`); every output value is computed by the same code whichever thread runs it, so the
//! results do not depend on the number of threads.
//!
//! Which cores run the work is the host's choice: the built-in threads start on first use, from the thread that
//! asks, and inherit its CPU affinity (Linux, Android); a host parallel-for runs tasks wherever it likes.

#[cfg(feature = "std")]
pub use imp::*;

/// Work (multiply-adds, roughly) below which a step stays on one thread (~7 us on one core): handing out tasks to
/// spinning workers costs ~1-2 us, waking sleeping ones 5-20.
pub const MIN_WORK: usize = 1 << 19;

/// Rows per task for `rows` rows: `max` on one thread; with more, about 4 tasks per thread (whole multiples of 8 rows,
/// at least 16) so they balance. How rows are grouped never changes a result.
pub fn block_rows(rows: usize, max: usize) -> usize {
    let t = threads();
    if t < 2 {
        return max;
    }
    rows.div_ceil(4 * t).div_ceil(8).saturating_mul(8).clamp(16, max)
}

#[cfg(feature = "std")]
mod imp {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};

    /// A thread waiting for the next step (or for the others to finish one) spins a while, then sleeps.
    ///
    /// x86-64: as long as waking from sleep takes there, measured as the pool runs (`Shared::wake_ns`): if the step
    /// comes within that, spinning was the cheaper wait; if not, it cost at most one wake-up more than sleeping at once
    /// would have (spin-then-block, 2-competitive). On a desktop CPU (64-core Zen 5, 4 threads) that is the same
    /// speed as a long spin (file pass +2.5%, live latency the same) for a quarter less CPU time.
    ///
    /// Elsewhere (phones) the same when every core the threads may run on shares one clock (Linux, Android: one
    /// cpufreq policy, as when pinned to the big cores of a phone): a Snapdragon 8 Elite, 2 threads on its 2 prime
    /// cores, live: 14-17% fewer cycles, words as soon. Else a fixed spin of `SPIN` checks (~tens of us): a core that
    /// sleeps between steps lets its own clock drop (frequency scaling), and the steps it takes next run slower; there
    /// (4 threads over prime and performance cores) the wake-up-time spin made a file pass 40-60% slower and live words
    /// 25-60 ms later; waiting in `wfe` instead of spinning made them later too, and used no less battery.
    /// The first estimate of the wake-up time (ns), before any is measured.
    const WAKE0: u64 = 20_000;
    const SPIN: u32 = 20_000;

    /// Spins until done() (timed: for about ns, else `SPIN` checks); whether it is done.
    fn spin_for(ns: u64, timed: bool, mut done: impl FnMut() -> bool) -> bool {
        if timed {
            let t = std::time::Instant::now();
            loop {
                for _ in 0..32 {
                    if done() {
                        return true;
                    }
                    core::hint::spin_loop();
                }
                if t.elapsed().as_nanos() as u64 >= ns {
                    return done();
                }
            }
        }
        for _ in 0..SPIN {
            if done() {
                return true;
            }
            core::hint::spin_loop();
        }
        done()
    }

    /// Whether waits spin for the measured wake-up time (`spin_for`) for threads started from this one: always on
    /// x86-64; elsewhere when all the cores this thread may run on share one clock.
    fn timed_spin() -> bool {
        cfg!(target_arch = "x86_64") || one_clock()
    }

    /// Linux, Android: the cores this thread may run on (its affinity) all belong to one cpufreq policy. False when
    /// that cannot be read.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn one_clock() -> bool {
        let read = |p: &str| std::fs::read_to_string(p).ok();
        let Some(cpus) = read("/proc/thread-self/status").and_then(|s| s.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:")).and_then(cpu_list)) else {
            return false;
        };
        let mut first = None;
        for c in cpus {
            let Some(policy) = read(&format!("/sys/devices/system/cpu/cpu{c}/cpufreq/related_cpus")).and_then(|s| cpu_list(&s)) else {
                return false;
            };
            if *first.get_or_insert_with(|| policy.clone()) != policy {
                return false;
            }
        }
        first.is_some()
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn one_clock() -> bool {
        false
    }

    /// A kernel CPU list ("0-5,7", or "6 7" as cpufreq writes it) -> its CPUs; None if malformed.
    #[cfg_attr(not(any(target_os = "linux", target_os = "android", test)), allow(dead_code))]
    fn cpu_list(s: &str) -> Option<Vec<usize>> {
        let mut v = Vec::new();
        for part in s.split(|c: char| c == ',' || c.is_whitespace()).filter(|p| !p.is_empty()) {
            let (a, b) = part.split_once('-').unwrap_or((part, part));
            let (a, b) = (a.trim().parse::<usize>().ok()?, b.trim().parse::<usize>().ok()?);
            if b < a || b - a > 4096 {
                return None;
            }
            v.extend(a..=b);
        }
        Some(v)
    }

    /// A host's parallel-for: run task(i) for every i in 0..n, possibly at the same time, and return when all have
    /// finished.
    pub type ParallelFor = Box<dyn Fn(usize, &(dyn Fn(usize) + Sync)) + Send + Sync>;

    /// How tasks run.
    pub enum Exec {
        Pool(Pool),
        Host(ParallelFor, usize),
    }

    impl Exec {
        fn threads(&self) -> usize {
            match self {
                Exec::Pool(p) => p.threads,
                Exec::Host(_, t) => *t,
            }
        }
    }

    thread_local! {
        /// The executor of the pass running on this thread (none: one thread; also none inside a task, so nested
        /// steps run where they are).
        static CUR: RefCell<Option<Arc<Exec>>> = const { RefCell::new(None) };
    }

    fn swap(e: Option<Arc<Exec>>) -> Option<Arc<Exec>> {
        CUR.with(|c| c.replace(e))
    }

    /// Puts the previous executor back, also when unwinding.
    struct Restore(Option<Arc<Exec>>);

    impl Drop for Restore {
        fn drop(&mut self) {
            swap(self.0.take());
        }
    }

    /// Runs `body` with `exec` as this thread's executor.
    pub fn enter<R>(exec: Option<&Arc<Exec>>, body: impl FnOnce() -> R) -> R {
        let _r = Restore(swap(exec.cloned()));
        body()
    }

    /// Runs `body` on this thread alone (its steps are not split).
    pub fn serial<R>(body: impl FnOnce() -> R) -> R {
        let _r = Restore(swap(None));
        body()
    }

    /// Threads the current step may use.
    pub fn threads() -> usize {
        CUR.with(|c| c.borrow().as_ref().map_or(1, |e| e.threads()))
    }

    /// f(i) for every i in 0..n, on the current executor's threads.
    pub fn run(n: usize, f: &(dyn Fn(usize) + Sync)) {
        let ex = if n > 1 { CUR.with(|c| c.borrow().clone()) } else { None };
        let Some(ex) = ex.filter(|e| e.threads() > 1) else {
            (0..n).for_each(f);
            return;
        };
        let _r = Restore(swap(None)); // tasks run single-threaded inside
        match &*ex {
            Exec::Pool(p) => p.run(n, f),
            Exec::Host(h, _) => h(n, f),
        }
    }

    /// f(i, chunk i) for the chunks of `out` (`len` values each, the last one shorter), in parallel when the current
    /// executor has threads and the step (`work` in all) is worth it.
    pub fn chunks_mut<T: Send>(out: &mut [T], len: usize, work: usize, f: impl Fn(usize, &mut [T]) + Sync) {
        if work < super::MIN_WORK || out.len() <= len || threads() < 2 {
            out.chunks_mut(len.max(1)).enumerate().for_each(|(i, c)| f(i, c));
            return;
        }
        // one lock per chunk, each taken once: disjoint &mut access, checked by the compiler
        let parts: Vec<Mutex<&mut [T]>> = out.chunks_mut(len.max(1)).map(Mutex::new).collect();
        run(parts.len(), &|i| f(i, &mut parts[i].lock().unwrap_or_else(|e| e.into_inner())));
    }

    /// [f(0), .., f(n - 1)], in parallel when the current executor has threads and the step (`work` in all) is
    /// worth it.
    pub fn map<T: Send>(n: usize, work: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        if work < super::MIN_WORK || n < 2 || threads() < 2 {
            return (0..n).map(f).collect();
        }
        let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
        run(n, &|i| *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(f(i)));
        slots.into_iter().map(|s| s.into_inner().unwrap_or_else(|e| e.into_inner()).expect("every task ran")).collect()
    }

    // ------------------------------------------------------------------------------------------ the built-in pool

    /// `threads - 1` workers (the caller is the last thread), started on first use. Workers take task indices from a
    /// shared counter: a slower or busier core just takes fewer.
    pub struct Pool {
        threads: usize,
        sh: Arc<Shared>,
        workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
        busy: Mutex<()>,
    }

    struct Shared {
        st: Mutex<State>,
        wake: Condvar,
        done: Condvar,
        // copies of `State::epoch` and `State::active` to spin on (the lock decides)
        epoch: AtomicU64,
        active: AtomicUsize,
        // when the current job was handed out (ns since t0), and the mean time a sleeping worker took to start on one
        t0: std::time::Instant,
        posted: AtomicU64,
        wake_ns: AtomicU64,
        // waits spin for wake_ns (`timed_spin`, decided when the workers start)
        timed: AtomicBool,
    }

    #[derive(Default)]
    struct State {
        job: Option<Job>,
        epoch: u64,
        active: usize, // workers inside the current job
        quit: bool,
        panicked: bool,
    }

    /// The task of one `run`, lent to the workers: valid while `run` waits for them (see `Pool::run`).
    #[derive(Clone, Copy)]
    struct Job {
        f: *const (dyn Fn(usize) + Sync),
        next: *const std::sync::atomic::AtomicUsize,
        n: usize,
    }

    // SAFETY: the pointees are Sync (a Sync closure, an atomic) and outlive every use (`Pool::run`).
    #[allow(unsafe_code)]
    unsafe impl Send for Job {}

    impl Pool {
        pub fn new(threads: usize) -> Pool {
            Pool {
                threads: threads.max(1),
                sh: Arc::new(Shared {
                    st: Mutex::new(State::default()),
                    wake: Condvar::new(),
                    done: Condvar::new(),
                    epoch: AtomicU64::new(0),
                    active: AtomicUsize::new(0),
                    t0: std::time::Instant::now(),
                    posted: AtomicU64::new(0),
                    wake_ns: AtomicU64::new(WAKE0),
                    timed: AtomicBool::new(cfg!(target_arch = "x86_64")),
                }),
                workers: Mutex::new(Vec::new()),
                busy: Mutex::new(()),
            }
        }

        fn lock(&self) -> std::sync::MutexGuard<'_, State> {
            self.sh.st.lock().unwrap_or_else(|e| e.into_inner())
        }

        #[allow(unsafe_code)]
        fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
            // one parallel step at a time (several sessions on one Models): the others run where they are
            let Ok(_busy) = self.busy.try_lock() else {
                (0..n).for_each(f);
                return;
            };
            self.start();
            let next = AtomicUsize::new(0);
            // SAFETY: only the lifetime is erased. Workers use `f` and `next` only between taking the job (under the
            // lock, `active += 1`) and reporting back (`active -= 1`); `Finish` (below; it also runs when unwinding)
            // withdraws the job and waits for `active == 0` before this frame - and `f`, `next` - can go.
            let f: &'static (dyn Fn(usize) + Sync) = unsafe { core::mem::transmute(f) };
            {
                let mut st = self.lock();
                st.job = Some(Job { f, next: &next, n });
                self.sh.posted.store(self.sh.t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                st.epoch += 1;
                st.panicked = false;
                self.sh.epoch.store(st.epoch, Ordering::Release);
            }
            self.sh.wake.notify_all();
            struct Finish<'a>(&'a Pool);
            impl Drop for Finish<'_> {
                fn drop(&mut self) {
                    let mut st = self.0.lock();
                    st.job = None;
                    if st.active > 0 {
                        drop(st);
                        let sh = &self.0.sh;
                        spin_for(sh.wake_ns.load(Ordering::Relaxed), sh.timed.load(Ordering::Relaxed), || sh.active.load(Ordering::Acquire) == 0);
                        st = self.0.lock();
                    }
                    while st.active > 0 {
                        st = self.0.sh.done.wait(st).unwrap_or_else(|e| e.into_inner());
                    }
                }
            }
            let fin = Finish(self);
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                f(i);
            }
            drop(fin);
            if self.lock().panicked {
                panic!("tinyasr: a parallel task panicked");
            }
        }

        /// The workers, spawned from the calling thread (they inherit its CPU affinity).
        fn start(&self) {
            let mut w = self.workers.lock().unwrap_or_else(|e| e.into_inner());
            if w.is_empty() && self.threads > 1 {
                // the workers inherit this thread's affinity: where they may run decides how they wait
                self.sh.timed.store(timed_spin(), Ordering::Relaxed);
            }
            while w.len() + 1 < self.threads {
                let sh = self.sh.clone();
                match std::thread::Builder::new().name("tinyasr".into()).spawn(move || worker(&sh)) {
                    Ok(h) => w.push(h),
                    Err(_) => break, // fewer threads: the work still all gets done
                }
            }
        }
    }

    #[allow(unsafe_code)]
    fn worker(sh: &Shared) {
        let mut seen = 0;
        loop {
            spin_for(sh.wake_ns.load(Ordering::Relaxed), sh.timed.load(Ordering::Relaxed), || sh.epoch.load(Ordering::Acquire) != seen);
            let job = {
                let mut st = sh.st.lock().unwrap_or_else(|e| e.into_inner());
                let mut slept = false;
                loop {
                    if st.quit {
                        return;
                    }
                    if let Some(j) = st.job.filter(|_| st.epoch != seen) {
                        seen = st.epoch;
                        st.active += 1;
                        sh.active.store(st.active, Ordering::Release);
                        if slept {
                            // how long this wake-up took (from the job's hand-out): the mean of the recent ones
                            let ns = (sh.t0.elapsed().as_nanos() as u64).saturating_sub(sh.posted.load(Ordering::Relaxed)).min(1_000_000);
                            let old = sh.wake_ns.load(Ordering::Relaxed);
                            sh.wake_ns.store((7 * old + ns) / 8, Ordering::Relaxed);
                        }
                        break j;
                    }
                    st = sh.wake.wait(st).unwrap_or_else(|e| e.into_inner());
                    slept = true;
                }
            };
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // SAFETY: the job is valid until this worker reports back below (`Pool::run`).
                let (f, next) = unsafe { (&*job.f, &*job.next) };
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= job.n {
                        break;
                    }
                    f(i);
                }
            }));
            let mut st = sh.st.lock().unwrap_or_else(|e| e.into_inner());
            st.panicked |= r.is_err();
            st.active -= 1;
            sh.active.store(st.active, Ordering::Release);
            if st.active == 0 {
                sh.done.notify_all();
            }
        }
    }

    impl Drop for Pool {
        fn drop(&mut self) {
            self.lock().quit = true;
            self.sh.wake.notify_all();
            for h in self.workers.get_mut().unwrap_or_else(|e| e.into_inner()).drain(..) {
                let _ = h.join();
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cpu_lists() {
            assert_eq!(cpu_list("0-5,7\n"), Some(vec![0, 1, 2, 3, 4, 5, 7]));
            assert_eq!(cpu_list("6 7\n"), Some(vec![6, 7]));
            assert_eq!(cpu_list("6,7"), Some(vec![6, 7]));
            assert_eq!(cpu_list(""), Some(vec![]));
            assert_eq!(cpu_list("3-1"), None);
            assert_eq!(cpu_list("x"), None);
            // whatever this machine says, asking never fails
            let _ = timed_spin();
        }

        #[test]
        fn pool_runs_every_task_once() {
            let ex = Arc::new(Exec::Pool(Pool::new(3)));
            enter(Some(&ex), || {
                for n in [0usize, 1, 2, 7, 1000] {
                    let mut v = vec![0u32; n * 3];
                    chunks_mut(&mut v, 3, usize::MAX, |i, c| c.iter_mut().for_each(|x| *x += i as u32 + 1));
                    assert!(v.chunks(3).enumerate().all(|(i, c)| c.iter().all(|&x| x == i as u32 + 1)));
                    assert_eq!(map(n, usize::MAX, |i| i * i), (0..n).map(|i| i * i).collect::<Vec<_>>());
                }
                // nested steps run inside the task
                let r = map(4, usize::MAX, |i| map(3, usize::MAX, |j| i * 10 + j));
                assert_eq!(r[3], vec![30, 31, 32]);
            });
        }

        #[test]
        fn host_parallel_for() {
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let c = calls.clone();
            let host: ParallelFor = Box::new(move |n, f| {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                std::thread::scope(|s| {
                    for i in 0..n {
                        s.spawn(move || f(i));
                    }
                });
            });
            let ex = Arc::new(Exec::Host(host, 4));
            let v = enter(Some(&ex), || map(5, usize::MAX, |i| i + 1));
            assert_eq!(v, vec![1, 2, 3, 4, 5]);
            assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        }

        #[test]
        fn task_panic_reaches_the_caller() {
            let ex = Arc::new(Exec::Pool(Pool::new(2)));
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| enter(Some(&ex), || map(64, usize::MAX, |i| if i == 63 { panic!("boom") } else { i }))));
            assert!(r.is_err());
            // and the pool still works
            assert_eq!(enter(Some(&ex), || map(3, usize::MAX, |i| i)), vec![0, 1, 2]);
        }
    }
}

/// Without std: one thread.
#[cfg(not(feature = "std"))]
mod imp {
    use alloc::vec::Vec;

    pub fn threads() -> usize {
        1
    }

    pub fn serial<R>(body: impl FnOnce() -> R) -> R {
        body()
    }

    pub fn chunks_mut<T>(out: &mut [T], len: usize, _work: usize, f: impl Fn(usize, &mut [T])) {
        out.chunks_mut(len.max(1)).enumerate().for_each(|(i, c)| f(i, c));
    }

    pub fn map<T>(n: usize, _work: usize, f: impl Fn(usize) -> T) -> Vec<T> {
        (0..n).map(f).collect()
    }
}

#[cfg(not(feature = "std"))]
pub use imp::*;
