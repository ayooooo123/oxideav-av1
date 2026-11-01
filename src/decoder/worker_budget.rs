//! Share a decoder's worker grant between ready frames and independent filter bands.
use std::cell::RefCell;
use std::sync::{Arc, mpsc, atomic::{AtomicUsize, Ordering}};

pub(super) const WORKERS_AVAILABLE: usize = usize::MAX;

#[derive(Debug)]
pub(super) struct Budget {
    free: AtomicUsize,
    notify: mpsc::Sender<usize>,
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<Budget>>> = const { RefCell::new(None) };
}

impl Budget {
    pub(super) fn new(workers: usize, notify: mpsc::Sender<usize>) -> Arc<Self> {
        Arc::new(Self { free: AtomicUsize::new(workers.max(1)), notify })
    }

    pub(super) fn try_frame(self: &Arc<Self>) -> Option<Lease> {
        self.free.fetch_update(Ordering::AcqRel, Ordering::Acquire, |free| free.checked_sub(1))
            .ok().map(|_| Lease { pool: Some(self.clone()), count: 1, workers: 1 })
    }
}

pub(crate) struct Lease {
    pool: Option<Arc<Budget>>,
    count: usize,
    workers: usize,
}

impl Lease {
    pub(crate) fn workers(&self) -> usize { self.workers }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.count != 0 {
            if let Some(pool) = &self.pool {
                pool.free.fetch_add(self.count, Ordering::Release);
                let _ = pool.notify.send(WORKERS_AVAILABLE);
            }
        }
    }
}

pub(crate) fn borrow(requested: usize) -> Lease {
    if requested <= 1 {
        return Lease { pool: None, count: 0, workers: 1 };
    }
    CURRENT.with_borrow(|current| {
        let Some(pool) = current else {
            return Lease { pool: None, count: 0, workers: requested };
        };
        let wanted = requested - 1;
        let old = pool.free.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |free| Some(free.saturating_sub(wanted))).unwrap();
        let count = old.min(wanted);
        Lease { pool: Some(pool.clone()), count, workers: count + 1 }
    })
}

pub(super) fn within<T>(pool: Arc<Budget>, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<Arc<Budget>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with_borrow_mut(|current| *current = self.0.take());
        }
    }
    let previous = CURRENT.with_borrow_mut(|current| current.replace(pool));
    let _restore = Restore(previous);
    run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_frame_workers_are_lent_and_returned_with_notifications() {
        let (tx, rx) = mpsc::channel();
        let pool = Budget::new(4, tx);
        let frame = pool.try_frame().unwrap();
        within(pool.clone(), || {
            let bands = borrow(4);
            assert_eq!(bands.workers(), 4);
            assert!(pool.try_frame().is_none());
            assert_eq!(borrow(4).workers(), 1);
            drop(bands);
            assert_eq!(rx.recv().unwrap(), WORKERS_AVAILABLE);
            let second = pool.try_frame().unwrap();
            assert_eq!(borrow(4).workers(), 3);
            drop(second);
        });
        drop(frame);
        assert_eq!(pool.free.load(Ordering::Acquire), 4);
    }
}
