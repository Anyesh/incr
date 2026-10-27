use std::sync::Arc;

use super::{
    left_rec_since, note_rec_left, own_episode, own_episodes, recs_left_count, Held, Look,
    OWN_EPISODES,
};
use crate::cycle::{self, Cycle, Erased};
use crate::deferred::Unbuilt;
use crate::handle::{ErasedIncr, RuntimeId};
use crate::node::NodeId;
use crate::replay;
use crate::runtime::GroupInfo;

fn value(v: u32) -> Option<Erased> {
    Some(Box::new(v))
}

fn read(b: &Option<Erased>) -> Option<u32> {
    b.as_ref().map(|v| *v.downcast_ref::<u32>().expect("a u32"))
}

fn read_unbuilt(b: &Option<Unbuilt>) -> Option<u32> {
    b.as_ref()
        .map(|v| *v.value().downcast_ref::<u32>().expect("a u32"))
}

#[test]
fn a_real_look_settles_what_a_speculative_run_wrote() {
    let mut e = Held::new(0, 1, false, true);
    let (replaced, spec) = e.record(Look::REAL, vec![(NodeId(2), 0)], value(1), false, 0, false);
    assert!(replaced.is_none() && spec.is_none());
    let (replaced, spec) = e.record(
        Look::SPECULATIVE,
        vec![(NodeId(3), 0)],
        value(7777),
        true,
        cycle::POISONED,
        false,
    );
    assert_eq!(read(&replaced), Some(1));
    assert!(spec.is_none());
    {
        let (seen, settled) = e.look(Look::SPECULATIVE);
        assert!(settled.is_none());
        assert_eq!(read(seen.buf()), Some(7777));
        assert_eq!(seen.last(), &[(NodeId(3), 0)]);
        assert!(seen.opened() && seen.poison() != 0);
    }
    assert!(e.spec_written());
    let (seen, settled) = e.look(Look::REAL);
    assert_eq!(read_unbuilt(&settled), Some(7777));
    assert!(seen.buf().is_none());
    assert_eq!(seen.deps(), &[(NodeId(2), 0)]);
    assert_eq!(seen.last(), &[(NodeId(2), 0)]);
    assert_eq!(seen.installed(), &[NodeId(2)]);
    assert!(!seen.opened() && seen.poison() == 0);
    assert!(!seen.spec_written());
}

#[test]
fn a_real_write_settles_before_it_records() {
    let mut e = Held::new(0, 1, false, true);
    e.record(
        Look::REAL,
        vec![(NodeId(2), 0)],
        value(1),
        false,
        cycle::POISONED,
        false,
    );
    e.record(
        Look::SPECULATIVE,
        vec![(NodeId(3), 0)],
        value(7777),
        false,
        0,
        false,
    );
    let (replaced, spec) = e.record(Look::REAL, vec![(NodeId(4), 0)], value(5), true, 0, false);
    assert!(replaced.is_none());
    assert_eq!(read_unbuilt(&spec), Some(7777));
    let (seen, settled) = e.look(Look::REAL);
    assert!(settled.is_none());
    assert_eq!(read(seen.buf()), Some(5));
    assert_eq!(seen.deps(), &[(NodeId(2), 0), (NodeId(4), 0)]);
    assert_eq!(seen.last(), &[(NodeId(4), 0)]);
    assert_eq!(seen.installed(), &[NodeId(2), NodeId(4)]);
    assert!(seen.opened() && seen.poison() != 0);
}

#[test]
fn an_adopted_speculative_run_survives_a_real_look() {
    let mut e = Held::new(0, 1, false, true);
    e.record(Look::REAL, vec![(NodeId(2), 0)], value(1), false, 0, false);
    e.record(
        Look::SPECULATIVE,
        vec![(NodeId(3), 0)],
        value(7777),
        false,
        0,
        false,
    );
    e.adopt();
    let (seen, settled) = e.into_look(Look::REAL);
    assert!(settled.is_none());
    assert_eq!(read(seen.buf()), Some(7777));
    assert_eq!(seen.deps(), &[(NodeId(2), 0), (NodeId(3), 0)]);
}

#[test]
fn a_settle_restores_the_record_from_before_the_first_speculative_run() {
    let mut e = Held::new(0, 1, false, true);
    e.record(Look::REAL, vec![(NodeId(2), 0)], None, false, 0, false);
    e.record(
        Look::SPECULATIVE,
        vec![(NodeId(3), 0)],
        value(7),
        true,
        0,
        false,
    );
    let (replaced, _) = e.record(
        Look::SPECULATIVE,
        vec![(NodeId(4), 0)],
        value(8),
        false,
        cycle::POISONED,
        false,
    );
    assert_eq!(read(&replaced), Some(7));
    let (seen, settled) = e.look(Look::REAL);
    assert_eq!(read_unbuilt(&settled), Some(8));
    assert_eq!(seen.deps(), &[(NodeId(2), 0)]);
    assert_eq!(seen.last(), &[(NodeId(2), 0)]);
    assert_eq!(seen.installed(), &[NodeId(2)]);
    assert!(!seen.opened() && seen.poison() == 0);
}

#[test]
fn a_speculative_look_or_a_discard_keeps_the_mark() {
    let mut e = Held::new(0, 1, false, true);
    e.record(Look::REAL, vec![(NodeId(2), 0)], None, false, 0, false);
    e.record(
        Look::SPECULATIVE,
        vec![(NodeId(3), 0)],
        value(7),
        false,
        0,
        false,
    );
    assert!(e.discard_buf().is_some());
    assert!(e.spec_written());
    let (seen, settled) = e.into_look(Look::SPECULATIVE);
    assert!(settled.is_none());
    assert!(seen.spec_written());
    assert_eq!(seen.last(), &[(NodeId(3), 0)]);
    let (seen, settled) = seen.into_held().into_look(Look::REAL);
    assert!(settled.is_none());
    assert_eq!(seen.last(), &[(NodeId(2), 0)]);
}

#[test]
fn a_look_here_follows_the_thread_speculation() {
    assert_eq!(Look::here(), Look::REAL);
    assert_eq!(Look::here_or(true), Look::SPECULATIVE);
    let _speculating = replay::Speculating::enter(RuntimeId::SENTINEL);
    assert_eq!(Look::here(), Look::SPECULATIVE);
    assert_eq!(Look::here_or(false), Look::SPECULATIVE);
}

#[test]
fn a_thread_counts_episodes_on_a_bounded_set_of_runtimes() {
    let rts: Vec<RuntimeId> = (0..20).map(|_| RuntimeId::allocate()).collect();
    for &rt in &rts {
        own_episode(rt, 1);
    }
    own_episode(rts[19], 1);
    assert_eq!(own_episodes(rts[19]), 2);
    assert_eq!(own_episodes(rts[0]), 0);
    assert!(OWN_EPISODES.with(|o| o.borrow().len()) < rts.len());
}

/// Entries of a Local runtime L and Shared runtimes S and T nested as
/// L, S, L, S, T, S on one thread: each runtime's first entry, and the
/// first Shared one, are told apart from the thread's outermost.
#[test]
fn entries_tell_each_runtimes_first_under_another_runtimes_outermost() {
    use super::{open_entry_of, EntryDepth};
    let (l, s, t) = (
        RuntimeId::for_test(1 << 40),
        RuntimeId::for_test((1 << 40) + 1),
        RuntimeId::for_test((1 << 40) + 2),
    );
    let flags = |e: &EntryDepth| {
        (
            e.is_outermost(),
            e.first_for_runtime(),
            e.shared_outermost(),
        )
    };
    assert!(!open_entry_of(l));
    let l1 = EntryDepth::enter(l, false);
    assert_eq!(flags(&l1), (true, true, false));
    {
        let s1 = EntryDepth::enter(s, true);
        assert_eq!(flags(&s1), (false, true, true));
        let s2 = EntryDepth::enter(s, true);
        assert_eq!(flags(&s2), (false, false, false));
        {
            let l2 = EntryDepth::enter(l, false);
            assert_eq!(flags(&l2), (false, false, false));
            let s3 = EntryDepth::enter(s, true);
            assert_eq!(flags(&s3), (false, false, false));
            let t1 = EntryDepth::enter(t, true);
            assert_eq!(flags(&t1), (false, true, false));
            let s4 = EntryDepth::enter(s, true);
            assert_eq!(flags(&s4), (false, false, false));
            assert!(open_entry_of(l) && open_entry_of(s) && open_entry_of(t));
            drop(s4);
            drop(t1);
            assert!(!open_entry_of(t));
            drop(s3);
            drop(l2);
        }
        drop(s2);
        assert!(open_entry_of(s));
        drop(s1);
    }
    assert!(!open_entry_of(s) && open_entry_of(l));
    let s5 = EntryDepth::enter(s, true);
    assert_eq!(flags(&s5), (false, true, true));
    drop(s5);
    drop(l1);
    assert!(!open_entry_of(l));
    let s6 = EntryDepth::enter(s, true);
    assert_eq!(flags(&s6), (true, true, true));
}

/// A thousand entries alternating between a Local and a Shared runtime:
/// only the first of each is its runtime's first, and every switch is
/// undone once they exit.
#[test]
fn alternating_runtimes_switch_back_on_exit() {
    use super::{open_entry_of, EntryDepth};
    let (l, s) = (
        RuntimeId::for_test(1 << 41),
        RuntimeId::for_test((1 << 41) + 1),
    );
    let entries: Vec<EntryDepth> = (0..1000)
        .map(|i| {
            let e = EntryDepth::enter(if i % 2 == 0 { l } else { s }, i % 2 == 1);
            assert_eq!(e.first_for_runtime(), i < 2, "entry {i}");
            assert_eq!(e.shared_outermost(), i == 1, "entry {i}");
            e
        })
        .collect();
    assert!(open_entry_of(l) && open_entry_of(s));
    let mut entries = entries.into_iter();
    let l1 = entries.next().unwrap();
    for e in entries.rev() {
        drop(e);
    }
    assert!(open_entry_of(l) && !open_entry_of(s));
    let s1 = EntryDepth::enter(s, true);
    assert!(!s1.is_outermost() && s1.first_for_runtime() && s1.shared_outermost());
    drop(s1);
    drop(l1);
}

/// An engine call from a destructor that raises and catches a discard of
/// its own.
struct DiscardOnDrop;

impl Drop for DiscardOnDrop {
    fn drop(&mut self) {
        super::KeptUnwind::around(|| {
            super::set_unwind_kind(Some(super::UnwindKind::Discard));
            super::set_unwind_kind(None);
        });
    }
}

/// A doomed frame runs on past its placeholder and panics in user code,
/// and a destructor's engine call raises and catches a discard: the lazy
/// doom's flight stays open for its waiters until the frame's end closes
/// it.
#[test]
fn a_destructors_discard_leaves_a_lazy_doom_in_flight() {
    crate::attempt::begin();
    super::open_lazy(1);
    assert!(crate::attempt::in_flight());
    let unwound = std::panic::catch_unwind(|| {
        let _guard = DiscardOnDrop;
        panic!("user panic past the placeholder");
    });
    assert!(unwound.is_err());
    assert!(
        crate::attempt::in_flight(),
        "the destructor's discard closed the lazy doom's flight"
    );
    assert!(!super::end_lazy(1));
    assert!(!crate::attempt::in_flight());
}

/// A walk counts as its own only the record changes this thread noted after
/// its expansion: one noted before it, which another thread's changes then
/// undid back to the same record, is another thread's.
#[test]
fn a_record_change_noted_before_the_mark_is_not_this_threads_since() {
    let rt = RuntimeId::for_test(1 << 41);
    let rec = Arc::new(GroupInfo {
        members: vec![(3, 0)],
        cycle: Cycle::new(vec![ErasedIncr::new(3, 0, rt)]),
    });
    note_rec_left(rt, 3, None);
    let mark = recs_left_count();
    assert!(left_rec_since(0, rt, 3, None));
    assert!(!left_rec_since(mark, rt, 3, None));
    note_rec_left(rt, 3, Some(rec.clone()));
    assert!(left_rec_since(mark, rt, 3, Some(&rec)));
    assert!(!left_rec_since(mark, rt, 3, None));
    assert!(!left_rec_since(mark, rt, 4, Some(&rec)));
}
