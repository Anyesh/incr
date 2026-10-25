use super::{own_episode, own_episodes, Held, Look, OWN_EPISODES};
use crate::cycle::Erased;
use crate::deferred::Unbuilt;
use crate::handle::RuntimeId;
use crate::node::NodeId;
use crate::replay;

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
    let (replaced, spec) = e.record(Look::REAL, vec![(NodeId(2), 0)], value(1), false, false);
    assert!(replaced.is_none() && spec.is_none());
    let (replaced, spec) = e.record(
        Look::SPECULATIVE,
        vec![(NodeId(3), 0)],
        value(7777),
        true,
        true,
    );
    assert_eq!(read(&replaced), Some(1));
    assert!(spec.is_none());
    {
        let (seen, settled) = e.look(Look::SPECULATIVE);
        assert!(settled.is_none());
        assert_eq!(read(seen.buf()), Some(7777));
        assert_eq!(seen.last(), &[(NodeId(3), 0)]);
        assert!(seen.opened() && seen.poisoned());
    }
    assert!(e.spec_written());
    let (seen, settled) = e.look(Look::REAL);
    assert_eq!(read_unbuilt(&settled), Some(7777));
    assert!(seen.buf().is_none());
    assert_eq!(seen.deps(), &[(NodeId(2), 0)]);
    assert_eq!(seen.last(), &[(NodeId(2), 0)]);
    assert_eq!(seen.installed(), &[NodeId(2)]);
    assert!(!seen.opened() && !seen.poisoned());
    assert!(!seen.spec_written());
}

#[test]
fn a_real_write_settles_before_it_records() {
    let mut e = Held::new(0, 1, false, true);
    e.record(Look::REAL, vec![(NodeId(2), 0)], value(1), false, true);
    e.record(
        Look::SPECULATIVE,
        vec![(NodeId(3), 0)],
        value(7777),
        false,
        false,
    );
    let (replaced, spec) = e.record(Look::REAL, vec![(NodeId(4), 0)], value(5), true, false);
    assert!(replaced.is_none());
    assert_eq!(read_unbuilt(&spec), Some(7777));
    let (seen, settled) = e.look(Look::REAL);
    assert!(settled.is_none());
    assert_eq!(read(seen.buf()), Some(5));
    assert_eq!(seen.deps(), &[(NodeId(2), 0), (NodeId(4), 0)]);
    assert_eq!(seen.last(), &[(NodeId(4), 0)]);
    assert_eq!(seen.installed(), &[NodeId(2), NodeId(4)]);
    assert!(seen.opened() && seen.poisoned());
}

#[test]
fn an_adopted_speculative_run_survives_a_real_look() {
    let mut e = Held::new(0, 1, false, true);
    e.record(Look::REAL, vec![(NodeId(2), 0)], value(1), false, false);
    e.record(
        Look::SPECULATIVE,
        vec![(NodeId(3), 0)],
        value(7777),
        false,
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
    e.record(Look::REAL, vec![(NodeId(2), 0)], None, false, false);
    e.record(
        Look::SPECULATIVE,
        vec![(NodeId(3), 0)],
        value(7),
        true,
        false,
    );
    let (replaced, _) = e.record(
        Look::SPECULATIVE,
        vec![(NodeId(4), 0)],
        value(8),
        false,
        true,
    );
    assert_eq!(read(&replaced), Some(7));
    let (seen, settled) = e.look(Look::REAL);
    assert_eq!(read_unbuilt(&settled), Some(8));
    assert_eq!(seen.deps(), &[(NodeId(2), 0)]);
    assert_eq!(seen.last(), &[(NodeId(2), 0)]);
    assert_eq!(seen.installed(), &[NodeId(2)]);
    assert!(!seen.opened() && !seen.poisoned());
}

#[test]
fn a_speculative_look_or_a_discard_keeps_the_mark() {
    let mut e = Held::new(0, 1, false, true);
    e.record(Look::REAL, vec![(NodeId(2), 0)], None, false, false);
    e.record(
        Look::SPECULATIVE,
        vec![(NodeId(3), 0)],
        value(7),
        false,
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
    let _speculating = replay::Speculating::enter();
    assert_eq!(Look::here(), Look::SPECULATIVE);
    assert_eq!(Look::here_or(false), Look::SPECULATIVE);
}

#[test]
fn a_thread_counts_episodes_on_a_bounded_set_of_runtimes() {
    let rts: Vec<RuntimeId> = (0..20).map(|_| RuntimeId::allocate()).collect();
    for &rt in &rts {
        own_episode(rt);
    }
    own_episode(rts[19]);
    assert_eq!(own_episodes(rts[19]), 2);
    assert_eq!(own_episodes(rts[0]), 0);
    assert!(OWN_EPISODES.with(|o| o.borrow().len()) < rts.len());
}

/// Entries of a Local runtime L and Shared runtimes S and T nested as
/// L, S, L, S, T on one thread: each runtime's first entry, and the
/// first Shared one, are told apart from the thread's outermost.
#[test]
fn entries_count_per_runtime_under_another_runtimes_outermost() {
    use super::{open_entry_of, EntryDepth};
    let (l, s, t) = (
        RuntimeId::for_test(u64::MAX - 21),
        RuntimeId::for_test(u64::MAX - 22),
        RuntimeId::for_test(u64::MAX - 23),
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
    let s1 = EntryDepth::enter(s, true);
    assert_eq!(flags(&s1), (false, true, true));
    let l2 = EntryDepth::enter(l, false);
    assert_eq!(flags(&l2), (false, false, false));
    let s2 = EntryDepth::enter(s, true);
    assert_eq!(flags(&s2), (false, false, false));
    let t1 = EntryDepth::enter(t, true);
    assert_eq!(flags(&t1), (false, true, false));
    assert!(open_entry_of(l) && open_entry_of(s) && open_entry_of(t));
    drop(t1);
    assert!(!open_entry_of(t));
    drop(s2);
    drop(l2);
    assert!(open_entry_of(s));
    drop(s1);
    assert!(!open_entry_of(s) && open_entry_of(l));
    let s3 = EntryDepth::enter(s, true);
    assert_eq!(flags(&s3), (false, true, true));
    drop(s3);
    drop(l1);
    assert!(!open_entry_of(l));
}

/// A third runtime's entries are listed; a runtime whose inner count ends
/// while another is listed is listed too on its next entry, never counted
/// twice.
#[test]
fn a_runtime_reentered_while_another_is_listed_counts_once() {
    use super::{open_entry_of, EntryDepth};
    let (l, s, t) = (
        RuntimeId::for_test(u64::MAX - 31),
        RuntimeId::for_test(u64::MAX - 32),
        RuntimeId::for_test(u64::MAX - 33),
    );
    let l1 = EntryDepth::enter(l, false);
    let s1 = EntryDepth::enter(s, true);
    let t1 = EntryDepth::enter(t, true);
    assert!(t1.first_for_runtime());
    drop(s1);
    assert!(!open_entry_of(s) && open_entry_of(t));
    let s2 = EntryDepth::enter(s, true);
    assert!(s2.first_for_runtime());
    let s3 = EntryDepth::enter(s, true);
    assert!(!s3.first_for_runtime());
    drop(t1);
    assert!(!open_entry_of(t) && open_entry_of(s));
    let t2 = EntryDepth::enter(t, true);
    assert!(t2.first_for_runtime());
    drop(t2);
    drop(s3);
    drop(s2);
    assert!(!open_entry_of(s) && open_entry_of(l));
    drop(l1);
}
