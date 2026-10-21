//! `MemoTable::get_or_insert`: a miss restores the stored entry and every
//! stored query dep under it, seeding each one only after its deps.

#[cfg(test)]
use std::cell::RefCell;
use std::collections::HashMap;

use incr_core::persist_hooks::{
    frame_doomed, pending_handle, seed_base, Deferred, Reuse, SeedBase, SeedDep,
};
use incr_core::{check_value_code, with_value_code, ErasedIncr, Incr, Runtime};

use super::handle::{MemoKey, MemoTable, MemoValue};
use super::mapping::map_stored;
use super::memo_core::{CoreRef, IndexGen, MemoCells, MemoState};
use super::table_core::{decode, encode, Seeded, TableCore};
use super::{fingerprint, Entry, StoredDep};

impl<C: MemoCells> CoreRef<C> {
    pub(crate) fn get_or_insert<K: MemoKey, T: MemoValue>(
        &self,
        rt: &Runtime<C>,
        table: MemoTable<K, T>,
        key: K,
    ) -> Incr<T> {
        let t: &TableCore<K, T, C> = self.table(table.memo, table.index);
        if let Some(handle) = t.probe(rt, &key) {
            return handle;
        }
        if frame_doomed(rt) {
            return pending_handle(rt);
        }
        check_value_code();
        let Some(bytes) = encode(&key) else {
            return t.cold(self, rt, key);
        };
        if !self.index().holds(table.index, &bytes) {
            return t.cold(self, rt, key);
        }
        let mut bag = Deferred::new();
        let handle = {
            let mut state = self.lock_memo();
            self.restore(&mut state, rt, t, key, bytes, &mut bag)
        };
        bag.drain(rt);
        handle
    }

    fn restore<K: MemoKey, T: MemoValue>(
        &self,
        state: &mut MemoState,
        rt: &Runtime<C>,
        t: &TableCore<K, T, C>,
        key: K,
        bytes: Vec<u8>,
        bag: &mut Deferred<C>,
    ) -> Incr<T> {
        if let Some(handle) = t.probe(rt, &key) {
            return handle;
        }
        let index = self.index();
        let base = seed_base(rt);
        #[cfg(test)]
        park(Park::AfterBase);
        let mut walk = Walk {
            core: self,
            rt,
            index: &index,
            visits: Vec::new(),
            ids: HashMap::new(),
        };
        let root = walk.visit(t.index, bytes);
        if !walk.read(state, root) {
            return t.cold(self, rt, key);
        }
        walk.run(state, root, &base, bag);
        // The root is seeded here, typed, so its stored key is checked
        // against the live one and its handle comes back typed.
        let entry = walk.visits[root].entry.as_ref().expect("read above");
        let same = with_value_code(|| decode::<K>(&entry.key).is_some_and(|k| k == key));
        if !same {
            corrupt(state);
            return t.cold(self, rt, key);
        }
        let Some((deps, reuse)) = walk.seed_deps(state, root) else {
            state.stats.cold += 1;
            return t.cold(self, rt, key);
        };
        #[cfg(test)]
        park(Park::BeforeSeed);
        let (handle, seeded) = t.seed(self, rt, key, &entry.value, &deps, reuse, &base, bag);
        record(state, handle.erase(), entry, &base, seeded);
        handle
    }
}

/// Where a restore calls this thread's [`PARK`] hook.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Park {
    /// The base is taken and no fingerprint is read yet.
    AfterBase,
    /// One entry's fingerprints are read and their `DepIndex` hold is
    /// released; the entry isn't seeded yet.
    BeforeSeed,
}

#[cfg(test)]
type ParkHook = Box<dyn Fn(Park)>;

#[cfg(test)]
thread_local! {
    static PARK: RefCell<Option<ParkHook>> = const { RefCell::new(None) };
}

#[cfg(test)]
fn park(point: Park) {
    PARK.with(|hook| {
        if let Some(hook) = &*hook.borrow() {
            hook(point);
        }
    });
}

fn corrupt(state: &mut MemoState) {
    state.stats.cold += 1;
    state.stats.corrupt += 1;
}

fn record(
    state: &mut MemoState,
    handle: ErasedIncr,
    entry: &Entry,
    base: &SeedBase,
    seeded: Seeded,
) {
    match seeded {
        Seeded::As(reuse) => {
            state
                .seed_info
                .insert(handle, (fingerprint(&entry.value), base.w0));
            match reuse {
                Reuse::IfUnchanged => state.stats.reused += 1,
                Reuse::RecomputeFirst => state.stats.revalidated += 1,
            }
        }
        Seeded::Live => {}
        Seeded::Corrupt => corrupt(state),
    }
}

/// A stored entry the walk reached, by live table and key bytes.
struct Visit {
    table: u32,
    key: Vec<u8>,
    entry: Option<Entry>,
    /// Per stored query dep, in order, the visit that restores it; `None`
    /// when the dep's table isn't live, which leaves this entry cold.
    children: Vec<Option<usize>>,
    /// Settled: live, seeded or cold. `None` once settled means the key
    /// doesn't decode, so nothing depending on it can be seeded.
    handle: Option<ErasedIncr>,
    mark: Option<Mark>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark {
    OnStack,
    Done,
}

struct Walk<'a, C: MemoCells> {
    core: &'a CoreRef<C>,
    rt: &'a Runtime<C>,
    index: &'a IndexGen,
    visits: Vec<Visit>,
    ids: HashMap<(u32, Vec<u8>), usize>,
}

impl<C: MemoCells> Walk<'_, C> {
    /// The one visit per `(table, key)`, unread until the walk reaches it.
    fn visit(&mut self, table: u32, key: Vec<u8>) -> usize {
        if let Some(&id) = self.ids.get(&(table, key.clone())) {
            return id;
        }
        let id = self.visits.len();
        self.visits.push(Visit {
            table,
            key: key.clone(),
            entry: None,
            children: Vec::new(),
            handle: None,
            mark: None,
        });
        self.ids.insert((table, key), id);
        id
    }

    /// Reads a visit's stored entry and names its query deps; `false` if
    /// it has none, counting a frame that fails to decode as corrupt.
    fn read(&mut self, state: &mut MemoState, id: usize) -> bool {
        let visit = &self.visits[id];
        let entry = match self
            .index
            .read(&*self.core.backend, visit.table, &visit.key)
        {
            Ok(Some(entry)) => entry,
            Ok(None) => return false,
            Err(()) => {
                corrupt(state);
                return false;
            }
        };
        let children = entry
            .deps
            .iter()
            .filter_map(|dep| match dep {
                StoredDep::Query { table, key, .. } => Some((*table, key)),
                _ => None,
            })
            .map(|(table, key)| {
                let table = self.index.live_table(table)?;
                Some(self.visit(table, key.clone()))
            })
            .collect();
        let visit = &mut self.visits[id];
        visit.entry = Some(entry);
        visit.children = children;
        true
    }

    /// Restores every visit under `root`, deps first, on an explicit stack,
    /// and leaves the root itself to the typed caller.
    fn run(&mut self, state: &mut MemoState, root: usize, base: &SeedBase, bag: &mut Deferred<C>) {
        self.visits[root].mark = Some(Mark::OnStack);
        let mut stack = vec![(root, 0_usize)];
        while let Some(&(id, next)) = stack.last() {
            let Some(&child) = self.visits[id].children.get(next) else {
                stack.pop();
                if id != root {
                    self.finish(state, id, base, bag);
                }
                self.visits[id].mark = Some(Mark::Done);
                continue;
            };
            stack.last_mut().expect("non-empty").1 += 1;
            let Some(child) = child else { continue };
            match self.visits[child].mark {
                Some(Mark::Done) => {}
                // A back edge, which only a corrupt memo holds: the dep is
                // treated as having no entry.
                Some(Mark::OnStack) => {
                    if self.visits[child].handle.is_none() {
                        self.cold(state, child);
                    }
                }
                None => {
                    if self.open(state, child) {
                        self.visits[child].mark = Some(Mark::OnStack);
                        stack.push((child, 0));
                    } else {
                        self.visits[child].mark = Some(Mark::Done);
                    }
                }
            }
        }
    }

    /// Settles a dep that needs no restore, or reads it: `true` if it has
    /// an entry to restore. A live node is used as it is, and a dep with no
    /// entry is created cold through its own table.
    fn open(&mut self, state: &mut MemoState, id: usize) -> bool {
        let visit = &self.visits[id];
        let table = &self.core.tables[visit.table as usize];
        match table.probe_bytes(self.rt, &visit.key) {
            Ok(Some(live)) => {
                self.visits[id].handle = Some(live);
                false
            }
            Ok(None) => {
                if self.read(state, id) {
                    true
                } else {
                    self.cold(state, id);
                    false
                }
            }
            Err(()) => {
                corrupt(state);
                false
            }
        }
    }

    /// `false`, counted corrupt, if the key doesn't decode.
    fn cold(&mut self, state: &mut MemoState, id: usize) -> bool {
        let visit = &self.visits[id];
        let table = &self.core.tables[visit.table as usize];
        let handle = table.cold_bytes(self.core, self.rt, &visit.key);
        if handle.is_none() {
            corrupt(state);
        }
        self.visits[id].handle = handle;
        handle.is_some()
    }

    /// Seeds a dep whose own deps are all settled, or creates it cold.
    fn finish(&mut self, state: &mut MemoState, id: usize, base: &SeedBase, bag: &mut Deferred<C>) {
        if self.visits[id].handle.is_some() {
            return;
        }
        let Some((deps, reuse)) = self.seed_deps(state, id) else {
            if self.cold(state, id) {
                state.stats.cold += 1;
            }
            return;
        };
        #[cfg(test)]
        park(Park::BeforeSeed);
        let visit = &self.visits[id];
        let entry = visit.entry.as_ref().expect("an opened visit has its entry");
        let table = &self.core.tables[visit.table as usize];
        let seeded = table.seed_bytes(
            self.core,
            self.rt,
            &visit.key,
            &entry.value,
            &deps,
            reuse,
            base,
            bag,
        );
        match seeded {
            Some((handle, seeded)) => {
                record(state, handle, entry, base, seeded);
                self.visits[id].handle = Some(handle);
            }
            None => corrupt(state),
        }
    }

    /// An entry's deps as seed deps, with the `Reuse` they allow; `None`
    /// when a dep maps to nothing, so the entry goes cold. Mapped fps are
    /// read here, after the restore took its base.
    fn seed_deps(&self, state: &MemoState, id: usize) -> Option<(Vec<SeedDep>, Reuse)> {
        let visit = &self.visits[id];
        let entry = visit.entry.as_ref()?;
        let mut deps = Vec::with_capacity(entry.deps.len());
        let mut unchanged = true;
        let mut children = visit.children.iter();
        for dep in &entry.deps {
            if let StoredDep::External { name, .. } = dep {
                self.core.external(self.rt, name);
            }
        }
        let map = self.core.deps.read();
        for dep in &entry.deps {
            match dep {
                StoredDep::Query { fp, .. } => {
                    let child = (*children.next()?)?;
                    let handle = self.visits[child].handle?;
                    let trusted = state
                        .seed_info
                        .get(&handle)
                        .filter(|(seeded, _)| seeded == fp)
                        .map(|&(_, stamp)| stamp);
                    unchanged &= trusted.is_some();
                    deps.push(SeedDep {
                        dep: handle,
                        trusted_stamp: trusted,
                    });
                }
                stored => {
                    let (handle, same) = map_stored(&map, self.rt, self.core.id, stored)?;
                    unchanged &= same;
                    deps.push(SeedDep {
                        dep: handle,
                        trusted_stamp: None,
                    });
                }
            }
        }
        let reuse = if unchanged {
            Reuse::IfUnchanged
        } else {
            Reuse::RecomputeFirst
        };
        Some((deps, reuse))
    }
}

#[cfg(test)]
#[path = "restore_tests.rs"]
mod tests;
