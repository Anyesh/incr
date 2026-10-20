use std::any::Any;

use incr_core::persist_hooks::{changed_at, set_returning_displaced};
use incr_core::value_slot::Compared;
use incr_core::{with_value_code, Cells, ErasedIncr, Incr, Runtime, Value};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::dep_index::{fp_of, DepFp, DepIndex, PanicGuard};
use super::logged::{Elements, LoggedKey};
use crate::{Error, Kind, Record, Result};

/// A key this process declared, tied to the runtime node that mirrors it.
pub(crate) trait Binding<C: Cells>: Send + Sync {
    fn kind(&self) -> Kind;

    fn key(&self) -> &str;

    /// Whether `handle`, a runtime handle erased by [`KeyRef`](crate::KeyRef),
    /// is the node this binding mirrors.
    fn is_handle(&self, handle: &dyn Any) -> bool;

    /// The runtime node mirroring the key, which undeclaring deletes.
    fn node(&self) -> ErasedIncr;

    /// The elements to hand back to the key's logged state when the
    /// binding goes, for a binding that holds them itself (see
    /// [`Binding::replace`]); `None` for one that holds nothing the log
    /// lacks. Taken before anything is removed, so a failure, such as
    /// [`Error::UnknownHandle`] for a collection whose node was deleted
    /// through the runtime and can no longer be read, changes nothing.
    fn unbind(&self, rt: &Runtime<C>) -> Result<Option<Elements>>;

    /// The key's fingerprint now, for registering it in a [`DepIndex`].
    /// Computed with no incr lock held and before the index is held.
    fn dep_fp(&self, rt: &Runtime<C>) -> Result<DepFp>;

    /// An input's mapped stamp, which registering it in a [`DepIndex`]
    /// copies; `None` for a collection.
    fn mapped_stamp(&self) -> Option<u64>;

    /// Applies `record`, one of this key's records in position order, to the
    /// node. `before` is the key's logged state without the record. With
    /// `deps`, the runtime write and the fingerprint update run inside one
    /// hold of it, and what the write displaced is dropped after the hold.
    fn apply(
        &mut self,
        rt: &Runtime<C>,
        record: &Record,
        before: Option<&LoggedKey>,
        deps: Option<&DepIndex>,
    ) -> Result<()>;

    /// Sets the node to exactly what `logged` holds, or to the binding's
    /// default when the log no longer knows the key. A collection binding
    /// holds its elements itself, so it takes them out of `logged`, but only
    /// once they all decoded; on an error nothing has changed. When
    /// `changes` is given, a collection binding pushes a `Remove` or
    /// `Insert` onto it per element occurrence it changed; an input's change
    /// is visible in the logged state alone, so input bindings push nothing.
    /// `deps` is held as for [`Binding::apply`].
    fn replace(
        &mut self,
        rt: &Runtime<C>,
        logged: Option<&mut LoggedKey>,
        changes: Option<&mut Vec<Record>>,
        deps: Option<&DepIndex>,
    ) -> Result<()>;

    /// Fails with [`Error::Bypassed`] when the node holds something other
    /// than what `logged`, the key's logged state, says it holds.
    #[cfg(debug_assertions)]
    fn check_bypass(&self, rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<()>;

    /// The records a snapshot keeps for this key, given its logged state.
    /// Empty when the key holds only its logged default, compared by
    /// decoded value, which is how compaction forgets it.
    fn fold(&self, rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<Vec<Record>>;

    fn as_any(&self) -> &dyn Any;
}

pub(crate) struct InputBinding<T: 'static> {
    pub key: String,
    pub handle: Incr<T>,
    /// The default of the key's live `Declare`, or of the last one when
    /// compaction has since forgotten the key, which is what the key holds
    /// then.
    pub default: T,
    /// The fingerprint of the bytes `default` was decoded from.
    pub default_fp: u128,
    /// The fingerprint of the bytes the runtime value was decoded from. A
    /// write the runtime treats as a no-op leaves it, because the runtime
    /// keeps the earlier value, which may differ in its bytes (-0.0 after
    /// 0.0).
    pub fp: u128,
    /// The input's `changed_at` when it was created or last written by a
    /// mapped write that wrote. A no-op write leaves it, so a bypass write
    /// through the runtime keeps `changed_at` away from it.
    pub stamp: u64,
}

impl<T: Value + DeserializeOwned> InputBinding<T> {
    /// The value `logged` gives the key, with the fingerprint of its bytes.
    fn value_of(&self, logged: Option<&LoggedKey>) -> Result<(T, u128)> {
        match logged.and_then(LoggedKey::current) {
            Some(bytes) => Ok((decode(&self.key, bytes)?, fp_of(bytes))),
            None => Ok((with_value_code(|| self.default.clone()), self.default_fp)),
        }
    }

    /// Sets the node to `value`, decoded from bytes fingerprinting as `fp`.
    fn write<C: Cells>(&mut self, rt: &Runtime<C>, value: T, fp: u128, deps: Option<&DepIndex>) {
        let mut hold = deps.map(DepIndex::write);
        let mut guard = PanicGuard::new(hold.as_deref_mut(), &self.key);
        let compared = set_returning_displaced(rt, self.handle, value);
        if matches!(compared, Compared::Replaced(_)) {
            self.fp = fp;
            self.stamp = changed_at(rt, self.handle);
            if let Some(map) = guard.deps() {
                map.set_written(&self.key, fp, self.stamp);
            }
        }
        guard.disarm();
        drop(hold);
        drop(compared);
    }
}

impl<C: Cells, T: Value + Serialize + DeserializeOwned> Binding<C> for InputBinding<T> {
    fn kind(&self) -> Kind {
        Kind::Input
    }

    fn key(&self) -> &str {
        &self.key
    }

    fn is_handle(&self, handle: &dyn Any) -> bool {
        handle.downcast_ref::<Incr<T>>() == Some(&self.handle)
    }

    fn node(&self) -> ErasedIncr {
        self.handle.erase()
    }

    fn unbind(&self, _rt: &Runtime<C>) -> Result<Option<Elements>> {
        Ok(None)
    }

    fn dep_fp(&self, _rt: &Runtime<C>) -> Result<DepFp> {
        Ok(DepFp::Known(self.fp))
    }

    fn mapped_stamp(&self) -> Option<u64> {
        Some(self.stamp)
    }

    fn apply(
        &mut self,
        rt: &Runtime<C>,
        record: &Record,
        before: Option<&LoggedKey>,
        deps: Option<&DepIndex>,
    ) -> Result<()> {
        let (value, fp) = match record {
            Record::Set { value, .. } => (decode(&self.key, value)?, fp_of(value)),
            // A forgotten key holds the default it last had, as after a
            // compaction forgot it.
            Record::Delete { .. } | Record::Forget { .. } => {
                (with_value_code(|| self.default.clone()), self.default_fp)
            }
            // Only the first Declare since the key was last forgotten is
            // live, as in the folded log.
            Record::Declare { default, .. }
                if before.is_none_or(|logged| logged.declared.is_none()) =>
            {
                self.default = decode(&self.key, default)?;
                self.default_fp = fp_of(default);
                if before.and_then(LoggedKey::current).is_some() {
                    return Ok(());
                }
                (with_value_code(|| self.default.clone()), self.default_fp)
            }
            _ => return Ok(()),
        };
        self.write(rt, value, fp, deps);
        Ok(())
    }

    fn replace(
        &mut self,
        rt: &Runtime<C>,
        logged: Option<&mut LoggedKey>,
        _changes: Option<&mut Vec<Record>>,
        deps: Option<&DepIndex>,
    ) -> Result<()> {
        let logged = logged.as_deref();
        let declared = logged.and_then(|logged| logged.declared.as_ref());
        let default = declared
            .map(|declared| {
                decode::<T>(&self.key, &declared.default)
                    .map(|default| (default, fp_of(&declared.default)))
            })
            .transpose()?;
        let (value, fp) = self.value_of(logged)?;
        if let Some((default, default_fp)) = default {
            self.default = default;
            self.default_fp = default_fp;
        }
        self.write(rt, value, fp, deps);
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_bypass(&self, rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<()> {
        let held = rt.get(self.handle);
        let (applied, _) = self.value_of(logged)?;
        // Equal bytes cover values unequal to themselves, such as NaN.
        if with_value_code(|| held == applied) || encode(&held)? == encode(&applied)? {
            Ok(())
        } else {
            Err(Error::Bypassed {
                key: self.key.clone(),
            })
        }
    }

    fn fold(&self, _rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<Vec<Record>> {
        let Some(logged) = logged else {
            return Ok(Vec::new());
        };
        let at_default = match (&logged.declared, &logged.value) {
            (Some(_), _) if !logged.elements.is_empty() => false,
            (Some(_), None) => true,
            (Some(declared), Some(value)) => {
                let value = decode::<T>(&self.key, value)?;
                let default = decode::<T>(&self.key, &declared.default)?;
                with_value_code(|| value == default)
            }
            (None, _) => false,
        };
        Ok(if at_default {
            Vec::new()
        } else {
            logged.records(&self.key)
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    with_value_code(|| postcard::to_stdvec(value)).map_err(Error::Encode)
}

/// Decodes a value logged under `key`, rejecting bytes left over, since
/// leftovers mean the bytes were written as some other type.
pub(crate) fn decode<T: DeserializeOwned>(key: &str, bytes: &[u8]) -> Result<T> {
    match with_value_code(|| postcard::take_from_bytes(bytes)) {
        Ok((value, [])) => Ok(value),
        Ok(_) => Err(Error::TypeMismatch {
            key: key.to_owned(),
            source: None,
        }),
        Err(source) => Err(Error::TypeMismatch {
            key: key.to_owned(),
            source: Some(source),
        }),
    }
}
