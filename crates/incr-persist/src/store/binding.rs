use std::any::Any;

use incr_core::{Cells, ErasedIncr, Incr, Runtime, Value};
use serde::de::DeserializeOwned;
use serde::Serialize;

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

    /// Applies `record`, one of this key's records in position order, to the
    /// node. `before` is the key's logged state without the record.
    fn apply(&mut self, rt: &Runtime<C>, record: &Record, before: Option<&LoggedKey>)
        -> Result<()>;

    /// Sets the node to exactly what `logged` holds, or to the binding's
    /// default when the log no longer knows the key. A collection binding
    /// holds its elements itself, so it takes them out of `logged`, but only
    /// once they all decoded; on an error nothing has changed. When
    /// `changes` is given, a collection binding pushes a `Remove` or
    /// `Insert` onto it per element occurrence it changed; an input's change
    /// is visible in the logged state alone, so input bindings push nothing.
    fn replace(
        &mut self,
        rt: &Runtime<C>,
        logged: Option<&mut LoggedKey>,
        changes: Option<&mut Vec<Record>>,
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
}

impl<T: Value + DeserializeOwned> InputBinding<T> {
    fn value_of(&self, logged: Option<&LoggedKey>) -> Result<T> {
        match logged.and_then(LoggedKey::current) {
            Some(bytes) => decode(&self.key, bytes),
            None => Ok(self.default.clone()),
        }
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

    fn apply(
        &mut self,
        rt: &Runtime<C>,
        record: &Record,
        before: Option<&LoggedKey>,
    ) -> Result<()> {
        let value = match record {
            Record::Set { value, .. } => decode(&self.key, value)?,
            Record::Delete { .. } => self.default.clone(),
            // Only the first Declare since the key was last forgotten is
            // live, as in the folded log.
            Record::Declare { default, .. }
                if before.is_none_or(|logged| logged.declared.is_none()) =>
            {
                self.default = decode(&self.key, default)?;
                if before.and_then(LoggedKey::current).is_some() {
                    return Ok(());
                }
                self.default.clone()
            }
            _ => return Ok(()),
        };
        rt.set(self.handle, value);
        Ok(())
    }

    fn replace(
        &mut self,
        rt: &Runtime<C>,
        logged: Option<&mut LoggedKey>,
        _changes: Option<&mut Vec<Record>>,
    ) -> Result<()> {
        let logged = logged.as_deref();
        let declared = logged.and_then(|logged| logged.declared.as_ref());
        let default = declared
            .map(|declared| decode(&self.key, &declared.default))
            .transpose()?;
        let value = self.value_of(logged)?;
        if let Some(default) = default {
            self.default = default;
        }
        rt.set(self.handle, value);
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_bypass(&self, rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<()> {
        let held = rt.get(self.handle);
        let applied = self.value_of(logged)?;
        // Equal bytes cover values unequal to themselves, such as NaN.
        if held == applied || encode(&held)? == encode(&applied)? {
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
                decode::<T>(&self.key, value)? == decode::<T>(&self.key, &declared.default)?
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
    postcard::to_stdvec(value).map_err(Error::Encode)
}

/// Decodes a value logged under `key`, rejecting bytes left over, since
/// leftovers mean the bytes were written as some other type.
pub(crate) fn decode<T: DeserializeOwned>(key: &str, bytes: &[u8]) -> Result<T> {
    match postcard::take_from_bytes(bytes) {
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
