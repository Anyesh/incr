use std::any::Any;

use incr_core::{Cells, Incr, Runtime, Value};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::logged::LoggedKey;
use crate::{Error, Kind, Record, Result};

/// A key this process declared, tied to the runtime node that mirrors it.
pub(crate) trait Binding<C: Cells>: Send + Sync {
    fn kind(&self) -> Kind;

    fn key(&self) -> &str;

    /// Whether `handle`, a runtime handle erased by [`KeyRef`](crate::KeyRef),
    /// is the node this binding mirrors.
    fn is_handle(&self, handle: &dyn Any) -> bool;

    /// Applies `record`, one of this key's records in position order, to the
    /// node. `before` is the key's logged state without the record.
    fn apply(&self, rt: &Runtime<C>, record: &Record, before: Option<&LoggedKey>) -> Result<()>;

    /// Sets the node to exactly what `logged` holds, or to the binding's
    /// default when the log no longer knows the key. A collection binding
    /// holds its elements itself, so it takes them out of `logged`, but only
    /// once they all decoded; on an error nothing has changed.
    fn replace(&self, rt: &Runtime<C>, logged: Option<&mut LoggedKey>) -> Result<()>;

    /// The records a snapshot keeps for this key, given its logged state.
    /// Empty when the key holds only its logged default, compared by
    /// decoded value, which is how compaction forgets it.
    fn fold(&self, rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<Vec<Record>>;

    fn as_any(&self) -> &dyn Any;
}

pub(crate) struct InputBinding<T: 'static> {
    pub key: String,
    pub handle: Incr<T>,
    /// The logged default as of binding, used when the log no longer
    /// declares the key.
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

impl<C: Cells, T: Value + DeserializeOwned> Binding<C> for InputBinding<T> {
    fn kind(&self) -> Kind {
        Kind::Input
    }

    fn key(&self) -> &str {
        &self.key
    }

    fn is_handle(&self, handle: &dyn Any) -> bool {
        handle.downcast_ref::<Incr<T>>() == Some(&self.handle)
    }

    fn apply(&self, rt: &Runtime<C>, record: &Record, before: Option<&LoggedKey>) -> Result<()> {
        let value = match record {
            Record::Set { value, .. } => decode(&self.key, value)?,
            Record::Delete { .. } => {
                let declared = before.and_then(|logged| logged.declared.as_ref());
                match declared {
                    Some(declared) => decode(&self.key, &declared.default)?,
                    None => self.default.clone(),
                }
            }
            Record::Declare { default, .. } if before.and_then(LoggedKey::current).is_none() => {
                decode(&self.key, default)?
            }
            _ => return Ok(()),
        };
        rt.set(self.handle, value);
        Ok(())
    }

    fn replace(&self, rt: &Runtime<C>, logged: Option<&mut LoggedKey>) -> Result<()> {
        rt.set(self.handle, self.value_of(logged.as_deref())?);
        Ok(())
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
