use std::any::Any;

use incr_core::{Cells, Incr, Runtime, Value};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::logged::LoggedKey;
use crate::{Error, Record, Result};

/// A key this process declared, tied to the runtime node that mirrors it.
pub(crate) trait Binding<C: Cells>: Send + Sync {
    /// Applies `record`, one of this key's records in position order, to the
    /// node. `before` is the key's logged state without the record.
    fn apply(&self, rt: &Runtime<C>, record: &Record, before: Option<&LoggedKey>) -> Result<()>;

    /// Sets the node to exactly what `logged` holds, or to the binding's
    /// default when the log no longer knows the key.
    fn replace(&self, rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<()>;

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

    fn replace(&self, rt: &Runtime<C>, logged: Option<&LoggedKey>) -> Result<()> {
        rt.set(self.handle, self.value_of(logged)?);
        Ok(())
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
