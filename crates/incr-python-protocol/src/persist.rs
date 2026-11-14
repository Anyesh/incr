//! `Serialize` and `Deserialize` for [`PyValue`] and [`PyKey`], behind the `persist`
//! feature. A value crosses serde as one byte string that a process-wide
//! [`Hook`] produces and reads back, so this crate needs no persistence
//! dependency and the binding decides what the bytes are.

use std::fmt;
use std::sync::OnceLock;

use pyo3::Python;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{PyKey, PyValue};

/// How a binding turns values into bytes and back. Both run with the GIL
/// held, on the thread serde runs on, and report a failure as the message
/// serde carries.
pub struct Hook {
    pub encode: fn(Python<'_>, &PyValue) -> Result<Vec<u8>, String>,
    pub decode: fn(Python<'_>, &[u8]) -> Result<PyValue, String>,
    /// A keyed collection's key alone, as its pairs encode their key half.
    pub encode_key: fn(Python<'_>, &PyKey) -> Result<Vec<u8>, String>,
    pub decode_key: fn(Python<'_>, &[u8]) -> Result<PyKey, String>,
}

static HOOK: OnceLock<Hook> = OnceLock::new();

/// Installs the process-wide hook; returns it back when one is installed
/// already.
pub fn install(hook: Hook) -> Result<(), Hook> {
    HOOK.set(hook)
}

fn hook() -> Result<&'static Hook, &'static str> {
    HOOK.get().ok_or("no value codec hook is installed")
}

impl Serialize for PyValue {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let hook = hook().map_err(serde::ser::Error::custom)?;
        let bytes =
            Python::attach(|py| (hook.encode)(py, self)).map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&bytes)
    }
}

struct Bytes;

impl<'de> Visitor<'de> for Bytes {
    type Value = PyValue;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the bytes of an encoded Python value")
    }

    fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<PyValue, E> {
        let hook = hook().map_err(E::custom)?;
        Python::attach(|py| (hook.decode)(py, bytes)).map_err(E::custom)
    }
}

impl<'de> Deserialize<'de> for PyValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_bytes(Bytes)
    }
}

impl Serialize for PyKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let hook = hook().map_err(serde::ser::Error::custom)?;
        let bytes =
            Python::attach(|py| (hook.encode_key)(py, self)).map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&bytes)
    }
}

struct KeyBytes;

impl<'de> Visitor<'de> for KeyBytes {
    type Value = PyKey;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the bytes of an encoded Python key")
    }

    fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<PyKey, E> {
        let hook = hook().map_err(E::custom)?;
        Python::attach(|py| (hook.decode_key)(py, bytes)).map_err(E::custom)
    }
}

impl<'de> Deserialize<'de> for PyKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_bytes(KeyBytes)
    }
}
