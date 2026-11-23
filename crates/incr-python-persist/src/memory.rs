//! `MemoryBackend` and `MemoryMemo`: reference implementations of the
//! backend protocols over incr-persist's in-memory backends, for tests and
//! development.

/// Defines the `PyMemoryBackend`, `PyMemoryMemo` and `PyMemoryMemoGuard`
/// classes in the calling module, under the Python module path `$module`,
/// raising the exceptions `$exc`.
#[macro_export]
macro_rules! memory_classes {
    ($exc:ty, $module:literal) => {
        use std::collections::HashSet;

        use $crate::incr_persist::memo::{
            decode_delta_header, decode_header, encode_delta_header, encode_header, DeltaHeader,
            EncodedEntry, GenId, Generation, IndexItem, MemoBackend, MemoryMemo, Tombstone,
        };
        use $crate::incr_persist::{Backend, MemoryBackend, Position, ReadOutcome, Record, Result};
        use $crate::memo_backend::{item_tuple, ItemTuple};
        use $crate::pyo3::prelude::*;
        use $crate::pyo3::types::{PyBytes, PyList};

        fn persist_error(py: Python<'_>, error: $crate::incr_persist::Error) -> PyErr {
            $crate::persist_error::<$exc>(py, error)
        }

        /// The bytes of a `bytes` object in one copy; PyO3's `Vec<u8>`
        /// extraction reads a `bytes` an item at a time.
        fn bytes_of(object: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
            match object.downcast::<PyBytes>() {
                Ok(bytes) => Ok(bytes.as_bytes().to_vec()),
                Err(_) => object.extract(),
            }
        }

        fn item_list(py: Python<'_>, items: &[IndexItem]) -> PyResult<Py<PyList>> {
            let yielder = $crate::incr_core::Yielder::ambient();
            Ok(PyList::new(py, items.iter().inspect(|_| yielder.tick()).map(item_tuple))?.unbind())
        }

        /// A log held in process memory: the reference implementation of the
        /// backend protocol, for tests and development. Stores opened on the same
        /// object share its log; `read_only()` gives a handle that cannot write.
        #[pyclass(name = "MemoryBackend", module = $module)]
        pub(crate) struct PyMemoryBackend {
            inner: MemoryBackend,
        }

        #[pymethods]
        impl PyMemoryBackend {
            #[new]
            fn new() -> Self {
                PyMemoryBackend {
                    inner: MemoryBackend::new(),
                }
            }

            fn read_only(&self) -> Self {
                PyMemoryBackend {
                    inner: self.inner.read_only(),
                }
            }

            fn append(&self, py: Python<'_>, records: Vec<Bound<'_, PyAny>>) -> PyResult<u64> {
                let yielder = $crate::incr_core::Yielder::ambient();
                let records = records
                    .iter()
                    .inspect(|_| yielder.tick())
                    .map(bytes_of)
                    .collect::<PyResult<Vec<_>>>()?;
                let records = records
                    .iter()
                    .inspect(|_| yielder.tick())
                    .map(|bytes| Record::decode(bytes))
                    .collect::<Result<Vec<_>>>()
                    .map_err(|e| persist_error(py, e))?;
                self.inner
                    .append(&records)
                    .map(|at| at.0)
                    .map_err(|e| persist_error(py, e))
            }

            /// Appends `records` right after `after` if the log's tail is
            /// still `after`, returning the last position; raises
            /// `TailMoved`, writing nothing, when the tail is elsewhere.
            fn append_at(
                &self,
                py: Python<'_>,
                after: u64,
                records: Vec<Bound<'_, PyAny>>,
            ) -> PyResult<u64> {
                let yielder = $crate::incr_core::Yielder::ambient();
                let records = records
                    .iter()
                    .inspect(|_| yielder.tick())
                    .map(bytes_of)
                    .collect::<PyResult<Vec<_>>>()?;
                let records = records
                    .iter()
                    .inspect(|_| yielder.tick())
                    .map(|bytes| Record::decode(bytes))
                    .collect::<Result<Vec<_>>>()
                    .map_err(|e| persist_error(py, e))?;
                self.inner
                    .conditional_append()
                    .expect("the memory backend has a conditional append")
                    .append_at(Position(after), &records)
                    .map(|at| at.0)
                    .map_err(|e| persist_error(py, e))
            }

            fn sync(&self, py: Python<'_>, upto: u64) -> PyResult<()> {
                self.inner
                    .sync(Position(upto))
                    .map_err(|e| persist_error(py, e))
            }

            fn read_from(&self, py: Python<'_>, after: u64) -> PyResult<Py<PyAny>> {
                let mut found = Vec::new();
                let outcome = self
                    .inner
                    .read_from(Position(after), &mut found)
                    .map_err(|e| persist_error(py, e))?;
                if let ReadOutcome::Compacted { snapshot_at } = outcome {
                    return Ok(("compacted", snapshot_at.0)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind());
                }
                let yielder = $crate::incr_core::Yielder::ambient();
                let items = found
                    .iter()
                    .inspect(|_| yielder.tick())
                    .map(|(at, record)| {
                        let bytes = record.encode().map_err(|e| persist_error(py, e))?;
                        (at.0, PyBytes::new(py, &bytes)).into_pyobject(py)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyList::new(py, items)?.into_any().unbind())
            }

            fn load_snapshot(&self, py: Python<'_>) -> PyResult<Option<(u64, Vec<Py<PyBytes>>)>> {
                let Some((at, records)) = self
                    .inner
                    .load_snapshot()
                    .map_err(|e| persist_error(py, e))?
                else {
                    return Ok(None);
                };
                let yielder = $crate::incr_core::Yielder::ambient();
                let records = records
                    .iter()
                    .inspect(|_| yielder.tick())
                    .map(|record| {
                        let bytes = record.encode().map_err(|e| persist_error(py, e))?;
                        Ok(PyBytes::new(py, &bytes).unbind())
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(Some((at.0, records)))
            }

            fn snapshot_id(&self, py: Python<'_>) -> PyResult<Option<(u64, u64)>> {
                Ok(self
                    .inner
                    .snapshot_id()
                    .map_err(|e| persist_error(py, e))?
                    .map(|id| (id.position.0, id.generation)))
            }

            fn compact(
                &self,
                py: Python<'_>,
                upto: u64,
                folded: Vec<Bound<'_, PyAny>>,
            ) -> PyResult<()> {
                let yielder = $crate::incr_core::Yielder::ambient();
                let folded = folded
                    .iter()
                    .inspect(|_| yielder.tick())
                    .map(bytes_of)
                    .collect::<PyResult<Vec<_>>>()?;
                let folded = folded
                    .iter()
                    .inspect(|_| yielder.tick())
                    .map(|bytes| Record::decode(bytes))
                    .collect::<Result<Vec<_>>>()
                    .map_err(|e| persist_error(py, e))?;
                self.inner
                    .compact(Position(upto), folded)
                    .map_err(|e| persist_error(py, e))
            }

            fn compact_streamed(
                &self,
                py: Python<'_>,
                upto: u64,
                records: Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let mut pulls = records.try_iter()?.map(|record| {
                    let bytes = record
                        .and_then(|record| bytes_of(&record))
                        .map_err($crate::backend::raised::<$exc>)?;
                    Record::decode(&bytes)
                });
                self.inner
                    .compact_streamed(Position(upto), &mut pulls)
                    .map_err(|e| persist_error(py, e))
            }

            fn writable(&self) -> bool {
                self.inner.writable()
            }
        }

        fn layer_tuple(
            py: Python<'_>,
            layer: &Generation,
        ) -> PyResult<(u64, Py<PyBytes>, Py<PyList>)> {
            let header = match (layer.base(), layer.parent()) {
                (Some(base), Some(parent)) => encode_delta_header(&DeltaHeader {
                    header: layer.header().clone(),
                    base,
                    parent,
                }),
                _ => encode_header(layer.header()),
            }
            .map_err(|e| persist_error(py, e))?;
            Ok((
                layer.id().0,
                PyBytes::new(py, &header).unbind(),
                item_list(py, layer.items())?,
            ))
        }

        /// A memo store held in process memory: the reference implementation of
        /// the memo backend protocol, for tests and development. Stores opened on
        /// the same object share its generations.
        #[pyclass(name = "MemoryMemo", module = $module)]
        pub(crate) struct PyMemoryMemo {
            inner: MemoryMemo,
        }

        #[pyclass(name = "MemoryMemoGuard", module = $module)]
        pub(crate) struct PyMemoryMemoGuard {
            guard: Option<<MemoryMemo as MemoBackend>::Guard>,
        }

        #[pymethods]
        impl PyMemoryMemoGuard {
            fn release(&mut self) {
                self.guard.take();
            }
        }

        #[pymethods]
        impl PyMemoryMemo {
            #[new]
            fn new() -> Self {
                PyMemoryMemo {
                    inner: MemoryMemo::new(),
                }
            }

            fn lock_checkpoint(&self, py: Python<'_>) -> PyResult<PyMemoryMemoGuard> {
                let guard = self
                    .inner
                    .lock_checkpoint()
                    .map_err(|e| persist_error(py, e))?;
                Ok(PyMemoryMemoGuard { guard: Some(guard) })
            }

            #[allow(clippy::type_complexity)]
            fn load(&self, py: Python<'_>) -> PyResult<Option<(u64, Py<PyBytes>, Py<PyList>)>> {
                let Some(generation) = self.inner.load().map_err(|e| persist_error(py, e))? else {
                    return Ok(None);
                };
                let header =
                    encode_header(generation.header()).map_err(|e| persist_error(py, e))?;
                Ok(Some((
                    generation.id().0,
                    PyBytes::new(py, &header).unbind(),
                    item_list(py, generation.items())?,
                )))
            }

            fn read_entry(
                &self,
                py: Python<'_>,
                generation: u64,
                item: ItemTuple,
            ) -> PyResult<Option<Py<PyBytes>>> {
                let item = IndexItem {
                    table: item.0,
                    key_hash: item.1,
                    loc: item.2,
                    len: item.3,
                };
                let frame = self
                    .inner
                    .read_entry(GenId(generation), &item)
                    .map_err(|e| persist_error(py, e))?;
                Ok(frame.map(|frame| PyBytes::new(py, &frame).unbind()))
            }

            fn write_generation(
                &self,
                py: Python<'_>,
                base: Option<u64>,
                generation: u64,
                header: Vec<u8>,
                entries: Vec<(u32, u64, Bound<'_, PyAny>)>,
            ) -> PyResult<Py<PyList>> {
                let header = decode_header(&header).map_err(|e| persist_error(py, e))?;
                let entries = entries
                    .into_iter()
                    .map(|(table, key_hash, frame)| {
                        Ok(EncodedEntry {
                            table,
                            key_hash,
                            frame: bytes_of(&frame)?,
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let written = self
                    .inner
                    .write_generation(base.map(GenId), GenId(generation), &header, &entries)
                    .map_err(|e| persist_error(py, e))?;
                item_list(py, written.items())
            }

            fn read_entries(
                &self,
                py: Python<'_>,
                generation: u64,
                items: Vec<ItemTuple>,
            ) -> PyResult<Vec<Option<Py<PyBytes>>>> {
                let items = items
                    .into_iter()
                    .map(|(table, key_hash, loc, len)| IndexItem {
                        table,
                        key_hash,
                        loc,
                        len,
                    })
                    .collect::<Vec<_>>();
                let frames = self
                    .inner
                    .read_entries(GenId(generation), &items)
                    .map_err(|e| persist_error(py, e))?;
                let yielder = $crate::incr_core::Yielder::ambient();
                Ok(frames
                    .into_iter()
                    .inspect(|_| yielder.tick())
                    .map(|frame| frame.map(|frame| PyBytes::new(py, &frame).unbind()))
                    .collect())
            }

            #[allow(clippy::type_complexity)]
            #[pyo3(signature = (known=None))]
            fn load_chain(
                &self,
                py: Python<'_>,
                known: Option<HashSet<u64>>,
            ) -> PyResult<Option<Vec<(u64, Option<Py<PyBytes>>, Option<Py<PyList>>)>>> {
                let known = known.unwrap_or_default();
                let Some(chain) = self
                    .inner
                    .load_chain(&[])
                    .map_err(|e| persist_error(py, e))?
                else {
                    return Ok(None);
                };
                let layers = chain
                    .layers()
                    .iter()
                    .map(|layer| {
                        if known.contains(&layer.id().0) {
                            return Ok((layer.id().0, None, None));
                        }
                        let (id, header, items) = layer_tuple(py, layer)?;
                        Ok((id, Some(header), Some(items)))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(Some(layers))
            }

            fn write_delta(
                &self,
                py: Python<'_>,
                tip: u64,
                generation: u64,
                header: Vec<u8>,
                entries: Vec<(u32, u64, Bound<'_, PyAny>)>,
                tombstones: Vec<(u32, u64)>,
            ) -> PyResult<Py<PyList>> {
                let header = decode_delta_header(&header).map_err(|e| persist_error(py, e))?;
                let entries = entries
                    .into_iter()
                    .map(|(table, key_hash, frame)| {
                        Ok(EncodedEntry {
                            table,
                            key_hash,
                            frame: bytes_of(&frame)?,
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let tombstones = tombstones
                    .into_iter()
                    .map(|(table, key_hash)| Tombstone { table, key_hash })
                    .collect::<Vec<_>>();
                let written = self
                    .inner
                    .write_delta(
                        GenId(tip),
                        GenId(generation),
                        &header,
                        &entries,
                        &tombstones,
                    )
                    .map_err(|e| persist_error(py, e))?;
                item_list(py, written.items())
            }

            fn write_generation_streamed(
                &self,
                py: Python<'_>,
                base: Option<u64>,
                generation: u64,
                header: Vec<u8>,
                frames: Bound<'_, PyAny>,
            ) -> PyResult<Py<PyList>> {
                let header = decode_header(&header).map_err(|e| persist_error(py, e))?;
                let mut pulls = frames.try_iter()?.map(|frame| {
                    let (table, key_hash, frame) = frame
                        .and_then(|frame| {
                            let (table, key_hash, frame) =
                                frame.extract::<(u32, u64, Bound<'_, PyAny>)>()?;
                            Ok((table, key_hash, bytes_of(&frame)?))
                        })
                        .map_err($crate::backend::raised::<$exc>)?;
                    Ok(EncodedEntry {
                        table,
                        key_hash,
                        frame,
                    })
                });
                let written = self
                    .inner
                    .write_generation_streamed(
                        base.map(GenId),
                        GenId(generation),
                        &header,
                        &mut pulls,
                    )
                    .map_err(|e| persist_error(py, e))?;
                item_list(py, written.items())
            }

            fn remove_generation(&self, py: Python<'_>, generation: u64) -> PyResult<()> {
                self.inner
                    .remove_generation(GenId(generation))
                    .map_err(|e| persist_error(py, e))
            }
        }
    };
}
