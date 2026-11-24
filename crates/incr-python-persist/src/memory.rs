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
        use $crate::incr_python_protocol::gil;
        use $crate::memo_backend::{item_tuple, ItemTuple};
        use $crate::pyo3::prelude::*;
        use $crate::pyo3::types::PyBytes;

        static APPEND_RECORDS: gil::Site = gil::Site::caller("mem.MemoryBackend.append.records");
        static APPEND_AT_RECORDS: gil::Site =
            gil::Site::caller("mem.MemoryBackend.append_at.records");
        static READ_FROM_RECORDS: gil::Site =
            gil::Site::caller("mem.MemoryBackend.read_from.records");
        static LOAD_SNAPSHOT_RECORDS: gil::Site =
            gil::Site::caller("mem.MemoryBackend.load_snapshot.records");
        static COMPACT_RECORDS: gil::Site = gil::Site::caller("mem.MemoryBackend.compact.records");
        static COMPACT_STREAMED_RECORDS: gil::Site =
            gil::Site::unbounded("mem.MemoryBackend.compact_streamed.records");
        static LOAD_ITEMS: gil::Site = gil::Site::unbounded("mem.MemoryMemo.load.items");
        static LOAD_CHAIN_KNOWN: gil::Site = gil::Site::one("mem.MemoryMemo.load_chain.known");
        static LOAD_CHAIN_LAYERS: gil::Site = gil::Site::one("mem.MemoryMemo.load_chain.layers");
        static LOAD_CHAIN_ITEMS: gil::Site =
            gil::Site::unbounded("mem.MemoryMemo.load_chain.items");
        static READ_ENTRIES_ITEMS: gil::Site =
            gil::Site::bounded("mem.MemoryMemo.read_entries.items", 256);
        static READ_ENTRIES_FRAMES: gil::Site =
            gil::Site::bounded("mem.MemoryMemo.read_entries.frames", 256);
        static WRITE_GENERATION_ENTRIES: gil::Site =
            gil::Site::caller("mem.MemoryMemo.write_generation.entries");
        static WRITE_GENERATION_ITEMS: gil::Site =
            gil::Site::caller("mem.MemoryMemo.write_generation.items");
        static WRITE_GENERATION_STREAMED_FRAMES: gil::Site =
            gil::Site::unbounded("mem.MemoryMemo.write_generation_streamed.frames");
        static WRITE_GENERATION_STREAMED_ITEMS: gil::Site =
            gil::Site::unbounded("mem.MemoryMemo.write_generation_streamed.items");
        static WRITE_DELTA_ENTRIES: gil::Site =
            gil::Site::unbounded("mem.MemoryMemo.write_delta.entries");
        static WRITE_DELTA_TOMBSTONES: gil::Site =
            gil::Site::unbounded("mem.MemoryMemo.write_delta.tombstones");
        static WRITE_DELTA_ITEMS: gil::Site =
            gil::Site::unbounded("mem.MemoryMemo.write_delta.items");

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

        fn record_of(record: &Bound<'_, PyAny>) -> Result<Record> {
            let bytes = bytes_of(record).map_err($crate::backend::raised::<$exc>)?;
            Record::decode(&bytes)
        }

        fn entry_of(entry: &Bound<'_, PyAny>) -> Result<EncodedEntry> {
            let (table, key_hash, frame) = entry
                .extract::<(u32, u64, Bound<'_, PyAny>)>()
                .map_err($crate::backend::raised::<$exc>)?;
            Ok(EncodedEntry {
                table,
                key_hash,
                frame: bytes_of(&frame).map_err($crate::backend::raised::<$exc>)?,
            })
        }

        fn tombstone_of(tombstone: &Bound<'_, PyAny>) -> Result<Tombstone> {
            let (table, key_hash) = tombstone
                .extract::<(u32, u64)>()
                .map_err($crate::backend::raised::<$exc>)?;
            Ok(Tombstone { table, key_hash })
        }

        fn index_item_of(item: &Bound<'_, PyAny>) -> PyResult<IndexItem> {
            let (table, key_hash, loc, len) = item.extract::<ItemTuple>()?;
            Ok(IndexItem {
                table,
                key_hash,
                loc,
                len,
            })
        }

        fn pull_records(
            site: &'static gil::Site,
            records: &Bound<'_, PyAny>,
        ) -> Result<Vec<Record>> {
            gil::pull(site, records, $crate::backend::raised::<$exc>, record_of)
        }

        fn pull_entries(
            site: &'static gil::Site,
            entries: &Bound<'_, PyAny>,
        ) -> Result<Vec<EncodedEntry>> {
            gil::pull(site, entries, $crate::backend::raised::<$exc>, entry_of)
        }

        fn header_bytes(header: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
            bytes_of(header)
        }

        fn item_list<'py>(
            py: Python<'py>,
            site: &'static gil::Site,
            items: &[IndexItem],
        ) -> PyResult<Bound<'py, PyAny>> {
            gil::build(
                py,
                site,
                items,
                |error| error,
                |py, item| Ok(item_tuple(item).into_pyobject(py)?.into_any()),
            )
        }

        fn encoded_records<'py>(
            py: Python<'py>,
            site: &'static gil::Site,
            records: &[Record],
        ) -> PyResult<Bound<'py, PyAny>> {
            gil::build(
                py,
                site,
                records,
                |error| error,
                |py, record| {
                    let bytes = record.encode().map_err(|e| persist_error(py, e))?;
                    Ok(PyBytes::new(py, &bytes).into_any())
                },
            )
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

            #[classattr]
            fn concurrent_compaction() -> bool {
                true
            }

            fn read_only(&self) -> Self {
                PyMemoryBackend {
                    inner: self.inner.read_only(),
                }
            }

            fn append(&self, py: Python<'_>, records: Bound<'_, PyAny>) -> PyResult<u64> {
                let records =
                    pull_records(&APPEND_RECORDS, &records).map_err(|e| persist_error(py, e))?;
                gil::leave(py, || self.inner.append(&records))
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
                records: Bound<'_, PyAny>,
            ) -> PyResult<u64> {
                let records =
                    pull_records(&APPEND_AT_RECORDS, &records).map_err(|e| persist_error(py, e))?;
                gil::leave(py, || {
                    self.inner
                        .conditional_append()
                        .expect("the memory backend has a conditional append")
                        .append_at(Position(after), &records)
                })
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
                let outcome = gil::leave(py, || self.inner.read_from(Position(after), &mut found))
                    .map_err(|e| persist_error(py, e))?;
                if let ReadOutcome::Compacted { snapshot_at } = outcome {
                    return Ok(("compacted", snapshot_at.0)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind());
                }
                Ok(gil::build(
                    py,
                    &READ_FROM_RECORDS,
                    &found,
                    |error| error,
                    |py, (at, record)| {
                        let bytes = record.encode().map_err(|e| persist_error(py, e))?;
                        Ok((at.0, PyBytes::new(py, &bytes))
                            .into_pyobject(py)?
                            .into_any())
                    },
                )?
                .unbind())
            }

            fn load_snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let Some((at, records)) = gil::leave(py, || self.inner.load_snapshot())
                    .map_err(|e| persist_error(py, e))?
                else {
                    return Ok(py.None());
                };
                let records = encoded_records(py, &LOAD_SNAPSHOT_RECORDS, &records)?;
                Ok((at.0, records).into_pyobject(py)?.into_any().unbind())
            }

            fn snapshot_id(&self, py: Python<'_>) -> PyResult<Option<(u64, u64)>> {
                Ok(self
                    .inner
                    .snapshot_id()
                    .map_err(|e| persist_error(py, e))?
                    .map(|id| (id.position.0, id.generation)))
            }

            fn compact(&self, py: Python<'_>, upto: u64, folded: Bound<'_, PyAny>) -> PyResult<()> {
                let folded =
                    pull_records(&COMPACT_RECORDS, &folded).map_err(|e| persist_error(py, e))?;
                gil::leave(py, || self.inner.compact(Position(upto), folded))
                    .map_err(|e| persist_error(py, e))
            }

            fn compact_streamed(
                &self,
                py: Python<'_>,
                upto: u64,
                records: Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let records = pull_records(&COMPACT_STREAMED_RECORDS, &records)
                    .map_err(|e| persist_error(py, e))?;
                gil::leave(py, || {
                    self.inner
                        .compact_streamed(Position(upto), &mut records.into_iter().map(Ok))
                })
                .map_err(|e| persist_error(py, e))
            }

            fn writable(&self) -> bool {
                self.inner.writable()
            }
        }

        fn layer_tuple<'py>(
            py: Python<'py>,
            layer: &Generation,
        ) -> PyResult<(u64, Bound<'py, PyBytes>, Bound<'py, PyAny>)> {
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
                PyBytes::new(py, &header),
                item_list(py, &LOAD_CHAIN_ITEMS, layer.items())?,
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

            #[classattr]
            fn concurrent_compaction() -> bool {
                true
            }

            fn lock_checkpoint(&self, py: Python<'_>) -> PyResult<PyMemoryMemoGuard> {
                let guard = gil::leave(py, || self.inner.lock_checkpoint())
                    .map_err(|e| persist_error(py, e))?;
                Ok(PyMemoryMemoGuard { guard: Some(guard) })
            }

            fn load(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let Some(generation) =
                    gil::leave(py, || self.inner.load()).map_err(|e| persist_error(py, e))?
                else {
                    return Ok(py.None());
                };
                let header =
                    encode_header(generation.header()).map_err(|e| persist_error(py, e))?;
                let items = item_list(py, &LOAD_ITEMS, generation.items())?;
                let id = generation.id().0;
                gil::leave(py, move || drop(generation));
                Ok((id, PyBytes::new(py, &header), items)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind())
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
                let frame = gil::leave(py, || self.inner.read_entry(GenId(generation), &item))
                    .map_err(|e| persist_error(py, e))?;
                Ok(frame.map(|frame| PyBytes::new(py, &frame).unbind()))
            }

            fn write_generation(
                &self,
                py: Python<'_>,
                base: Option<u64>,
                generation: u64,
                header: Bound<'_, PyAny>,
                entries: Bound<'_, PyAny>,
            ) -> PyResult<Py<PyAny>> {
                let header =
                    decode_header(&header_bytes(&header)?).map_err(|e| persist_error(py, e))?;
                let entries = pull_entries(&WRITE_GENERATION_ENTRIES, &entries)
                    .map_err(|e| persist_error(py, e))?;
                let written = gil::leave(py, || {
                    self.inner.write_generation(
                        base.map(GenId),
                        GenId(generation),
                        &header,
                        &entries,
                    )
                })
                .map_err(|e| persist_error(py, e))?;
                let items = item_list(py, &WRITE_GENERATION_ITEMS, written.items())?;
                gil::leave(py, move || drop(written));
                Ok(items.unbind())
            }

            fn read_entries(
                &self,
                py: Python<'_>,
                generation: u64,
                items: Bound<'_, PyAny>,
            ) -> PyResult<Py<PyAny>> {
                let items = gil::pull(&READ_ENTRIES_ITEMS, &items, |error| error, index_item_of)?;
                let frames = gil::leave(py, || self.inner.read_entries(GenId(generation), &items))
                    .map_err(|e| persist_error(py, e))?;
                Ok(gil::build(
                    py,
                    &READ_ENTRIES_FRAMES,
                    frames,
                    |error| error,
                    |py, frame| {
                        Ok(match frame {
                            Some(frame) => PyBytes::new(py, &frame).into_any(),
                            None => py.None().into_bound(py),
                        })
                    },
                )?
                .unbind())
            }

            #[pyo3(signature = (known=None))]
            fn load_chain(
                &self,
                py: Python<'_>,
                known: Option<Bound<'_, PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                let known = match known {
                    Some(known) if !known.is_none() => gil::pull(
                        &LOAD_CHAIN_KNOWN,
                        &known,
                        |error| error,
                        |id| id.extract::<u64>(),
                    )?
                    .into_iter()
                    .collect::<HashSet<u64>>(),
                    _ => HashSet::new(),
                };
                let Some(chain) = gil::leave(py, || self.inner.load_chain(&[]))
                    .map_err(|e| persist_error(py, e))?
                else {
                    return Ok(py.None());
                };
                let layers = gil::build(
                    py,
                    &LOAD_CHAIN_LAYERS,
                    chain.layers(),
                    |error| error,
                    |py, layer| {
                        if known.contains(&layer.id().0) {
                            return Ok((layer.id().0, py.None(), py.None())
                                .into_pyobject(py)?
                                .into_any());
                        }
                        let (id, header, items) = layer_tuple(py, layer)?;
                        Ok((id, header, items).into_pyobject(py)?.into_any())
                    },
                )?;
                gil::leave(py, move || drop(chain));
                Ok(layers.unbind())
            }

            fn write_delta(
                &self,
                py: Python<'_>,
                tip: u64,
                generation: u64,
                header: Bound<'_, PyAny>,
                entries: Bound<'_, PyAny>,
                tombstones: Bound<'_, PyAny>,
            ) -> PyResult<Py<PyAny>> {
                let header = decode_delta_header(&header_bytes(&header)?)
                    .map_err(|e| persist_error(py, e))?;
                let entries = pull_entries(&WRITE_DELTA_ENTRIES, &entries)
                    .map_err(|e| persist_error(py, e))?;
                let tombstones = gil::pull(
                    &WRITE_DELTA_TOMBSTONES,
                    &tombstones,
                    $crate::backend::raised::<$exc>,
                    tombstone_of,
                )
                .map_err(|e| persist_error(py, e))?;
                let written = gil::leave(py, || {
                    self.inner.write_delta(
                        GenId(tip),
                        GenId(generation),
                        &header,
                        &entries,
                        &tombstones,
                    )
                })
                .map_err(|e| persist_error(py, e))?;
                let items = item_list(py, &WRITE_DELTA_ITEMS, written.items())?;
                gil::leave(py, move || drop((written, entries, tombstones)));
                Ok(items.unbind())
            }

            fn write_generation_streamed(
                &self,
                py: Python<'_>,
                base: Option<u64>,
                generation: u64,
                header: Bound<'_, PyAny>,
                frames: Bound<'_, PyAny>,
            ) -> PyResult<Py<PyAny>> {
                let header =
                    decode_header(&header_bytes(&header)?).map_err(|e| persist_error(py, e))?;
                let entries = pull_entries(&WRITE_GENERATION_STREAMED_FRAMES, &frames)
                    .map_err(|e| persist_error(py, e))?;
                let written = gil::leave(py, || {
                    self.inner.write_generation_streamed(
                        base.map(GenId),
                        GenId(generation),
                        &header,
                        &mut entries.into_iter().map(Ok),
                    )
                })
                .map_err(|e| persist_error(py, e))?;
                let items = item_list(py, &WRITE_GENERATION_STREAMED_ITEMS, written.items())?;
                gil::leave(py, move || drop(written));
                Ok(items.unbind())
            }

            fn remove_generation(&self, py: Python<'_>, generation: u64) -> PyResult<()> {
                gil::leave(py, || self.inner.remove_generation(GenId(generation)))
                    .map_err(|e| persist_error(py, e))
            }
        }
    };
}
