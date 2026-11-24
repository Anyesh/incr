use std::iter::Peekable;
use std::vec;

use incr_core::Yielder;
use serde::Serialize;

use super::binding::encode;
use super::dep_index::fp_of;
use super::keyed::UNLOGGED_RANK;
use super::logged::LoggedKey;
use crate::{Record, Result};

/// A keyed collection's pairs and their ranks, each named by a `Row`, as
/// a snapshot fold reads them.
pub(crate) trait PairRows {
    type Pair: Serialize;
    type Row: Copy;

    fn key(&self) -> &str;

    /// Runs `f` on the pair at `row` and its rank. Fails when the row can
    /// no longer be read, as a live handle can after a write around the
    /// store.
    fn with_pair<R>(&self, row: Self::Row, f: impl FnOnce(&Self::Pair, u128) -> R) -> Result<R>;
}

/// A plain collection's distinct elements and their multiplicities, each
/// named by a `Row`, as a snapshot fold reads them.
pub(crate) trait ElementRows {
    type Element: Serialize;
    type Row: Copy;

    fn key(&self) -> &str;

    /// Runs `f` on the element at `row` and its multiplicity. Fails when
    /// the row can no longer be read, as a live slot can after a write
    /// around the store.
    fn with_element<R>(
        &self,
        row: Self::Row,
        f: impl FnOnce(&Self::Element, usize) -> R,
    ) -> Result<R>;
}

/// Each distinct element's row with its [`element_fingerprint`].
pub(crate) type Fingerprints<R> = Vec<(u64, R)>;

/// A keyed key's snapshot records, made as they are pulled: `logged`'s
/// records, then one `Insert` per pair. `rows` lists every pair in rank
/// order, so those at [`UNLOGGED_RANK`] come first; they are ranked by
/// the fingerprint of their encoding and merged in, a logged pair going
/// first on a tie. Empty when there are no pairs and no logged value.
pub(crate) fn fold_pairs<S: PairRows>(
    source: S,
    rows: Vec<S::Row>,
    logged: Option<&LoggedKey>,
    yielder: &Yielder<'_>,
) -> Result<impl Iterator<Item = Result<Record>>> {
    let has_value = logged.is_some_and(|logged| logged.value.is_some());
    let logged = logged.filter(|_| !rows.is_empty() || has_value);
    let mut logged_rows = rows.into_iter().peekable();
    let mut unlogged = Vec::new();
    while let Some(&row) = logged_rows.peek() {
        yielder.tick();
        let element = source.with_pair(row, |pair, rank| {
            (rank == UNLOGGED_RANK).then(|| encode(source.key(), pair))
        })?;
        match element {
            Some(element) => unlogged.push((fp_of(&element?), row)),
            None => break,
        }
        logged_rows.next();
    }
    unlogged.sort_unstable_by_key(|&(fp, _)| fp);
    let head = logged.map_or_else(Vec::new, |logged| logged.records(source.key()));
    Ok(head.into_iter().map(Ok).chain(RankMerge {
        source,
        logged: logged_rows,
        logged_rank: None,
        unlogged: unlogged.into_iter().peekable(),
        failed: false,
    }))
}

/// A plain collection key's snapshot records, made as they are pulled:
/// `logged`'s records, then one `Insert` per element occurrence, the
/// distinct elements ordered by the high 64 bits of their encoding's
/// fingerprint and ties by the encodings themselves, so every process
/// lists the same elements in the same order. Empty when there
/// are no elements and no logged value.
pub(crate) fn fold_elements<S: ElementRows>(
    source: S,
    fingerprints: Fingerprints<S::Row>,
    logged: Option<&LoggedKey>,
    yielder: &Yielder<'_>,
) -> Result<impl Iterator<Item = Result<Record>>> {
    let rows = rank_elements(&source, fingerprints, yielder)?;
    let has_value = logged.is_some_and(|logged| logged.value.is_some());
    let logged = logged.filter(|_| !rows.is_empty() || has_value);
    let head = logged.map_or_else(Vec::new, |logged| logged.records(source.key()));
    let mut rows = rows.into_iter();
    let mut repeat: Option<(Vec<u8>, usize)> = None;
    let mut failed = false;
    let elements = std::iter::from_fn(move || {
        if failed {
            return None;
        }
        if let Some((element, left)) = &mut repeat {
            *left -= 1;
            let element = if *left == 0 {
                repeat.take().map(|(element, _)| element)?
            } else {
                element.clone()
            };
            return Some(Ok(insert(source.key(), element)));
        }
        let row = rows.next()?;
        match encode_element(&source, row) {
            Ok((element, count)) => {
                if count > 1 {
                    repeat = Some((element.clone(), count - 1));
                }
                Some(Ok(insert(source.key(), element)))
            }
            Err(e) => {
                failed = true;
                Some(Err(e))
            }
        }
    });
    Ok(head.into_iter().map(Ok).chain(elements))
}

/// The key [`fold_elements`] first orders an element by.
pub(crate) fn element_fingerprint<T: Serialize>(key: &str, element: &T) -> Result<u64> {
    Ok((fp_of(&encode(key, element)?) >> 64) as u64)
}

/// Orders the rows as [`fold_elements`] lists them. Keeps 16 bytes per
/// element rather than its encoding, which the fold makes again as it
/// yields.
fn rank_elements<S: ElementRows>(
    source: &S,
    mut rows: Vec<(u64, S::Row)>,
    yielder: &Yielder<'_>,
) -> Result<Vec<S::Row>> {
    rows.sort_unstable_by_key(|&(fp, _)| fp);
    let mut at = 0;
    while at < rows.len() {
        yielder.tick();
        let run = rows[at..]
            .iter()
            .take_while(|&&(fp, _)| fp == rows[at].0)
            .count();
        if run > 1 {
            let mut tied = rows[at..at + run]
                .iter()
                .map(|&(fp, row)| Ok((encode_element(source, row)?.0, fp, row)))
                .collect::<Result<Vec<_>>>()?;
            tied.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            for (into, (_, fp, row)) in rows[at..at + run].iter_mut().zip(tied) {
                *into = (fp, row);
            }
        }
        at += run;
    }
    Ok(rows.into_iter().map(|(_, row)| row).collect())
}

fn encode_element<S: ElementRows>(source: &S, row: S::Row) -> Result<(Vec<u8>, usize)> {
    source.with_element(row, |element, count| {
        encode(source.key(), element).map(|element| (element, count))
    })?
}

fn insert(key: &str, element: Vec<u8>) -> Record {
    Record::Insert {
        key: key.to_owned(),
        element,
    }
}

/// [`fold_pairs`]' `Insert`s, each pair encoded as it is pulled.
struct RankMerge<S: PairRows> {
    source: S,
    logged: Peekable<vec::IntoIter<S::Row>>,
    /// The rank of the pair `logged` peeks at, once read.
    logged_rank: Option<u128>,
    unlogged: Peekable<vec::IntoIter<(u128, S::Row)>>,
    failed: bool,
}

impl<S: PairRows> RankMerge<S> {
    fn next_row(&mut self) -> Result<Option<S::Row>> {
        let take_logged = match (self.unlogged.peek(), self.logged.peek()) {
            (_, None) => false,
            (None, Some(_)) => true,
            (Some(&(fp, _)), Some(&row)) => {
                let rank = match self.logged_rank {
                    Some(rank) => rank,
                    None => *self
                        .logged_rank
                        .insert(self.source.with_pair(row, |_, rank| rank)?),
                };
                rank <= fp
            }
        };
        Ok(if take_logged {
            self.logged_rank = None;
            self.logged.next()
        } else {
            self.unlogged.next().map(|(_, row)| row)
        })
    }

    fn next_record(&mut self) -> Result<Option<Record>> {
        let Some(row) = self.next_row()? else {
            return Ok(None);
        };
        let source = &self.source;
        let element = source.with_pair(row, |pair, _| encode(source.key(), pair))??;
        Ok(Some(insert(source.key(), element)))
    }
}

impl<S: PairRows> Iterator for RankMerge<S> {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Result<Record>> {
        if self.failed {
            return None;
        }
        let next = self.next_record();
        self.failed = next.is_err();
        next.transpose()
    }
}

/// Pairs and ranks a capture handed over, for a fold off the owner thread.
#[allow(dead_code)]
pub(crate) struct CapturedPairs<P> {
    key: String,
    pairs: Vec<(P, u128)>,
}

#[allow(dead_code)]
impl<P: Serialize> CapturedPairs<P> {
    /// Orders `pairs` by rank and returns the rows [`fold_pairs`] takes.
    pub(crate) fn new(key: String, mut pairs: Vec<(P, u128)>) -> (Self, Vec<usize>) {
        pairs.sort_unstable_by_key(|&(_, rank)| rank);
        let rows = (0..pairs.len()).collect();
        (CapturedPairs { key, pairs }, rows)
    }
}

impl<P: Serialize> PairRows for CapturedPairs<P> {
    type Pair = P;
    type Row = usize;

    fn key(&self) -> &str {
        &self.key
    }

    fn with_pair<R>(&self, row: usize, f: impl FnOnce(&P, u128) -> R) -> Result<R> {
        let (pair, rank) = &self.pairs[row];
        Ok(f(pair, *rank))
    }
}

/// Distinct elements and multiplicities a capture handed over, for a fold
/// off the owner thread.
#[allow(dead_code)]
pub(crate) struct CapturedElements<T> {
    key: String,
    elements: Vec<(T, usize)>,
}

#[allow(dead_code)]
impl<T: Serialize> CapturedElements<T> {
    /// Returns the elements with the fingerprints [`fold_elements`] takes.
    pub(crate) fn new(
        key: String,
        elements: Vec<(T, usize)>,
    ) -> Result<(Self, Fingerprints<usize>)> {
        let fingerprints = elements
            .iter()
            .enumerate()
            .map(|(row, (element, _))| Ok((element_fingerprint(&key, element)?, row)))
            .collect::<Result<_>>()?;
        Ok((CapturedElements { key, elements }, fingerprints))
    }
}

impl<T: Serialize> ElementRows for CapturedElements<T> {
    type Element = T;
    type Row = usize;

    fn key(&self) -> &str {
        &self.key
    }

    fn with_element<R>(&self, row: usize, f: impl FnOnce(&T, usize) -> R) -> Result<R> {
        let (element, count) = &self.elements[row];
        Ok(f(element, *count))
    }
}
