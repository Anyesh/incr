//! A shop whose pricing config and order log live in an incr-persist file
//! log. Config entries are persisted inputs under the `config/` prefix,
//! orders are a persisted collection, and revenue is an incr query over
//! both, so every process that opens the log computes the same revenue.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use incr_core::Incr;
use incr_persist::{FileBackend, Input, Local, Options, PersistedCollection, Result, Store};
use serde::{Deserialize, Serialize};

const CONFIG: &str = "config/";
const ORDERS: &str = "orders";
const PRICE: &str = "price.";
const DISCOUNT: &str = "discount_pct";
const SKUS: [&str; 3] = ["apple", "fig", "pear"];

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    pub sku: String,
    pub qty: u64,
}

/// One acknowledged unit of work: each variant is a single append, so it is
/// either wholly in the log or not at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    Set {
        key: String,
        value: u64,
    },
    Order(Order),
    Cancel(Order),
    PriceAndOrder {
        key: String,
        value: u64,
        order: Order,
    },
}

pub struct Shop {
    store: Store<Local, FileBackend>,
    config: BTreeMap<String, Input<u64>>,
    orders: PersistedCollection<Order, Local>,
}

impl Shop {
    /// Opens the log in `dir` for writing, creating it if needed. Only one
    /// process at a time can hold it this way.
    pub fn open(dir: &Path) -> Result<Shop> {
        Shop::over(Store::open(FileBackend::open(dir)?, Options::default())?)
    }

    /// Opens an existing log without taking the writer lock, so it works
    /// while a writer is running. Fails if no writer has ever opened `dir`.
    pub fn open_read_only(dir: &Path) -> Result<Shop> {
        Shop::over(Store::open(
            FileBackend::open_read_only(dir)?,
            Options::default(),
        )?)
    }

    fn over(store: Store<Local, FileBackend>) -> Result<Shop> {
        let orders = store.collection(ORDERS)?;
        let mut shop = Shop {
            store,
            config: BTreeMap::new(),
            orders,
        };
        for key in shop.store.restored_keys(CONFIG)? {
            shop.declare(&key[CONFIG.len()..])?;
        }
        Ok(shop)
    }

    /// The input for config entry `name`, declared with default 0 if the
    /// log has never seen it.
    pub fn declare(&mut self, name: &str) -> Result<Input<u64>> {
        if let Some(input) = self.config.get(name) {
            return Ok(*input);
        }
        let input = self.store.input(&format!("{CONFIG}{name}"), 0)?;
        self.config.insert(name.to_owned(), input);
        Ok(input)
    }

    /// Returns once the op is durable (the store syncs every write) and
    /// applied to this process's runtime.
    pub fn apply(&mut self, op: &Op) -> Result<()> {
        match op {
            Op::Set { key, value } => {
                let input = self.declare(key)?;
                self.store.set(input, *value)?;
            }
            Op::Order(order) => {
                self.orders.insert(&self.store, order.clone())?;
            }
            Op::Cancel(order) => {
                self.orders.delete(&self.store, order)?;
            }
            Op::PriceAndOrder { key, value, order } => {
                let input = self.declare(key)?;
                self.store.batch(|b| {
                    b.set(input, *value).insert(&self.orders, order.clone());
                })?;
            }
        }
        Ok(())
    }

    pub fn checkpoint(&self) -> Result<()> {
        self.store.checkpoint().map(drop)
    }

    pub fn orders(&self) -> Vec<Order> {
        let mut orders = self.orders.handle().values(self.store.runtime());
        orders.sort();
        orders
    }

    /// Builds the revenue query: the sum of `qty * price.<sku>` over all
    /// orders, less `discount_pct` percent. The query reads the config
    /// entries declared when it is built, so declare every entry first;
    /// one declared afterwards is not part of it.
    pub fn revenue(&self) -> Incr<u64> {
        let rt = self.store.runtime();
        let quantities = self.orders.handle().reduce(rt, |orders: &[Order]| {
            let mut by_sku = BTreeMap::<String, u64>::new();
            for order in orders {
                *by_sku.entry(order.sku.clone()).or_default() += order.qty;
            }
            by_sku
        });
        let prices: BTreeMap<String, Incr<u64>> = self
            .config
            .iter()
            .filter_map(|(name, input)| {
                let sku = name.strip_prefix(PRICE)?;
                Some((sku.to_owned(), input.handle()))
            })
            .collect();
        let discount = self.config.get(DISCOUNT).map(|input| input.handle());
        rt.create_query(move |rt| {
            let gross: u64 = rt
                .get(quantities)
                .iter()
                .map(|(sku, qty)| qty * prices.get(sku).map_or(0, |price| rt.get(*price)))
                .sum();
            let discount = discount.map_or(0, |d| rt.get(d)).min(100);
            gross * (100 - discount) / 100
        })
    }

    pub fn view(&self, revenue: Incr<u64>) -> View {
        let rt = self.store.runtime();
        View {
            config: self
                .config
                .iter()
                .map(|(name, input)| (name.clone(), rt.get(input.handle())))
                .collect(),
            orders: self.orders(),
            revenue: rt.get(revenue),
        }
    }
}

/// Everything a process can observe about the shop. Two processes agree on
/// the shop's state exactly when their views print the same.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct View {
    pub config: BTreeMap<String, u64>,
    pub orders: Vec<Order>,
    pub revenue: u64,
}

impl fmt::Display for View {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "config:")?;
        for (name, value) in &self.config {
            writeln!(f, "  {name} = {value}")?;
        }
        writeln!(f, "orders:")?;
        for order in &self.orders {
            writeln!(f, "  #{} {} x{}", order.id, order.sku, order.qty)?;
        }
        writeln!(f, "revenue: {}", self.revenue)
    }
}

/// The config entries [`churn_op`] writes, which a churn writer declares
/// before its first op so that no op appends a `Declare` of its own.
pub fn churn_keys() -> Vec<String> {
    SKUS.iter()
        .map(|sku| format!("{PRICE}{sku}"))
        .chain([DISCOUNT.to_owned()])
        .collect()
}

/// The `i`th op of the deterministic workload `churn` runs. Every op
/// changes the view, so a restarted process shows exactly which ops landed.
pub fn churn_op(i: u64) -> Op {
    let sku = SKUS[(i / 6 % 3) as usize];
    let order = |id: u64| Order {
        id,
        sku: sku.to_owned(),
        qty: id % 5 + 1,
    };
    match i % 6 {
        0 | 2 => Op::Order(order(i)),
        1 => Op::Set {
            key: format!("{PRICE}{sku}"),
            value: 100 + i,
        },
        3 => Op::Cancel(order(i - 1)),
        4 => Op::Set {
            key: DISCOUNT.to_owned(),
            value: i % 30,
        },
        _ => Op::PriceAndOrder {
            key: format!("{PRICE}{sku}"),
            value: 1000 + i,
            order: order(i),
        },
    }
}
