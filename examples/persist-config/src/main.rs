use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use incr_persist_config::{churn_keys, churn_op, Op, Order, Shop};

const USAGE: &str = "usage: incr-persist-config <dir> <command>

commands:
  show [--read-only]   print config, orders and revenue
  set <name> <value>   set a config entry, e.g. price.apple 120 or discount_pct 10
  order <sku> <qty>    record an order
  cancel <id>          remove an order
  checkpoint           compact the log into a snapshot
  churn <from>         run the test workload from op <from> until killed";

const CHECKPOINT_EVERY: u64 = 16;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[&str]) -> Result<(), Box<dyn Error>> {
    let Some((dir, command)) = args.split_first() else {
        return Err(USAGE.into());
    };
    let dir = Path::new(dir);
    match command {
        ["show"] => show(&Shop::open(dir)?),
        ["show", "--read-only"] => show(&Shop::open_read_only(dir)?),
        ["set", name, value] => {
            let mut shop = Shop::open(dir)?;
            shop.apply(&Op::Set {
                key: (*name).to_owned(),
                value: value.parse()?,
            })?;
            show(&shop)
        }
        ["order", sku, qty] => {
            let mut shop = Shop::open(dir)?;
            let id = shop.orders().iter().map(|o| o.id + 1).max().unwrap_or(0);
            shop.apply(&Op::Order(Order {
                id,
                sku: (*sku).to_owned(),
                qty: qty.parse()?,
            }))?;
            show(&shop)
        }
        ["cancel", id] => {
            let mut shop = Shop::open(dir)?;
            let id: u64 = id.parse()?;
            let order = shop
                .orders()
                .into_iter()
                .find(|o| o.id == id)
                .ok_or_else(|| format!("no order #{id}"))?;
            shop.apply(&Op::Cancel(order))?;
            show(&shop)
        }
        ["checkpoint"] => {
            let shop = Shop::open(dir)?;
            shop.checkpoint()?;
            show(&shop)
        }
        ["churn", from] => churn(Shop::open(dir)?, from.parse()?),
        _ => Err(USAGE.into()),
    }
}

fn show(shop: &Shop) -> Result<(), Box<dyn Error>> {
    print!("{}", shop.view(shop.revenue()));
    Ok(())
}

/// Applies `churn_op(from)`, `churn_op(from + 1)`, ... forever, printing
/// `ack <i> revenue=<r>` once op `i` has returned Ok, which with the
/// default `Durability::Sync` means it is in the log for good.
fn churn(mut shop: Shop, from: u64) -> Result<(), Box<dyn Error>> {
    for key in churn_keys() {
        shop.declare(&key)?;
    }
    let revenue = shop.revenue();
    let mut out = std::io::stdout().lock();
    writeln!(out, "ready")?;
    out.flush()?;
    for i in from.. {
        shop.apply(&churn_op(i))?;
        let r = shop.view(revenue).revenue;
        writeln!(out, "ack {i} revenue={r}")?;
        out.flush()?;
        if i % CHECKPOINT_EVERY == CHECKPOINT_EVERY - 1 {
            shop.checkpoint()?;
        }
    }
    Ok(())
}
