// Main and spreadsheet-t10: try_get, cycle strategies and budgets exist.
const API: &str = "full";

fn configure<C: Cells>(rt: &Runtime<C>, frames: Option<u32>) {
    rt.set_placeholder(0_u64);
    rt.set_max_frames(frames.unwrap_or(u32::MAX));
    rt.set_max_stack_bytes(0);
}

fn try_read<C: Cells>(rt: &Runtime<C>, h: Incr<u64>) -> Option<u64> {
    rt.try_get(h).ok()
}

fn query<C: Cells, F>(rt: &Runtime<C>, f: F, strat: Strat, replay: bool) -> Incr<u64>
where
    F: Fn(&Runtime<C>) -> u64 + Send + Sync + 'static,
{
    use incr_core::{CycleStrategy, Fixpoint, QueryOpts, Replay};
    let cycle = match strat {
        Strat::None => CycleStrategy::None,
        Strat::Fallback(v) => CycleStrategy::fallback(move |_| v),
        Strat::Fixpoint { initial, limit } => {
            CycleStrategy::Fixpoint(Fixpoint::new(move || initial).max_iterations(limit))
        }
    };
    let replay = if replay {
        Replay::Allowed
    } else {
        Replay::Native
    };
    rt.create_query_with(f, QueryOpts { cycle, replay })
}
