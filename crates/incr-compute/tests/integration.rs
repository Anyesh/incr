use incr_compute::{IncrCollection, Runtime};

#[test]
fn spec_example_width_height_area() {
    let rt = Runtime::new();

    let width = rt.create_input(10.0_f64);
    let height = rt.create_input(5.0_f64);

    let area = rt.create_query(move |rt| rt.get(width) * rt.get(height));

    let description = rt.create_query(move |rt| format!("Area is {}", rt.get(area)));

    assert_eq!(rt.get(description), "Area is 50");

    rt.set(width, 12.0);
    assert_eq!(rt.get(description), "Area is 60");
}

#[test]
fn spec_example_incremental_updates() {
    let rt = Runtime::new();

    let x = rt.create_input(1_i64);
    let y = rt.create_input(2_i64);

    let sum = rt.create_query(move |rt| rt.get(x) + rt.get(y));
    let doubled = rt.create_query(move |rt| rt.get(sum) * 2);
    let label = rt.create_query(move |rt| format!("result: {}", rt.get(doubled)));

    assert_eq!(rt.get(label), "result: 6"); // (1+2)*2 = 6

    rt.set(x, 10);
    assert_eq!(rt.get(label), "result: 24"); // (10+2)*2 = 24

    rt.set(y, 5);
    assert_eq!(rt.get(label), "result: 30"); // (10+5)*2 = 30
}

#[test]
fn complex_graph_with_early_cutoff() {
    use std::cell::Cell;
    use std::rc::Rc;

    let rt = Runtime::new();

    let raw_score = rt.create_input(85_i64);

    let normalize_count = Rc::new(Cell::new(0_u32));
    let nc = normalize_count.clone();
    let normalized = rt.create_query(move |rt| {
        nc.set(nc.get() + 1);
        rt.get(raw_score).clamp(0, 100)
    });

    let format_count = Rc::new(Cell::new(0_u32));
    let fc = format_count.clone();
    let display = rt.create_query(move |rt| {
        fc.set(fc.get() + 1);
        let score = rt.get(normalized);
        if score >= 90 {
            "A".to_string()
        } else if score >= 80 {
            "B".to_string()
        } else {
            "C".to_string()
        }
    });

    assert_eq!(rt.get(display), "B");
    assert_eq!(normalize_count.get(), 1);
    assert_eq!(format_count.get(), 1);

    rt.set(raw_score, 95);
    assert_eq!(rt.get(display), "A");
    assert_eq!(normalize_count.get(), 2);
    assert_eq!(format_count.get(), 2);

    rt.set(raw_score, 150);
    assert_eq!(rt.get(display), "A");
    assert_eq!(normalize_count.get(), 3);
    assert_eq!(format_count.get(), 3);

    // Early cutoff: 200 clamped to 100, same as 150 clamped to 100
    rt.set(raw_score, 200);
    assert_eq!(rt.get(display), "A");
    assert_eq!(normalize_count.get(), 4);
    assert_eq!(format_count.get(), 3); // NOT recomputed — early cutoff!
}

#[test]
fn string_values_work() {
    let rt = Runtime::new();

    let first = rt.create_input("Hello".to_string());
    let last = rt.create_input("World".to_string());

    let full = rt.create_query(move |rt| format!("{} {}", rt.get(first), rt.get(last)));

    assert_eq!(rt.get(full), "Hello World");

    rt.set(first, "Goodbye".to_string());
    assert_eq!(rt.get(full), "Goodbye World");
}

#[test]
fn collection_feeds_function_query() {
    let rt = Runtime::new();
    let scores = rt.create_collection::<i64>();
    let high_scores = scores.filter(&rt, |s| *s >= 90);
    let count = high_scores.count(&rt);

    let summary = rt.create_query(move |rt| {
        let n = rt.get(count);
        format!("{} students scored 90+", n)
    });

    scores.insert(&rt, 85);
    scores.insert(&rt, 92);
    scores.insert(&rt, 78);
    scores.insert(&rt, 95);

    assert_eq!(rt.get(summary), "2 students scored 90+");

    scores.insert(&rt, 91);
    assert_eq!(rt.get(summary), "3 students scored 90+");

    scores.delete(&rt, &92);
    assert_eq!(rt.get(summary), "2 students scored 90+");
}

#[test]
fn full_pipeline_filter_map_count_query() {
    #[derive(Clone, Hash, Eq, PartialEq, Debug)]
    struct User {
        name: String,
        age: i32,
        active: bool,
    }

    let rt = Runtime::new();
    let users: IncrCollection<User> = rt.create_collection();

    let active_adults = users
        .filter(&rt, |u| u.active)
        .filter(&rt, |u| u.age >= 18)
        .map(&rt, |u| u.name.clone());

    let count = active_adults.count(&rt);

    let summary = rt.create_query(move |rt| format!("{} active adults", rt.get(count)));

    users.insert(
        &rt,
        User {
            name: "Alice".into(),
            age: 30,
            active: true,
        },
    );
    users.insert(
        &rt,
        User {
            name: "Bob".into(),
            age: 16,
            active: true,
        },
    );
    users.insert(
        &rt,
        User {
            name: "Carol".into(),
            age: 25,
            active: false,
        },
    );

    assert_eq!(rt.get(summary), "1 active adults");

    users.insert(
        &rt,
        User {
            name: "Dave".into(),
            age: 22,
            active: true,
        },
    );
    assert_eq!(rt.get(summary), "2 active adults");

    users.delete(
        &rt,
        &User {
            name: "Alice".into(),
            age: 30,
            active: true,
        },
    );
    assert_eq!(rt.get(summary), "1 active adults");
}

#[test]
fn sort_pairwise_map_reduce_pipeline() {
    // Simulates: given a set of visit timestamps, compute total gaps between
    // consecutive visits. This is the core pattern for travel time calculation.
    let rt = Runtime::new();
    let visits = rt.create_collection::<i64>(); // timestamps

    let sorted = visits.sort_by_key(&rt, |t: &i64| *t);
    let pairs = sorted.pairwise(&rt);

    let gaps = pairs.map(&rt, |(a, b): &(i64, i64)| b - a);

    // Sum all gaps
    let total_gap = gaps.reduce(&rt, |elements| -> i64 { elements.iter().sum() });

    // Start with visits at times 10, 30, 50
    visits.insert(&rt, 10);
    visits.insert(&rt, 30);
    visits.insert(&rt, 50);
    assert_eq!(rt.get(total_gap), 40); // (30-10) + (50-30) = 40

    // Insert visit at time 20: gaps become 10 + 10 + 20 = 40 (same total!)
    visits.insert(&rt, 20);
    assert_eq!(rt.get(total_gap), 40); // (20-10) + (30-20) + (50-30) = 40

    // Delete visit at time 30: gaps become 10 + 30 = 40 (still same!)
    visits.delete(&rt, &30);
    assert_eq!(rt.get(total_gap), 40); // (20-10) + (50-20) = 40

    // Insert visit at time 100: adds a big gap
    visits.insert(&rt, 100);
    assert_eq!(rt.get(total_gap), 90); // (20-10) + (50-20) + (100-50) = 90

    visits.delete(&rt, &10);
    assert_eq!(rt.get(total_gap), 80); // (50-20) + (100-50) = 80
}

#[test]
fn pipeline_early_cutoff() {
    // Verify that early cutoff works through the full pipeline:
    // if total doesn't change, downstream isn't recomputed
    use std::cell::Cell;
    use std::rc::Rc;

    let rt = Runtime::new();
    let visits = rt.create_collection::<i64>();
    let sorted = visits.sort_by_key(&rt, |t: &i64| *t);
    let pairs = sorted.pairwise(&rt);
    let gaps = pairs.map(&rt, |(a, b): &(i64, i64)| b - a);
    let total_gap = gaps.reduce(&rt, |elements| -> i64 { elements.iter().sum() });

    let downstream_evals = Rc::new(Cell::new(0_u32));
    let dc = downstream_evals.clone();
    let label = rt.create_query(move |rt| {
        dc.set(dc.get() + 1);
        format!("total={}", rt.get(total_gap))
    });

    visits.insert(&rt, 10);
    visits.insert(&rt, 30);
    visits.insert(&rt, 50);
    assert_eq!(rt.get(label), "total=40");
    assert_eq!(downstream_evals.get(), 1);

    // Insert 20 between 10 and 30: total gap is still 40
    visits.insert(&rt, 20);
    assert_eq!(rt.get(label), "total=40");
    // Early cutoff: total_gap unchanged, so label shouldn't recompute
    assert_eq!(downstream_evals.get(), 1);
}
