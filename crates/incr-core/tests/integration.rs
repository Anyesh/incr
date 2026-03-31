// crates/incr-core/tests/integration.rs
use incr_core::{IncrCollection, Runtime};

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

    let full = rt.create_query(move |rt| {
        format!("{} {}", rt.get(first), rt.get(last))
    });

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

    let summary = rt.create_query(move |rt| {
        format!("{} active adults", rt.get(count))
    });

    users.insert(&rt, User { name: "Alice".into(), age: 30, active: true });
    users.insert(&rt, User { name: "Bob".into(), age: 16, active: true });
    users.insert(&rt, User { name: "Carol".into(), age: 25, active: false });

    assert_eq!(rt.get(summary), "1 active adults");

    users.insert(&rt, User { name: "Dave".into(), age: 22, active: true });
    assert_eq!(rt.get(summary), "2 active adults");

    users.delete(&rt, &User { name: "Alice".into(), age: 30, active: true });
    assert_eq!(rt.get(summary), "1 active adults");
}
