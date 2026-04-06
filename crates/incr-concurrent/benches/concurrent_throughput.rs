use criterion::{criterion_group, criterion_main, Criterion};
use incr_concurrent::Runtime;
use std::sync::Arc;
use std::thread;

fn concurrent_reads(c: &mut Criterion) {
    let rt = Arc::new(Runtime::new());
    let input = rt.create_input(42_u64);
    let query = rt.create_query(move |rt| rt.get(input) * 2);
    let _ = rt.get(query);

    for n_readers in [1, 2, 4, 8] {
        c.bench_function(&format!("concurrent_reads_{n_readers}"), |b| {
            b.iter(|| {
                let handles: Vec<_> = (0..n_readers)
                    .map(|_| {
                        let rt = rt.clone();
                        thread::spawn(move || {
                            for _ in 0..1000 {
                                let _ = rt.get(query);
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            })
        });
    }
}

fn concurrent_read_write(c: &mut Criterion) {
    let rt = Arc::new(Runtime::new());
    let input = rt.create_input(0_u64);
    let query = rt.create_query(move |rt| rt.get(input) + 1);
    let _ = rt.get(query);

    c.bench_function("concurrent_read_write_4readers", |b| {
        b.iter(|| {
            let writer_rt = rt.clone();
            let writer = thread::spawn(move || {
                for i in 0..100u64 {
                    writer_rt.set(input, i);
                }
            });
            let readers: Vec<_> = (0..4)
                .map(|_| {
                    let rt = rt.clone();
                    thread::spawn(move || {
                        for _ in 0..1000 {
                            let _ = rt.get(query);
                        }
                    })
                })
                .collect();
            writer.join().unwrap();
            for r in readers {
                r.join().unwrap();
            }
        })
    });
}

criterion_group!(benches, concurrent_reads, concurrent_read_write);
criterion_main!(benches);
