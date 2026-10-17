//! wasm32 tests and benches for incr-core. Run with
//! `cargo test -p incr-wasm-tests --target wasm32-unknown-unknown`, which
//! uses the wasm-bindgen-test-runner configured in `.cargo/config.toml`.

// Cargo cannot set `panic` per package, so this relies on the target's own
// default (wasm32-unknown-unknown aborts) and fails the build if a global
// RUSTFLAGS or profile change ever makes these tests unwind instead.
#[cfg(all(target_arch = "wasm32", not(panic = "abort")))]
compile_error!("incr-wasm-tests must be built with panic=abort on wasm32");
