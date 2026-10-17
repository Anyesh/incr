fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // rustc passes `-z stack-size=1048576` for wasm32-unknown-unknown before
    // user link args, so this later value wins. 1 MiB overflows in debug well
    // short of the nesting depths the wasm tests must reach.
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        println!("cargo:rustc-link-arg=-zstack-size=8388608");
    }
}
