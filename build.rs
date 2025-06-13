use std::env;

fn main() {
    let mklroot = env::var("MKLROOT").expect("MKLROOT environment variable not set");

    println!("cargo:rustc-link-search=native={}\\lib\\intel64", mklroot);
    println!("cargo:rustc-link-lib=dylib=mkl_rt");

    println!("cargo:rustc-link-search=native=C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2025.1\\lib");
    println!("cargo:rustc-link-lib=dylib=libiomp5md");
}
