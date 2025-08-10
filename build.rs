fn main() {
    println!("cargo:rerun-if-changed=src/sys");
    println!("cargo:rerun-if-changed=build.rs");

    // Torch
    println!("cargo:rustc-link-search=/usr/local/lib/python3.12/dist-packages/torch/lib");

    // CPU
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=c10");

    // CUDA
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=c10_cuda");

    // println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=z");

    cxx_build::bridges(["src/sys/features.rs", "src/sys/spectrogram.rs"])
        .file("3rdparty/cnpy/cnpy.cpp")
        .file("src/sys/spectrogram.cpp")
        .include("3rdparty")
        .include("/usr/local/lib/python3.12/dist-packages/torch/include")
        .include("/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include")
        .std("c++20")
        .cuda(true)
        .static_crt(cfg!(target_os = "windows"))
        .flag_if_supported("/EHsc")
        .compile("spectrogram");
}
