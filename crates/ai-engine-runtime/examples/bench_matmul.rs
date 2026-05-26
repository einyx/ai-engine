//! Quick benchmark: GEMM (burn matmul) vs ndarray GEMV for seq=1 decode.

use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use ndarray::{Array1, Array2, linalg::{general_mat_mul, general_mat_vec_mul}};
use std::time::Instant;

type B = NdArray;

const IN: usize = 2048;
const OUT: usize = 8192;
const N: usize = 50;

fn bench_burn_matmul() -> f64 {
    let dev = burn_ndarray::NdArrayDevice::default();
    let x: Tensor<B, 3> = Tensor::ones([1, 1, IN], &dev);
    let w: Tensor<B, 2> = Tensor::ones([IN, OUT], &dev);

    // warm up
    let _ = x.clone().matmul(w.clone().unsqueeze::<3>());

    let t0 = Instant::now();
    for _ in 0..N {
        let _ = x.clone().matmul(w.clone().unsqueeze::<3>());
    }
    t0.elapsed().as_secs_f64() * 1000.0 / N as f64
}

fn bench_ndarray_gemv() -> f64 {
    // w stored as [IN, OUT]; for sgemv we need to call it as transposed
    // i.e., w.t() is [OUT, IN], x is [IN], result is [OUT]
    let w = Array2::<f32>::ones((IN, OUT));
    let x_vec = Array1::<f32>::ones(IN);
    let mut out = Array1::<f32>::zeros(OUT);

    // w.t() is [OUT, IN] - use that for gemv
    // warm up
    general_mat_vec_mul(1.0f32, &w.t(), &x_vec, 0.0f32, &mut out);

    let t0 = Instant::now();
    for _ in 0..N {
        general_mat_vec_mul(1.0f32, &w.t(), &x_vec, 0.0f32, &mut out);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / N as f64
}

fn bench_ndarray_gemm_m1() -> f64 {
    // Same as what burn does: [1, IN] × [IN, OUT]
    let a = Array2::<f32>::ones((1, IN));
    let b = Array2::<f32>::ones((IN, OUT));
    let mut c = Array2::<f32>::zeros((1, OUT));

    general_mat_mul(1.0f32, &a, &b, 0.0f32, &mut c);

    let t0 = Instant::now();
    for _ in 0..N {
        general_mat_mul(1.0f32, &a, &b, 0.0f32, &mut c);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / N as f64
}

fn main() {
    let gemm_burn = bench_burn_matmul();
    let gemm_nd = bench_ndarray_gemm_m1();
    let gemv_nd = bench_ndarray_gemv();

    println!("burn matmul [1,1,{IN}] x [1,{IN},{OUT}]: {gemm_burn:.3} ms/call");
    println!("ndarray gemm [1,{IN}] x [{IN},{OUT}]: {gemm_nd:.3} ms/call");
    println!("ndarray gemv w.t()[{OUT},{IN}] x [{IN}]: {gemv_nd:.3} ms/call");
    let speedup = gemm_burn / gemv_nd;
    println!("speedup gemv vs burn matmul: {speedup:.2}x");
}
