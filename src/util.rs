#[cfg(feature = "intel-mkl")]
use std::os::raw::c_int;

use ndarray::{Array1, ArrayViewMut1, ArrayViewMut2};

pub fn l2_normalize(mut v: ArrayViewMut1<f32>) -> f32 {
    let norm = v.dot(&v).sqrt();

    if norm != 0. {
        v /= norm;
    }

    norm
}

pub fn l2_normalize_array(mut v: ArrayViewMut2<f32>) -> Array1<f32> {
    let mut norms = Vec::with_capacity(v.nrows());
    for embedding in v.outer_iter_mut() {
        norms.push(l2_normalize(embedding));
    }

    norms.into()
}

#[cfg(feature = "intel-mkl")]
#[allow(dead_code)]
#[no_mangle]
extern "C" fn mkl_serv_intel_cpu_true() -> c_int {
    1
}
