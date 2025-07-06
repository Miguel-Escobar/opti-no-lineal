pub fn squared_relu(x: f64) -> f64 {
    (x.max(0.0)).powi(2)
}

pub fn squared_relu_grad(x: f64) -> f64 {
    if x > 0.0 {
        2.0 * x
    } else {
        0.0
    }
}

pub fn squared(x: f64) -> f64 {
    x * x
}

pub fn squared_grad(x: f64) -> f64 {
    2.0 * x
}

pub fn alpha<const NUM_HARD_CONSTRAINTS: usize, const NUM_SOFT_CONSTRAINTS: usize>(ax_minus_b: [f64; NUM_HARD_CONSTRAINTS], ex_minus_e: [f64; NUM_SOFT_CONSTRAINTS]) -> f64 {
    let a_term: f64 = ax_minus_b.iter().map(|&x| squared_relu(x)).sum();
    let e_term: f64 = if  ex_minus_e.iter().map(|&x| squared(x)).sum();
    a_term + e_term
}

// Vectors must be smartly selected. Can be made more robust in the future.
pub fn alpha_partial_grad<const NUM_HARD_CONSTRAINTS: usize, const NUM_SOFT_CONSTRAINTS: usize, const N: usize>(ax_minus_b: [f64; NUM_HARD_CONSTRAINTS], ex_minus_e: [f64; NUM_SOFT_CONSTRAINTS], a_vec: [f64; N], e_vec: [f64; N]) -> [f64; N] {
    std::array::from_fn::<f64, N, _>(|i | squared_relu_grad(ax_minus_b[i])*a_vec[i] + squared(ex_minus_e[i])*e_vec[i])
}