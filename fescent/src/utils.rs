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
    let e_term: f64 = ex_minus_e.iter().map(|&x| squared(x)).sum();
    a_term + e_term
}

// Vectors must be smartly selected. Can be made more robust in the future.
pub fn alpha_partial_grad<const NUM_HARD_CONSTRAINTS: usize, const NUM_SOFT_CONSTRAINTS: usize, const N: usize>(ax_minus_b: [f64; NUM_HARD_CONSTRAINTS], ex_minus_e: [f64; NUM_SOFT_CONSTRAINTS], a_vec: [f64; N], e_vec: [f64; N]) -> [f64; N] {
    let mut output: [f64; N] = [0.0; N];
    for i in 0..N {
        for ax_b in ax_minus_b {
            output[i] += ax_b*a_vec[i];
        };

        for ex_e in ex_minus_e {
            output[i] += ex_e*e_vec[i]
        }
    };
    output
}