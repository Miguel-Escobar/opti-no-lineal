use rand::prelude::*;
use crate::probs::{FullGradient, SparseGradient};

pub fn nesterov<P, const N: usize>(prob: P, x0: [f64; N], alpha: f64, beta: f64, n: u32) -> ([f64; N], f64)
where
    P: FullGradient<N>
{
    let mut x_present: [f64; N] = x0;
    let mut x_past: [f64; N] = x0;
    for _ in 0..n {
        let v: [f64; N] = std::array::from_fn(|i| x_present[i] + beta * (x_present[i] - x_past[i]));
        let grad_v: [f64; N] = prob.gradient(&v);
        x_past = x_present;
        x_present = std::array::from_fn(|i| v[i] - alpha * grad_v[i]);
    }
    (x_present, prob.objective(&x_present))
}

pub fn sgd<const N: usize, const S: usize, P>(prob: P,x0: [f64; N], alpha: f64, n: usize) -> ([f64; N], f64)
where
    P: SparseGradient<N, S>,
{
    let mut x_present: [f64; N] = x0.clone();
    let mut rng = rand::rng();
    for _ in 0..n {
        let rand_term = rng.random_range(0..N);
        let grad_and_index = prob.sparse_gradient(&x_present, rand_term);
        for (component, i) in grad_and_index {
            x_present[i] -= alpha * component 
        }
    }
    (x_present, prob.objective(&x_present))
}