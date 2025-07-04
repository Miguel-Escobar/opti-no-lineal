use rand::prelude::*;

pub fn nesterov<F, G>(_f: F, gradf: G, x0: [f64; 2], alpha: f64, beta: f64, n: u32) -> [f64; 2]
where
    F: Fn(&[f64; 2]) -> f64,
    G: Fn(&[f64; 2]) -> [f64; 2],
{
    let mut x_present: [f64; 2] = x0;
    let mut x_past: [f64; 2] = x0;
    for _ in 0..n {
        let v: [f64; 2] = std::array::from_fn(|i| x_present[i] + beta * (x_present[i] - x_past[i]));
        let grad_v: [f64; 2] = gradf(&v);
        x_past = x_present;
        x_present = std::array::from_fn(|i| v[i] - alpha * grad_v[i]);
    }
    x_present
}

pub fn sgd<const N: usize, const S: usize, F, G>(_f: F, gradf: G, x0: [f64; N], alpha: f64, n: u32) -> [f64; N]
where
    F: Fn(&[f64; N]) -> f64,
    G: Fn(&[f64; N], usize) -> [(f64, usize); S],
{
    let mut x_present: [f64; N] = x0.clone();
    let mut rng = rand::rng();
    for _ in 0..n {
        let rand_term = rng.random_range(0..(N-1));
        let grad_and_index = gradf(&x_present, rand_term);
        for (component, i) in grad_and_index {
            x_present[i] -= alpha * component
        }
    }
    x_present
}