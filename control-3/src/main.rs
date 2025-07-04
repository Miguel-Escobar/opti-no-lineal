use rand::prelude::*;

fn fn_for_nesterov(point: &[f64; 2]) -> f64 {
    point.iter().map(|x| x * x).sum()
}

fn gradf_for_nesterov(point: &[f64; 2]) -> [f64; 2] {
    point.map(|x| 2.0 * x)
}

fn fn_for_sgd<const N: usize>(x: &[f64; N]) -> f64 {
    x.windows(2)
        .map(|x| 50.0 * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2))
        .sum()
}

fn batch_gradf<const N: usize>(x: &[f64; N], index: usize) -> [f64; N] {
    std::array::from_fn(|i| {
        if i == index {
            match i {
                i if (1..(N - 1)).contains(&i) => {
                    -200.0 * (x[i + 1] - x[i].powi(2)) * x[i] - 2.0 * (1.0 - x[i])
                        + 100.0 * (x[i] - x[i - 1].powi(2))
                }
                i if i == N - 1 => 100.0 * (x[i] - x[i - 1].powi(2)),
                i if i == 0 => -200.0 * (x[i + 1] - x[i].powi(2)) * x[i] - 2.0 * (1.0 - x[i]),
                _ => 0.0,
            }
        } else {
            0.0
        }
    })
}

fn nesterov(
    _f: fn(&[f64; 2]) -> f64,
    gradf: fn(&[f64; 2]) -> [f64; 2],
    x0: [f64; 2],
    alpha: f64,
    beta: f64,
    n: u32,
) -> [f64; 2] {
    let mut x_present: [f64; 2] = x0;
    let mut x_past: [f64; 2] = x0;
    for _ in 1..n {
        let v: [f64; 2] = std::array::from_fn(|i| x_present[i] + beta * (x_present[i] - x_past[i]));
        let grad_v: [f64; 2] = gradf(&v);
        x_past = x_present;
        x_present = std::array::from_fn(|i| v[i] - alpha * grad_v[i]);
    }
    x_present
}

fn sgd<const N: usize>(
    _f: fn(&[f64; N]) -> f64,
    gradf: fn(&[f64; N], usize) -> [f64; N],
    x0: [f64; N],
    alpha: f64,
    n: u32,
) -> [f64; N] {
    let mut x_present: [f64; N] = x0.clone();
    let mut rng = rand::rng();
    for _ in 1..n {
        let grad = gradf(&x_present, rng.random_range(0..N));
        x_present = std::array::from_fn(|i| x_present[i] - alpha * grad[i]);
    }
    x_present
}

fn main() {
    let point_sgd: [f64; 1000] = [0.5; 1000];
    let test_val_sgd: [f64; 1000] = sgd(fn_for_sgd, batch_gradf, point_sgd, 0.001, 1_000_000);
    println!("final val sgd = {}", fn_for_sgd(&test_val_sgd));

    let point_nesterov: [f64; 2] = [3.0, 5.0];
    let test_val_nesterov: [f64; 2] = nesterov(fn_for_nesterov, gradf_for_nesterov, point_nesterov, 0.1, 0.1, 100);
    println!("final val nesterov = {}", fn_for_nesterov(&test_val_nesterov))
}
