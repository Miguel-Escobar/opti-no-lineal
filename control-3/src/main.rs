mod algorithms;

fn fn_for_nesterov(point: &[f64; 2]) -> f64 {
    point.iter().map(|x| x * x).sum()
}

fn gradf_for_nesterov(point: &[f64; 2]) -> [f64; 2] {
    point.map(|x| 2.0 * x)
}

fn fn_for_sgd<const N: usize>(x: &[f64; N]) -> f64 {
    x.windows(2)
        .map(|x| (50.0 * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2))/(N as f64 - 1.0))
        .sum()
}

fn element_wise_gradf<const N: usize>(x: &[f64; N], index: usize) -> [(f64, usize); 2] {
    match index {
        index if index == N - 2 => [(100.0 * (x[index] - x[index - 1].powi(2))/ (N as f64), index + 1), (0.0, index)],
        _ => {
            let diff_factor = x[index + 1] - x[index].powi(2);
            [
                ((100.0 * diff_factor) / (N as f64), index + 1),
                ((-200.0 * diff_factor * x[index] - 2.0 * (1.0 - x[index])) / (N as f64 - 1.0), index)
            ]
        }
    }
}

fn main() {
    let point_sgd: [f64; 1000] = [0.5; 1000];
    let test_val_sgd: [f64; 1000] =
        algorithms::sgd(fn_for_sgd, element_wise_gradf, point_sgd, 0.1, 100_000);
    println!("final val sgd = {}", fn_for_sgd(&test_val_sgd));
    let point_nesterov: [f64; 2] = [3.0, 5.0];
    let test_val_nesterov: [f64; 2] = algorithms::nesterov(
        fn_for_nesterov,
        gradf_for_nesterov,
        point_nesterov,
        0.1,
        0.1,
        100,
    );
    println!(
        "final val nesterov = {}",
        fn_for_nesterov(&test_val_nesterov)
    )
}
