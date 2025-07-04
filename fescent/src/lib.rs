use pyo3::prelude::*;
mod algorithms;

fn fn_for_nesterov(point: &[f64; 2]) -> f64 {
    point.iter().map(|x| x * x).sum()
}

fn gradf_for_nesterov(point: &[f64; 2]) -> [f64; 2] {
    point.map(|x| 2.0 * x)
}

fn fn_for_sgd<const N: usize>(x: &[f64; N]) -> f64 {
    x.windows(2)
        .map(|x| (50.0 * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2))/(N as f64))
        .sum()
}

fn element_wise_gradf<const N: usize>(x: &[f64; N], index: usize) -> [(f64, usize); 2] {
    match index {
        index if index == N - 2 => [(100.0 * (x[index] - x[index - 1].powi(2))/ (N as f64), index + 1), (0.0, index)],
        _ => {
            let diff_factor = x[index + 1] - x[index].powi(2);
            [
                ((100.0 * diff_factor) / (N as f64), index + 1),
                ((-200.0 * diff_factor * x[index] - 2.0 * (1.0 - x[index])) / (N as f64), index)
            ]
        }
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn fescent(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
