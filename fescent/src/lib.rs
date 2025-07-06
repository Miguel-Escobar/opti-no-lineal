mod algorithms;
mod probs;
mod utils;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn optimize_nesterov<'py>(
    init_point: PyReadonlyArray1<'py, f64>,
    mu: f64,
    alpha: f64,
    beta: f64,
    n_iter: u32,
    problem: usize,
) -> PyResult<([f64; 2], f64)> {
    let init_point: [f64; 2] = init_point
        .as_slice()?
        .try_into()
        .expect("Init point must have length 2");
    let (point, f_val) = match problem {
        1 => {
            let prob = probs::NesterovProblem1 { mu };
            algorithms::nesterov(prob, init_point, alpha, beta, n_iter)
        }
        2 => {
            let prob = probs::NesterovProblem2 { mu };
            algorithms::nesterov(prob, init_point, alpha, beta, n_iter)
        }
        _ => panic!("Unvalid problem type! Only 1 and 2 are supported."),
    };
    Ok((point, f_val))
}

#[pymodule]
fn fescent<'py>(py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_nesterov, m)?)?;
    Ok(())
}
