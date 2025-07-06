use pyo3::prelude::*;
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
mod algorithms;
mod utils;
mod probs;

#[pyfunction]
fn optimize_nesterov(mu: f64, alpha: f64, beta: f64, n_iter: usize, problem: usize) -> (PyArrayDyn<f64>, f64) {
    let init_point: [f64; 2] = [0.0, 0.0];
    let objective = match problem {
        1 => |point: &[f64; 2]| probs::fn_for_nesterov_1(point) + mu * utils::alpha(probs::nesterov_ab_1(point),probs::nesterov_ee_1(point)),
        2 => |point: &[f64; 2]| probs::fn_for_nesterov_2(point) + mu * utils::alpha(probs::nesterov_ab_2(point),probs::nesterov_ee_2(point)),
    };
    let grad_objective = match problem {
        1 => |point: &[f64; 2]| {
            let grad = gradf_for_nesterov_1(point);
            let penalization_grad = penalization_grad_nesterov_1(point);
            [grad[0] + mu * penalization_grad[0], grad[1] + mu * penalization_grad[1]]
        },
        2 => |point: &[f64; 2]| {
            let grad = gradf_for_nesterov_2(point);
            let penalization_grad = penalization_grad_nesterov_2(point);
            [grad[0] + mu * penalization_grad[0], grad[1] + mu * penalization_grad[1]]
        },
        _ => panic!("Invalid problem number. Options: 1 or 2."),
    };


}

#[pyfunction]
fn optimize_sgd(mu: f64, alpha: f64, n_iter: usize) -> (PyArrayDyn<f64>, f64) {}
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
