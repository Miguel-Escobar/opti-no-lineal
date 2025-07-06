mod algorithms;
mod probs;
mod utils;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

use crate::probs::ConstraintInfo;

#[pyfunction]
fn optimize_nesterov<'py>(
    py: Python<'py>,
    init_point: PyReadonlyArray1<'py, f64>,
    mu: f64,
    alpha: f64,
    beta: f64,
    n_iter: u32,
    problem: usize,
) ->  PyResult<(Py<PyArray1<f64>>, f64, f64)> {
    let init_point: [f64; 2] = init_point
        .as_slice()?
        .try_into()
        .expect("Init point must have length 2");
    let (point, f_val, test_val) = match problem {
        1 => {
            let prob = probs::NesterovProblem1 { mu };
            let (point, f_val) = algorithms::nesterov(prob, init_point, alpha, beta, n_iter);
            (point, f_val, probs::NesterovProblem1 { mu }.test_alpha_val(&point))
        }
        2 => {
            let prob = probs::NesterovProblem2 { mu };
            let (point, f_val) = algorithms::nesterov(prob, init_point, alpha, beta, n_iter);
            (point, f_val, probs::NesterovProblem1 { mu }.test_alpha_val(&point))
        }
        _ => panic!("Unvalid problem type! Only 1 and 2 are supported."),
    };
    let np_array = point.to_pyarray(py).to_owned();
    Ok((np_array.into(), f_val, test_val))
}

#[pyfunction]
fn optimize_sgd<'py>(
    py: Python<'py>,
    init_point: PyReadonlyArray1<'py, f64>,
    mu: f64,
    alpha: f64, 
    n_iter: usize,
) ->  PyResult<(Py<PyArray1<f64>>, f64, f64)> {
    let init_point: [f64; 1000] = init_point
        .as_slice()?
        .try_into()
        .expect("Init point must have length 1000");
    let (point, f_val) = algorithms::sgd(probs::SgdProblem{ mu }, init_point, alpha, n_iter);
    let np_array = point.to_pyarray(py).to_owned();
    Ok((np_array.into(), f_val, probs::SgdProblem{ mu }.test_alpha_val(&point)))
}


#[pymodule]
fn fescent<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_nesterov, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_sgd, m)?)?;
    Ok(())
}
