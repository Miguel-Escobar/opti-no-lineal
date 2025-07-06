use crate::utils;

fn fn_for_nesterov_1(point: &[f64; 2]) -> f64 {
    point.iter().map(|x| x * x).sum()
}

fn gradf_for_nesterov_1(point: &[f64; 2]) -> [f64; 2] {
    point.map(|x| 2.0 * x)
}

fn fn_for_nesterov_2(point: &[f64; 2]) -> f64 {
    (1.0 - point[0]).powf(1.5) + 100.0 * (point[1] - point[0].powi(2)).powi(2)
}

fn gradf_for_nesterov_2(point: &[f64; 2]) -> [f64; 2] {
    [
        -1.5 * (1.0 - point[0]).powf(0.5) - 400.0 * (point[1] - point[0].powi(2)) * point[0],
        200.0 * (point[1] - point[0].powi(2)),
    ]
}

fn fn_for_sgd<const N: usize>(x: &[f64; N]) -> f64 {
    x.windows(2)
        .map(|x| (50.0 * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2)))
        .sum()
}

fn element_wise_gradf<const N: usize>(x: &[f64; N], index: usize) -> [(f64, usize); 2] {
    let diff_factor = x[index + 1] - x[index].powi(2);
    [
        ((100.0 * diff_factor), index + 1),
        ((-200.0 * diff_factor * x[index] - 2.0 * (1.0 - x[index])), index)
    ]
}

pub trait OptProblem<const DOMAIN_SIZE: usize> {
    fn objective(&self, point: &[f64; DOMAIN_SIZE]) -> f64;
}
pub trait FullGradient<const DOMAIN_SIZE: usize>: OptProblem<DOMAIN_SIZE>{
    fn gradient(&self, point: &[f64; DOMAIN_SIZE]) -> [f64; DOMAIN_SIZE];
}

pub trait SparseGradient<const DOMAIN_SIZE: usize, const N: usize>: OptProblem<DOMAIN_SIZE> {
    fn sparse_gradient(&self, point: &[f64; DOMAIN_SIZE], index: usize) -> [(f64, usize); N];
}

pub trait ConstraintInfo<const NUM_SOFT_CONSTRAINTS: usize, const NUM_HARD_CONSTRAINTS: usize, const DOMAIN_SIZE: usize> {
    const A_VEC: [f64; DOMAIN_SIZE];
    const E_VEC: [f64; DOMAIN_SIZE];
    fn ax_minus_b(&self, point: &[f64; DOMAIN_SIZE]) -> [f64; NUM_SOFT_CONSTRAINTS];
    fn ex_minus_e(&self, point: &[f64; DOMAIN_SIZE]) -> [f64; NUM_HARD_CONSTRAINTS];
    fn test_alpha_val(&self, point: &[f64; DOMAIN_SIZE]) -> f64;
}

pub struct NesterovProblem1 {
    pub mu: f64,
}

impl ConstraintInfo<1, 0, 2> for NesterovProblem1 {
    const A_VEC: [f64; 2] = [1.0, 1.0]; // Son las filas de la matriz A sumadas (osea sumas sobre las columnas).
    
    const E_VEC: [f64; 2] = [0.0, 0.0]; // Ídem pero para E.

    fn ax_minus_b(&self, point: &[f64; 2]) -> [f64; 1] {
        [(point[0] + point[1]) + 100.0]
    }
    fn ex_minus_e(&self, _point: &[f64; 2]) -> [f64; 0] {
        []
    }

    fn test_alpha_val(&self, point: &[f64; 2]) -> f64 {
        self.mu * utils::alpha(self.ax_minus_b(point), self.ex_minus_e(point))
    }
}


impl OptProblem<2> for NesterovProblem1 {
    fn objective(&self, point: &[f64; 2]) -> f64 {
        fn_for_nesterov_1(point) + self.mu * utils::alpha(
            self.ax_minus_b(point),
            self.ex_minus_e(point)
        )
    }
}

impl FullGradient<2> for NesterovProblem1 {
    fn gradient(&self, point: &[f64; 2]) -> [f64; 2] {
        let obj_grad = gradf_for_nesterov_1(point);
        let alpha_grad: [f64; 2] = utils::alpha_partial_grad(self.ax_minus_b(point), self.ex_minus_e(point), NesterovProblem1::A_VEC, NesterovProblem1::E_VEC);
        std::array::from_fn::<f64, 2, _>(|i: usize| obj_grad[i] + self.mu * alpha_grad[i])
    }
}

pub struct NesterovProblem2 {
    pub mu: f64
}


impl ConstraintInfo<1, 1, 2> for NesterovProblem2 {
    const A_VEC: [f64; 2] = [1.0, 1.0]; // Son las filas de la matriz A sumadas (osea sumas sobre las columnas).
    
    const E_VEC: [f64; 2] = [1.0, -5.0]; // Ídem pero para E.

    fn ax_minus_b(&self, point: &[f64; 2]) -> [f64; 1] {
        [(point[0] + point[1]) - 5.0]
    }
    fn ex_minus_e(&self, point: &[f64; 2]) -> [f64; 1] {
        [(point[0] - 5.0 * point[1]) - 2.0]
    }

    fn test_alpha_val(&self, point: &[f64; 2]) -> f64 {
        self.mu * utils::alpha(self.ax_minus_b(point), self.ex_minus_e(point))
    }
}


impl OptProblem<2> for NesterovProblem2 {
    fn objective(&self, point: &[f64; 2]) -> f64 {
        fn_for_nesterov_2(point) + self.mu * utils::alpha(
            self.ax_minus_b(point),
            self.ex_minus_e(point)
        )
    }
}

impl FullGradient<2> for NesterovProblem2 {
    fn gradient(&self, point: &[f64; 2]) -> [f64; 2] {
        let obj_grad = gradf_for_nesterov_2(point);
        let alpha_grad: [f64; 2] = utils::alpha_partial_grad(self.ax_minus_b(point), self.ex_minus_e(point), NesterovProblem2::A_VEC, NesterovProblem2::E_VEC);
        std::array::from_fn::<f64, 2, _>(|i: usize| obj_grad[i] + self.mu * alpha_grad[i])
    }
}

pub struct SgdProblem {
    pub mu: f64
}

impl ConstraintInfo<0, 1, 1000> for SgdProblem {
    const A_VEC: [f64; 1000] = [0.0; 1000]; // Son las filas de la matriz A sumadas (osea sumas sobre las columnas).
    
    const E_VEC: [f64; 1000] = [1.0; 1000]; // Ídem pero para E.

    fn ax_minus_b(&self, _point: &[f64; 1000]) -> [f64; 0] {
        []
    }

    fn ex_minus_e(&self, point: &[f64; 1000]) -> [f64; 1] {
        [point.iter().sum::<f64>() - 1001.0]
    }

    fn test_alpha_val(&self, point: &[f64; 1000]) -> f64 {
        self.mu * utils::alpha(self.ax_minus_b(point), self.ex_minus_e(point))
    }
}


impl OptProblem<1000> for SgdProblem {
    fn objective(&self, point: &[f64; 1000]) -> f64 {
        fn_for_sgd(point) + self.mu * utils::alpha(self.ax_minus_b(point), self.ex_minus_e(point))
    }
}

impl SparseGradient<1000, 3> for SgdProblem {
    fn sparse_gradient(&self, point: &[f64; 1000], index: usize) -> [(f64, usize); 3] {
        let penalization_term = utils::alpha_partial_grad(self.ax_minus_b(point), self.ex_minus_e(point), [0.0], [1.0]);
        let objective_term = element_wise_gradf(point, index);
        [
            objective_term[0],
            objective_term[1],
            (self.mu * penalization_term[0], index),
        ]
    }
}
