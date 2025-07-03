fn funcion_de_prueba(point: &[f64; 2]) -> f64 {
    point.iter().map(|x| x * x).sum()
}

fn gradf_de_prueba(point: &[f64; 2]) -> [f64; 2] {
    point.map(|x| 2.0 * x)
}

fn nesterov(_f: fn(&[f64; 2]) -> f64, gradf: fn(&[f64; 2]) -> [f64; 2], x0: [f64; 2], alpha: f64, beta: f64, n: u32) -> [f64; 2] {
    let mut x_present: [f64; 2] = x0.clone();
    let mut x_past: [f64; 2] = x0.clone();
    for _ in 1..n {
        let v: [f64; 2] = std::array::from_fn(|i| {
            x_present[i] + beta * (x_present[i] - x_past[i])
        });
        let grad_v:[f64; 2] = gradf(&v);
        x_past = x_present;
        x_present = std::array::from_fn(|i| {
            v[i] - alpha * grad_v[i]
        });
    }
    x_present
}

fn sgd<const N: usize>(_f: fn(&[f64; N]) -> f64, gradf: fn(&[f64; N]) -> [f64; N], x0: [f64; N], alpha: f64, n: u32) -> [f64; N] {
    let mut x_present: [f64; N] = x0.clone();
    for _ in 1..n {
        let grad = gradf(&x_present);
        x_present = std::array::from_fn(|i| {
            x_present[i] - alpha * grad[i]
        });
    }
    x_present
}

fn main() {
    let point: [f64; 2] = [3.0, 2.0];
    let test_val: [f64; 2] = sgd(funcion_de_prueba, gradf_de_prueba, point, 0.1, 100);
    println!("point = {:?}", point);
    println!("test = {:?}", test_val);
}