use nalgebra::{DMatrix, DVector};

use crate::error::{Error, Result, dim_error};

pub trait StationaryKernel {
    fn input_dimension(&self) -> usize;
    fn evaluate(&self, tau: &DVector<f64>) -> f64;

    fn gradient(&self, tau: &DVector<f64>, derivative_indices: &[usize]) -> DVector<f64> {
        let eps = 1e-6;
        DVector::from_iterator(
            derivative_indices.len(),
            derivative_indices.iter().map(|&idx| {
                let mut plus = tau.clone();
                let mut minus = tau.clone();
                plus[idx] += eps;
                minus[idx] -= eps;
                (self.evaluate(&plus) - self.evaluate(&minus)) / (2.0 * eps)
            }),
        )
    }
}

#[derive(Debug, Clone)]
pub struct SquaredExponentialKernel {
    sigma_squared: f64,
    length_scale_squared: DVector<f64>,
}

impl SquaredExponentialKernel {
    pub fn new(sigma: f64, length_scale: DVector<f64>) -> Result<Self> {
        if sigma <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "sigma",
                value: sigma,
            });
        }
        if length_scale.is_empty() {
            return Err(Error::Empty("length_scale"));
        }
        for value in length_scale.iter().copied() {
            if value <= 0.0 {
                return Err(Error::NonPositiveParameter {
                    name: "length_scale",
                    value,
                });
            }
        }
        Ok(Self {
            sigma_squared: sigma.powi(2),
            length_scale_squared: length_scale.map(|value| value.powi(2)),
        })
    }
}

impl StationaryKernel for SquaredExponentialKernel {
    fn input_dimension(&self) -> usize {
        self.length_scale_squared.len()
    }

    fn evaluate(&self, tau: &DVector<f64>) -> f64 {
        assert_eq!(tau.len(), self.input_dimension());
        let scaled_norm: f64 = tau
            .iter()
            .zip(self.length_scale_squared.iter())
            .map(|(tau_i, ell_sq)| tau_i.powi(2) / ell_sq)
            .sum();
        self.sigma_squared * (-0.5 * scaled_norm).exp()
    }

    fn gradient(&self, tau: &DVector<f64>, derivative_indices: &[usize]) -> DVector<f64> {
        let kernel = self.evaluate(tau);
        DVector::from_iterator(
            derivative_indices.len(),
            derivative_indices
                .iter()
                .map(|&idx| -kernel * tau[idx] / self.length_scale_squared[idx]),
        )
    }
}

#[derive(Debug, Clone)]
pub struct Matern32Kernel {
    sigma_squared: f64,
    length_scale_squared: DVector<f64>,
}

impl Matern32Kernel {
    pub fn new(sigma: f64, length_scale: DVector<f64>) -> Result<Self> {
        require_positive("sigma", sigma)?;
        require_positive_vector("length_scale", &length_scale)?;
        Ok(Self {
            sigma_squared: sigma.powi(2),
            length_scale_squared: length_scale.map(|value| value.powi(2)),
        })
    }
}

impl StationaryKernel for Matern32Kernel {
    fn input_dimension(&self) -> usize {
        self.length_scale_squared.len()
    }

    fn evaluate(&self, tau: &DVector<f64>) -> f64 {
        assert_eq!(tau.len(), self.input_dimension());
        let r = scaled_distance(tau, &self.length_scale_squared);
        let scaled = 3.0_f64.sqrt() * r;
        self.sigma_squared * (1.0 + scaled) * (-scaled).exp()
    }
}

#[derive(Debug, Clone)]
pub struct Matern52Kernel {
    sigma_squared: f64,
    length_scale_squared: DVector<f64>,
}

impl Matern52Kernel {
    pub fn new(sigma: f64, length_scale: DVector<f64>) -> Result<Self> {
        require_positive("sigma", sigma)?;
        require_positive_vector("length_scale", &length_scale)?;
        Ok(Self {
            sigma_squared: sigma.powi(2),
            length_scale_squared: length_scale.map(|value| value.powi(2)),
        })
    }
}

impl StationaryKernel for Matern52Kernel {
    fn input_dimension(&self) -> usize {
        self.length_scale_squared.len()
    }

    fn evaluate(&self, tau: &DVector<f64>) -> f64 {
        assert_eq!(tau.len(), self.input_dimension());
        let r = scaled_distance(tau, &self.length_scale_squared);
        let sqrt5_r = 5.0_f64.sqrt() * r;
        self.sigma_squared * (1.0 + sqrt5_r + 5.0 * r.powi(2) / 3.0) * (-sqrt5_r).exp()
    }
}

#[derive(Debug, Clone)]
pub struct PeriodicKernel {
    sigma_squared: f64,
    length_scale_squared: DVector<f64>,
    period: DVector<f64>,
}

impl PeriodicKernel {
    pub fn new(sigma: f64, length_scale: DVector<f64>, period: DVector<f64>) -> Result<Self> {
        require_positive("sigma", sigma)?;
        require_positive_vector("length_scale", &length_scale)?;
        require_positive_vector("period", &period)?;
        if length_scale.len() != period.len() {
            return Err(dim_error(
                "periodic kernel",
                length_scale.len().to_string(),
                period.len().to_string(),
            ));
        }
        Ok(Self {
            sigma_squared: sigma.powi(2),
            length_scale_squared: length_scale.map(|value| value.powi(2)),
            period,
        })
    }
}

impl StationaryKernel for PeriodicKernel {
    fn input_dimension(&self) -> usize {
        self.length_scale_squared.len()
    }

    fn evaluate(&self, tau: &DVector<f64>) -> f64 {
        assert_eq!(tau.len(), self.input_dimension());
        let scaled_periodic_distance: f64 = tau
            .iter()
            .zip(self.length_scale_squared.iter())
            .zip(self.period.iter())
            .map(|((tau_i, ell_sq), period_i)| {
                let sine = (std::f64::consts::PI * tau_i / period_i).sin();
                sine.powi(2) / ell_sq
            })
            .sum();
        self.sigma_squared * (-2.0 * scaled_periodic_distance).exp()
    }
}

#[derive(Debug, Clone)]
pub struct LocallyPeriodicKernel {
    periodic: PeriodicKernel,
    envelope: SquaredExponentialKernel,
}

impl LocallyPeriodicKernel {
    pub fn new(
        sigma: f64,
        periodic_length_scale: DVector<f64>,
        period: DVector<f64>,
        envelope_length_scale: DVector<f64>,
    ) -> Result<Self> {
        if periodic_length_scale.len() != envelope_length_scale.len() {
            return Err(dim_error(
                "locally periodic kernel",
                periodic_length_scale.len().to_string(),
                envelope_length_scale.len().to_string(),
            ));
        }
        Ok(Self {
            periodic: PeriodicKernel::new(1.0, periodic_length_scale, period)?,
            envelope: SquaredExponentialKernel::new(sigma, envelope_length_scale)?,
        })
    }
}

impl StationaryKernel for LocallyPeriodicKernel {
    fn input_dimension(&self) -> usize {
        self.periodic.input_dimension()
    }

    fn evaluate(&self, tau: &DVector<f64>) -> f64 {
        self.periodic.evaluate(tau) * self.envelope.evaluate(tau)
    }
}

#[derive(Debug, Clone)]
pub struct KernelSum<L, R> {
    left: L,
    right: R,
    input_dimension: usize,
}

impl<L: StationaryKernel, R: StationaryKernel> KernelSum<L, R> {
    pub fn new(left: L, right: R) -> Result<Self> {
        let input_dimension = require_same_kernel_dimension(&left, &right, "kernel sum")?;
        Ok(Self {
            left,
            right,
            input_dimension,
        })
    }
}

impl<L: StationaryKernel, R: StationaryKernel> StationaryKernel for KernelSum<L, R> {
    fn input_dimension(&self) -> usize {
        self.input_dimension
    }

    fn evaluate(&self, tau: &DVector<f64>) -> f64 {
        self.left.evaluate(tau) + self.right.evaluate(tau)
    }

    fn gradient(&self, tau: &DVector<f64>, derivative_indices: &[usize]) -> DVector<f64> {
        self.left.gradient(tau, derivative_indices) + self.right.gradient(tau, derivative_indices)
    }
}

#[derive(Debug, Clone)]
pub struct KernelProduct<L, R> {
    left: L,
    right: R,
    input_dimension: usize,
}

impl<L: StationaryKernel, R: StationaryKernel> KernelProduct<L, R> {
    pub fn new(left: L, right: R) -> Result<Self> {
        let input_dimension = require_same_kernel_dimension(&left, &right, "kernel product")?;
        Ok(Self {
            left,
            right,
            input_dimension,
        })
    }
}

impl<L: StationaryKernel, R: StationaryKernel> StationaryKernel for KernelProduct<L, R> {
    fn input_dimension(&self) -> usize {
        self.input_dimension
    }

    fn evaluate(&self, tau: &DVector<f64>) -> f64 {
        self.left.evaluate(tau) * self.right.evaluate(tau)
    }

    fn gradient(&self, tau: &DVector<f64>, derivative_indices: &[usize]) -> DVector<f64> {
        self.left.gradient(tau, derivative_indices) * self.right.evaluate(tau)
            + self.right.gradient(tau, derivative_indices) * self.left.evaluate(tau)
    }
}

#[derive(Debug, Clone)]
pub struct GaussianProcessData {
    pub input_data: DMatrix<f64>,
    pub output_data: DVector<f64>,
    pub output_noise_variance: f64,
}

impl GaussianProcessData {
    pub fn new(
        input_data: DMatrix<f64>,
        output_data: DVector<f64>,
        output_noise_variance: f64,
    ) -> Result<Self> {
        if input_data.ncols() != output_data.len() {
            return Err(dim_error(
                "gaussian process data",
                format!("{} outputs", input_data.ncols()),
                format!("{} outputs", output_data.len()),
            ));
        }
        if output_noise_variance < 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "output_noise_variance",
                value: output_noise_variance,
            });
        }
        Ok(Self {
            input_data,
            output_data,
            output_noise_variance,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GaussianProcess<K> {
    kernel: K,
    input_data: DMatrix<f64>,
    k_inv: DMatrix<f64>,
    k_inv_y: DVector<f64>,
    state_dependency: Vec<bool>,
    control_dependency: Vec<bool>,
    state_indices: Vec<usize>,
    control_indices: Vec<usize>,
    kernel_zero: f64,
}

impl<K: StationaryKernel> GaussianProcess<K> {
    pub fn new(
        data: GaussianProcessData,
        kernel: K,
        state_dependency: Vec<bool>,
        control_dependency: Vec<bool>,
    ) -> Result<Self> {
        let input_dim = data.input_data.nrows();
        if input_dim != kernel.input_dimension() {
            return Err(dim_error(
                "gaussian process kernel",
                kernel.input_dimension().to_string(),
                input_dim.to_string(),
            ));
        }
        let active_dim = state_dependency.iter().filter(|&&x| x).count()
            + control_dependency.iter().filter(|&&x| x).count();
        if active_dim != input_dim {
            return Err(dim_error(
                "gaussian process dependencies",
                input_dim.to_string(),
                active_dim.to_string(),
            ));
        }

        let n = data.input_data.ncols();
        let mut covariance = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let tau = data.input_data.column(i) - data.input_data.column(j);
                covariance[(i, j)] = kernel.evaluate(&tau.into_owned());
            }
        }
        for i in 0..n {
            covariance[(i, i)] += data.output_noise_variance;
        }
        let lu = covariance.lu();
        let identity = DMatrix::identity(n, n);
        let k_inv = lu
            .solve(&identity)
            .ok_or(Error::LinearSolve("gaussian process covariance"))?;
        let k_inv_y = &k_inv * &data.output_data;
        let state_indices: Vec<usize> =
            (0..state_dependency.iter().filter(|&&x| x).count()).collect();
        let control_indices = (state_indices.len()..input_dim).collect::<Vec<usize>>();
        let kernel_zero = kernel.evaluate(&DVector::zeros(input_dim));

        Ok(Self {
            kernel,
            input_data: data.input_data,
            k_inv,
            k_inv_y,
            state_dependency,
            control_dependency,
            state_indices,
            control_indices,
            kernel_zero,
        })
    }

    pub fn mean(&self, state: &DVector<f64>, control: &DVector<f64>) -> Result<f64> {
        let diff = self.point_differences(state, control)?;
        let k_star = self.kernel_vector(&diff);
        Ok(k_star.dot(&self.k_inv_y))
    }

    pub fn variance(&self, state: &DVector<f64>, control: &DVector<f64>) -> Result<f64> {
        let diff = self.point_differences(state, control)?;
        let k_star = self.kernel_vector(&diff);
        Ok(self.kernel_zero - k_star.dot(&(&self.k_inv * &k_star)))
    }

    pub fn mean_gradient_state(
        &self,
        state: &DVector<f64>,
        control: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        let diff = self.point_differences(state, control)?;
        let active = self.gradient_active(&diff, &self.state_indices, &self.k_inv_y);
        Ok(scatter_active(&self.state_dependency, &active))
    }

    pub fn mean_gradient_control(
        &self,
        state: &DVector<f64>,
        control: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        let diff = self.point_differences(state, control)?;
        let active = self.gradient_active(&diff, &self.control_indices, &self.k_inv_y);
        Ok(scatter_active(&self.control_dependency, &active))
    }

    pub fn variance_gradient_state(
        &self,
        state: &DVector<f64>,
        control: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        let diff = self.point_differences(state, control)?;
        let k_star = self.kernel_vector(&diff);
        let projected = &self.k_inv * k_star;
        let active = -2.0 * self.gradient_active(&diff, &self.state_indices, &projected);
        Ok(scatter_active(&self.state_dependency, &active))
    }

    pub fn variance_gradient_control(
        &self,
        state: &DVector<f64>,
        control: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        let diff = self.point_differences(state, control)?;
        let k_star = self.kernel_vector(&diff);
        let projected = &self.k_inv * k_star;
        let active = -2.0 * self.gradient_active(&diff, &self.control_indices, &projected);
        Ok(scatter_active(&self.control_dependency, &active))
    }

    fn kernel_vector(&self, point_diff: &DMatrix<f64>) -> DVector<f64> {
        DVector::from_fn(point_diff.ncols(), |i, _| {
            self.kernel.evaluate(&point_diff.column(i).into_owned())
        })
    }

    fn gradient_active(
        &self,
        point_diff: &DMatrix<f64>,
        indices: &[usize],
        projected: &DVector<f64>,
    ) -> DVector<f64> {
        let mut out = DVector::zeros(indices.len());
        for i in 0..point_diff.ncols() {
            let gradient = self
                .kernel
                .gradient(&point_diff.column(i).into_owned(), indices);
            out += gradient * projected[i];
        }
        out
    }

    fn point_differences(
        &self,
        state: &DVector<f64>,
        control: &DVector<f64>,
    ) -> Result<DMatrix<f64>> {
        if state.len() != self.state_dependency.len() {
            return Err(dim_error(
                "gaussian process state",
                self.state_dependency.len().to_string(),
                state.len().to_string(),
            ));
        }
        if control.len() != self.control_dependency.len() {
            return Err(dim_error(
                "gaussian process control",
                self.control_dependency.len().to_string(),
                control.len().to_string(),
            ));
        }
        let mut evaluation_point = DVector::zeros(self.input_data.nrows());
        let mut index = 0;
        for (i, depends) in self.state_dependency.iter().copied().enumerate() {
            if depends {
                evaluation_point[index] = state[i];
                index += 1;
            }
        }
        for (i, depends) in self.control_dependency.iter().copied().enumerate() {
            if depends {
                evaluation_point[index] = control[i];
                index += 1;
            }
        }
        let mut out = DMatrix::zeros(self.input_data.nrows(), self.input_data.ncols());
        for i in 0..self.input_data.ncols() {
            out.set_column(i, &(&evaluation_point - self.input_data.column(i)));
        }
        Ok(out)
    }
}

fn scatter_active(mask: &[bool], active: &DVector<f64>) -> DVector<f64> {
    let mut out = DVector::zeros(mask.len());
    let mut index = 0;
    for (i, depends) in mask.iter().copied().enumerate() {
        if depends {
            out[i] = active[index];
            index += 1;
        }
    }
    out
}

fn require_positive(name: &'static str, value: f64) -> Result<()> {
    if value <= 0.0 {
        Err(Error::NonPositiveParameter { name, value })
    } else {
        Ok(())
    }
}

fn require_positive_vector(name: &'static str, values: &DVector<f64>) -> Result<()> {
    if values.is_empty() {
        return Err(Error::Empty(name));
    }
    for value in values.iter().copied() {
        require_positive(name, value)?;
    }
    Ok(())
}

fn scaled_distance(tau: &DVector<f64>, length_scale_squared: &DVector<f64>) -> f64 {
    tau.iter()
        .zip(length_scale_squared.iter())
        .map(|(tau_i, ell_sq)| tau_i.powi(2) / ell_sq)
        .sum::<f64>()
        .sqrt()
}

fn require_same_kernel_dimension<L: StationaryKernel, R: StationaryKernel>(
    left: &L,
    right: &R,
    context: &'static str,
) -> Result<usize> {
    if left.input_dimension() != right.input_dimension() {
        return Err(dim_error(
            context,
            left.input_dimension().to_string(),
            right.input_dimension().to_string(),
        ));
    }
    Ok(left.input_dimension())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gp_interpolates_low_noise_training_point() {
        let data = GaussianProcessData::new(
            DMatrix::from_row_slice(1, 3, &[0.0, 1.0, 2.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
            1e-10,
        )
        .unwrap();
        let kernel = SquaredExponentialKernel::new(1.0, DVector::from_vec(vec![0.5])).unwrap();
        let gp = GaussianProcess::new(data, kernel, vec![true], vec![]).unwrap();
        let mean = gp
            .mean(&DVector::from_vec(vec![1.0]), &DVector::zeros(0))
            .unwrap();
        assert!((mean - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stationary_kernels_have_expected_value_at_origin() {
        let tau = DVector::from_vec(vec![0.0, 0.0]);
        let length_scale = DVector::from_element(2, 1.5);
        let period = DVector::from_element(2, 2.0);

        let se = SquaredExponentialKernel::new(2.0, length_scale.clone()).unwrap();
        let m32 = Matern32Kernel::new(2.0, length_scale.clone()).unwrap();
        let m52 = Matern52Kernel::new(2.0, length_scale.clone()).unwrap();
        let periodic = PeriodicKernel::new(2.0, length_scale.clone(), period.clone()).unwrap();
        let local =
            LocallyPeriodicKernel::new(2.0, length_scale.clone(), period, length_scale).unwrap();

        for value in [
            se.evaluate(&tau),
            m32.evaluate(&tau),
            m52.evaluate(&tau),
            periodic.evaluate(&tau),
            local.evaluate(&tau),
        ] {
            assert!((value - 4.0).abs() < 1e-12);
        }
    }

    #[test]
    fn kernel_sum_and_product_compose_values() {
        let tau = DVector::from_vec(vec![0.25]);
        let a = SquaredExponentialKernel::new(2.0, DVector::from_vec(vec![1.0])).unwrap();
        let b = Matern32Kernel::new(3.0, DVector::from_vec(vec![1.0])).unwrap();
        let expected_sum = a.evaluate(&tau) + b.evaluate(&tau);
        let expected_product = a.evaluate(&tau) * b.evaluate(&tau);

        let sum = KernelSum::new(a.clone(), b.clone()).unwrap();
        let product = KernelProduct::new(a, b).unwrap();

        assert!((sum.evaluate(&tau) - expected_sum).abs() < 1e-12);
        assert!((product.evaluate(&tau) - expected_product).abs() < 1e-12);
        assert!(
            KernelSum::new(
                SquaredExponentialKernel::new(1.0, DVector::from_vec(vec![1.0])).unwrap(),
                SquaredExponentialKernel::new(1.0, DVector::from_vec(vec![1.0, 2.0])).unwrap(),
            )
            .is_err()
        );
    }
}
