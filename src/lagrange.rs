use core::ops::Mul;
use ark_std::{vec, vec::Vec};
use ark_ff::FftField;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};

use crate::cfg_iter;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::Error;

#[derive(Debug, Clone)]
pub(crate) struct LagrangeInterpContext<F: FftField> {
    lag_polys: Vec<DensePolynomial<F>>,
}

impl<F: FftField> LagrangeInterpContext<F> {
    pub fn new_from_points(points: &[F]) -> Result<Self, Error> {
        // Generate the non-normalized lagrange polynomials. These are zero on all points other
        // than the target point, and some nonzero value on that point.
        let non_normalized_polys: Vec<_> = cfg_iter!(points)
            .map(|(j, _x_j)| {
                let mut l_poly: DensePolynomial<F> =
                    DensePolynomial::from_coefficients_vec(vec![F::one()]);
                for (k, x_k) in points.iter().enumerate() {
                    if j == k {
                        continue;
                    }
                    let tmp_poly: DensePolynomial<F> =
                        DensePolynomial::from_coefficients_vec(vec![-(*x_k), F::one()]);
                    // This does fft mul... not sure if it's actually faster
                    l_poly = l_poly.mul(&tmp_poly);
                }
                l_poly
            })
            .collect();

        let lag_polys = cfg_iter!(non_normalized_polys)
            .map(|(i, non_normed)| {
                // Evaluate at target, divide by result
                // so that the polynomial evaluates to 1 at the target
                non_normed
                    .evaluate(&points[i])
                    .inverse()
                    .ok_or(Error::DivisorIsZero)
                    .map(|v| non_normed * v)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self { lag_polys })
    }

    /// Given evals $((y_{1, 1}, \ldots y_{1_k}), \ldots (y_{l, 1}, \ldots y_{l, k}))$, points
    /// $(x_1, \ldots x_k)$, and scalars $(\gamma_1, \ldots, \gamma_l)$, this method
    /// computes $\sum_{i=1}^l \gamma_i r_i$ where $r_i$ is the unique degree $k$ polynomial such that
    /// $r_i(x_j) = y_{i, j}$
    pub fn lagrange_interp_linear_combo(
        &self,
        evals: &[impl AsRef<[F]>],
        scalars: &[F],
    ) -> Result<DensePolynomial<F>, Error> {
        let mut targets = vec![F::zero(); self.lag_polys.len()];
        for i in 0..evals.len() {
            let eval = evals[i].as_ref();
            for j in 0..eval.len() {
                // Our target at x_j is \sum gamma_i * y_{i, j}
                // Does this as_ref() call introduce any overhead?
                targets[j] += scalars[i] * eval[j];
            }
        }
        // Now we just interpolate to targets
        targets
            .into_iter()
            .enumerate()
            .map(|(j, target)| self.lag_polys[j].mul(target))
            .reduce(|x, y| x + y)
            .ok_or(Error::NoPointsGiven)
    }
}
