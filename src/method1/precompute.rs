use ark_poly::univariate::DensePolynomial;
use merlin::Transcript;
use std::usize;

use ark_ec::pairing::Pairing;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{vanishing_polynomial, Error, Proof};
use crate::lagrange::LagrangeInterpContext;
use crate::traits::{Committer, PolyMultiProof};
use crate::{cfg_iter, Commitment};

#[derive(Clone, Debug)]
pub struct M1Precomp<E: Pairing> {
    pub inner: super::M1NoPrecomp<E>,
    point_sets: Vec<Vec<E::ScalarField>>,
    vanishing_polys: Vec<DensePolynomial<E::ScalarField>>,
    g2_zeros: Vec<E::G2>,
    lagrange_ctxs: Vec<LagrangeInterpContext<E::ScalarField>>,
}

impl<E: Pairing> M1Precomp<E> {
    pub fn from_inner(
        inner: super::M1NoPrecomp<E>,
        point_sets: Vec<Vec<<E as Pairing>::ScalarField>>,
    ) -> Result<Self, Error> {
        let vanishing_polys: Vec<_> = cfg_iter!(point_sets)
            .map(|(_, ps)| vanishing_polynomial(ps))
            .collect();
        let g2_zeros = cfg_iter!(vanishing_polys)
            .map(|(_, p)| crate::curve_msm::<E::G2>(&inner.powers_of_g2, p))
            .collect::<Result<Vec<_>, Error>>()?;
        let lagrange_ctxs = cfg_iter!(point_sets)
            .map(|(_, ps)| LagrangeInterpContext::new_from_points(ps.as_ref()))
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(M1Precomp {
            inner,
            point_sets,
            vanishing_polys,
            g2_zeros,
            lagrange_ctxs,
        })
    }
}

impl<E: Pairing> Committer<E> for M1Precomp<E> {
    fn commit(
        &self,
        poly: impl AsRef<[<E as Pairing>::ScalarField]>,
    ) -> Result<Commitment<E>, Error> {
        self.inner.commit(poly)
    }
}

impl<E: Pairing> PolyMultiProof<E> for M1Precomp<E> {
    type Proof = Proof<E>;

    fn open(
        &self,
        transcript: &mut Transcript,
        evals: &[impl AsRef<[<E as Pairing>::ScalarField]>],
        polys: &[impl AsRef<[<E as Pairing>::ScalarField]>],
        point_set_index: usize,
    ) -> Result<Self::Proof, Error> {
        self.inner.open_with_vanishing_poly(
            transcript,
            evals,
            polys,
            &self.point_sets[point_set_index],
            &self.vanishing_polys[point_set_index],
        )
    }

    fn verify(
        &self,
        transcript: &mut Transcript,
        commits: &[Commitment<E>],
        point_set_index: usize,
        evals: &[impl AsRef<[<E as Pairing>::ScalarField]>],
        proof: &Self::Proof,
    ) -> Result<bool, Error> {
        self.inner.verify_with_lag_ctx_g2_zeros(
            transcript,
            commits,
            &self.point_sets[point_set_index],
            evals,
            proof,
            &self.lagrange_ctxs[point_set_index],
            &self.g2_zeros[point_set_index],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::M1Precomp;
    use crate::{
        method1::M1NoPrecomp,
        test_rng,
        traits::{Committer, PolyMultiProof},
    };
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
    use ark_std::UniformRand;
    use merlin::Transcript;

    #[test]
    fn test_basic_open_works() {
        let points = (0..30)
            .map(|_| Fr::rand(&mut test_rng()))
            .collect::<Vec<_>>();
        let s = M1NoPrecomp::<Bls12_381>::new(256, 32, &mut test_rng());
        let s = M1Precomp::from_inner(s, vec![points.clone()]).expect("Failed to construct");
        let polys = (0..20)
            .map(|_| DensePolynomial::<Fr>::rand(50, &mut test_rng()))
            .collect::<Vec<_>>();
        let evals: Vec<Vec<_>> = polys
            .iter()
            .map(|p| points.iter().map(|x| p.evaluate(x)).collect())
            .collect();
        let coeffs = polys.iter().map(|p| p.coeffs.clone()).collect::<Vec<_>>();
        let commits = coeffs
            .iter()
            .map(|p| s.commit(p).expect("Commit failed"))
            .collect::<Vec<_>>();
        let mut transcript = Transcript::new(b"testing");
        let open = s
            .open(&mut transcript, &evals, &coeffs, 0)
            .expect("Open failed");
        let mut transcript = Transcript::new(b"testing");
        assert_eq!(
            Ok(true),
            s.verify(&mut transcript, &commits, 0, &evals, &open)
        );
    }
}
