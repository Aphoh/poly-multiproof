use crate::{
    lagrange::LagrangeInterpContext,
    traits::{Committer, PolyMultiProofNoPrecomp},
};
use ark_poly::univariate::DensePolynomial;
use ark_std::UniformRand;
use merlin::Transcript;
use std::usize;

use ark_ec::{pairing::Pairing, CurveGroup};
use ark_std::rand::RngCore;

use crate::{get_challenge, get_field_size, transcribe_points_and_evals, Commitment};

use super::{
    gen_curve_powers, gen_powers, linear_combination, poly_div_q_r, vanishing_polynomial, Error,
};

pub mod precompute;

#[derive(Clone, Debug)]
pub struct M1NoPrecomp<E: Pairing> {
    pub powers_of_g1: Vec<E::G1Affine>,
    pub powers_of_g2: Vec<E::G2Affine>,
}

#[derive(Debug, Clone)]
pub struct Proof<E: Pairing>(E::G1Affine);

impl<E: Pairing> M1NoPrecomp<E> {
    pub fn new(max_coeffs: usize, max_pts: usize, rng: &mut impl RngCore) -> Self {
        let x = E::ScalarField::rand(rng);
        let g1 = E::G1::rand(rng);
        let g2 = E::G2::rand(rng);
        Self::new_from_scalar(x, g1, g2, max_coeffs, max_pts)
    }

    pub fn new_from_scalar(
        x: E::ScalarField,
        g1: E::G1,
        g2: E::G2,
        max_coeffs: usize,
        max_pts: usize,
    ) -> Self {
        let n_g2_powers = max_pts + 1;
        let x_powers = gen_powers(x, std::cmp::max(max_coeffs, n_g2_powers));

        let powers_of_g1 = gen_curve_powers::<E::G1>(x_powers.as_ref(), g1);
        let powers_of_g2 = gen_curve_powers::<E::G2>(x_powers[..n_g2_powers].as_ref(), g2);

        Self::new_from_affine(powers_of_g1, powers_of_g2)
    }

    pub fn new_from_powers(powers_of_g1: &[E::G1], powers_of_g2: &[E::G2]) -> Self {
        Self {
            powers_of_g1: powers_of_g1.iter().map(|s| s.into_affine()).collect(),
            powers_of_g2: powers_of_g2.iter().map(|s| s.into_affine()).collect(),
        }
    }

    pub fn new_from_affine(powers_of_g1: Vec<E::G1Affine>, powers_of_g2: Vec<E::G2Affine>) -> Self {
        Self {
            powers_of_g1,
            powers_of_g2,
        }
    }

    fn open_with_vanishing_poly(
        &self,
        transcript: &mut Transcript,
        evals: &[impl AsRef<[E::ScalarField]>],
        polys: &[impl AsRef<[E::ScalarField]>],
        points: &[E::ScalarField],
        vp: &DensePolynomial<E::ScalarField>,
    ) -> Result<Proof<E>, Error> {
        // Commit the evals and the points to the transcript
        let field_size_bytes = get_field_size::<E::ScalarField>();
        transcribe_points_and_evals(transcript, points, evals, field_size_bytes)?;

        // Read the challenge
        let gamma = get_challenge::<E::ScalarField>(transcript, b"open gamma", field_size_bytes);
        // Make the gamma powers
        let gammas = gen_powers::<E::ScalarField>(gamma, self.powers_of_g1.len());
        // Take a linear combo of gammas with the polynomials
        let fsum = linear_combination::<E::ScalarField>(polys, &gammas)
            .ok_or(Error::NoPolynomialsGiven)?;

        // Polynomial divide, the remained would contain the gamma * ri_s,
        // The result is the correct quotient
        let (q, _) = poly_div_q_r(DensePolynomial { coeffs: fsum }.into(), vp.into())?;
        // Open to the resulting polynomial
        Ok(Proof(
            super::curve_msm::<E::G1>(&self.powers_of_g1, &q)?.into_affine(),
        ))
    }

    fn verify_with_lag_ctx_g2_zeros(
        &self,
        transcript: &mut Transcript,
        commits: &[Commitment<E>],
        points: &[E::ScalarField],
        evals: &[impl AsRef<[E::ScalarField]>],
        proof: &Proof<E>,
        lag_ctx: &LagrangeInterpContext<E::ScalarField>,
        g2_zeros: &E::G2,
    ) -> Result<bool, Error> {
        let field_size_bytes = get_field_size::<E::ScalarField>();
        transcribe_points_and_evals(transcript, points, evals, field_size_bytes)?;
        let gamma = get_challenge(transcript, b"open gamma", field_size_bytes);
        // Aggregate the r_is and then do a single msm of just the ri's and gammas
        let gammas = gen_powers(gamma, evals.len());

        // Get the gamma^i r_i polynomials with lagrange interp. This does both the lagrange interp
        // and the gamma mul in one step so we can just lagrange interp once.
        let gamma_ris = lag_ctx.lagrange_interp_linear_combo(evals, &gammas)?.coeffs;
        let gamma_ris_pt = super::curve_msm::<E::G1>(&self.powers_of_g1, gamma_ris.as_ref())?;

        // Then do a single msm of the gammas and commitments
        let cms = commits.iter().map(|i| i.0).collect::<Vec<_>>();
        let gamma_cm_pt = super::curve_msm::<E::G1>(&cms, gammas.as_ref())?;

        let g2 = self.powers_of_g2[0];

        Ok(E::pairing(gamma_cm_pt - gamma_ris_pt, g2) == E::pairing(proof.0, g2_zeros))
    }
}

impl<E: Pairing> Committer<E> for M1NoPrecomp<E> {
    fn commit(&self, poly: impl AsRef<[E::ScalarField]>) -> Result<Commitment<E>, Error> {
        let res = super::curve_msm::<E::G1>(&self.powers_of_g1, poly.as_ref())?;
        Ok(Commitment(res.into_affine()))
    }
}

impl<E: Pairing> PolyMultiProofNoPrecomp<E> for M1NoPrecomp<E> {
    type Proof = Proof<E>;

    fn open(
        &self,
        transcript: &mut Transcript,
        evals: &[impl AsRef<[E::ScalarField]>],
        polys: &[impl AsRef<[E::ScalarField]>],
        points: &[E::ScalarField],
    ) -> Result<Proof<E>, Error> {
        let vp = vanishing_polynomial(points.as_ref());
        self.open_with_vanishing_poly(transcript, evals, polys, points, &vp)
    }

    fn verify(
        &self,
        transcript: &mut Transcript,
        commits: &[Commitment<E>],
        points: &[E::ScalarField],
        evals: &[impl AsRef<[E::ScalarField]>],
        proof: &Proof<E>,
    ) -> Result<bool, Error> {
        let vp = vanishing_polynomial(points);
        let g2_zeros = super::curve_msm::<E::G2>(&self.powers_of_g2, &vp)?;
        let lag_ctx = LagrangeInterpContext::new_from_points(points)?;
        self.verify_with_lag_ctx_g2_zeros(
            transcript, commits, points, evals, proof, &lag_ctx, &g2_zeros,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::M1NoPrecomp;
    use crate::{
        test_rng,
        traits::{Committer, PolyMultiProofNoPrecomp},
    };
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
    use ark_std::UniformRand;
    use merlin::Transcript;

    #[test]
    fn test_basic_open_works() {
        let s = M1NoPrecomp::<Bls12_381>::new(256, 30, &mut test_rng());
        let points = (0..30)
            .map(|_| Fr::rand(&mut test_rng()))
            .collect::<Vec<_>>();
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
            .open(&mut transcript, &evals, &coeffs, &points)
            .expect("Open failed");
        let mut transcript = Transcript::new(b"testing");
        assert_eq!(
            Ok(true),
            s.verify(&mut transcript, &commits, &points, &evals, &open)
        );
    }
}
