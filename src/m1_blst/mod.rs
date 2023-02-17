use crate::{
    gen_curve_powers_proj,
    lagrange::LagrangeInterpContext,
    traits::{Committer, PolyMultiProofNoPrecomp},
};
use ark_poly::univariate::DensePolynomial;
use ark_std::UniformRand;
use blst::{p1_affines, p2_affines};
use merlin::Transcript;
use std::usize;

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_std::rand::RngCore;

use crate::{get_challenge, get_field_size, transcribe_points_and_evals, Commitment};

use super::{gen_powers, linear_combination, poly_div_q_r, vanishing_polynomial, Error};

pub use ark_bls12_381::{
    Bls12_381, Fr, G1Affine, G1Projective as G1, G2Affine, G2Projective as G2,
};

mod fast_msm;
pub mod precompute;

pub struct M1NoPrecomp {
    pub powers_of_g1: Vec<G1>,
    pub powers_of_g2: Vec<G2>,
    prepped_g1s: p1_affines,
    prepped_g2s: p2_affines,
}

impl Clone for M1NoPrecomp {
    fn clone(&self) -> Self {
        Self::new_from_powers(self.powers_of_g1.clone(), self.powers_of_g2.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Proof(G1Affine);

impl M1NoPrecomp {
    pub fn new(max_coeffs: usize, max_pts: usize, rng: &mut impl RngCore) -> Self {
        let x = Fr::rand(rng);
        let g1 = G1::rand(rng);
        let g2 = G2::rand(rng);
        Self::new_from_scalar(x, g1, g2, max_coeffs, max_pts)
    }

    pub fn new_from_scalar(x: Fr, g1: G1, g2: G2, max_coeffs: usize, max_pts: usize) -> Self {
        let n_g2_powers = max_pts + 1;
        let x_powers = gen_powers(x, std::cmp::max(max_coeffs, n_g2_powers));

        let powers_of_g1 = gen_curve_powers_proj::<G1>(x_powers.as_ref(), g1);
        let powers_of_g2 = gen_curve_powers_proj::<G2>(x_powers[..n_g2_powers].as_ref(), g2);

        Self::new_from_powers(powers_of_g1, powers_of_g2)
    }

    pub fn new_from_affine(g1s: &[G1Affine], g2s: &[G2Affine]) -> Self {
        Self::new_from_powers(
            g1s.iter().map(|i| i.into_group()).collect::<Vec<_>>(),
            g2s.iter().map(|i| i.into_group()).collect::<Vec<_>>(),
        )
    }

    pub fn new_from_powers(powers_of_g1: Vec<G1>, powers_of_g2: Vec<G2>) -> Self {
        let prepped_g1s = fast_msm::prep_g1s(&powers_of_g1);
        let prepped_g2s = fast_msm::prep_g2s(&powers_of_g2);
        Self {
            powers_of_g1,
            powers_of_g2,
            prepped_g1s,
            prepped_g2s,
        }
    }

    fn open_with_vanishing_poly(
        &self,
        transcript: &mut Transcript,
        evals: &[impl AsRef<[Fr]>],
        polys: &[impl AsRef<[Fr]>],
        points: &[Fr],
        vp: &DensePolynomial<Fr>,
    ) -> Result<Proof, Error> {
        // Commit the evals and the points to the transcript
        let field_size_bytes = get_field_size::<Fr>();
        transcribe_points_and_evals(transcript, points, evals, field_size_bytes)?;

        // Read the challenge
        let gamma = get_challenge::<Fr>(transcript, b"open gamma", field_size_bytes);
        // Make the gamma powers
        let gammas = gen_powers::<Fr>(gamma, self.powers_of_g1.len());
        // Take a linear combo of gammas with the polynomials
        let fsum = linear_combination::<Fr>(polys, &gammas).ok_or(Error::NoPolynomialsGiven)?;

        // Polynomial divide, the remained would contain the gamma * ri_s,
        // The result is the correct quotient
        let (q, _) = poly_div_q_r(DensePolynomial { coeffs: fsum }.into(), vp.into())?;
        // Open to the resulting polynomial
        Ok(Proof(
            fast_msm::g1_msm(&self.prepped_g1s, &q, self.powers_of_g1.len())?.into_affine(),
        ))
    }

    fn verify_with_lag_ctx_g2_zeros(
        &self,
        transcript: &mut Transcript,
        commits: &[Commitment<Bls12_381>],
        points: &[Fr],
        evals: &[impl AsRef<[Fr]>],
        proof: &Proof,
        lag_ctx: &LagrangeInterpContext<Fr>,
        g2_zeros: &G2,
    ) -> Result<bool, Error> {
        let field_size_bytes = get_field_size::<Fr>();
        transcribe_points_and_evals(transcript, points, evals, field_size_bytes)?;
        let gamma = get_challenge(transcript, b"open gamma", field_size_bytes);
        // Aggregate the r_is and then do a single msm of just the ri's and gammas
        let gammas = gen_powers(gamma, evals.len());

        // Get the gamma^i r_i polynomials with lagrange interp. This does both the lagrange interp
        // and the gamma mul in one step so we can just lagrange interp once.
        let gamma_ris = lag_ctx.lagrange_interp_linear_combo(evals, &gammas)?.coeffs;
        let gamma_ris_pt =
            fast_msm::g1_msm(&self.prepped_g1s, &gamma_ris, self.powers_of_g1.len())?;

        // Then do a single msm of the gammas and commitments
        let cms = commits.iter().map(|i| i.0.into_group()).collect::<Vec<_>>();
        let cms_prep = fast_msm::prep_g1s(&cms.as_slice());
        let gamma_cm_pt = fast_msm::g1_msm(&cms_prep, &gammas, cms.len())?;

        let g2 = self.powers_of_g2[0];

        Ok(Bls12_381::pairing(gamma_cm_pt - gamma_ris_pt, g2)
            == Bls12_381::pairing(proof.0, g2_zeros))
    }
}

impl Committer<Bls12_381> for M1NoPrecomp {
    fn commit(&self, poly: impl AsRef<[Fr]>) -> Result<Commitment<Bls12_381>, Error> {
        let res = fast_msm::g1_msm(&self.prepped_g1s, poly.as_ref(), self.powers_of_g1.len())?;
        Ok(Commitment(res.into_affine()))
    }
}

impl PolyMultiProofNoPrecomp<Bls12_381> for M1NoPrecomp {
    type Proof = Proof;

    fn open(
        &self,
        transcript: &mut Transcript,
        evals: &[impl AsRef<[Fr]>],
        polys: &[impl AsRef<[Fr]>],
        points: &[Fr],
    ) -> Result<Proof, Error> {
        let vp = vanishing_polynomial(points.as_ref());
        self.open_with_vanishing_poly(transcript, evals, polys, points, &vp)
    }

    fn verify(
        &self,
        transcript: &mut Transcript,
        commits: &[Commitment<Bls12_381>],
        points: &[Fr],
        evals: &[impl AsRef<[Fr]>],
        proof: &Proof,
    ) -> Result<bool, Error> {
        let vp = vanishing_polynomial(points);
        let g2_zeros = fast_msm::g2_msm(&self.prepped_g2s, &vp, self.powers_of_g2.len())?;
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
    use ark_bls12_381::Fr;
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
    use ark_std::UniformRand;
    use merlin::Transcript;

    #[test]
    fn test_basic_open_works() {
        let s = M1NoPrecomp::new(256, 32, &mut test_rng());
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

    #[test]
    fn test_single_row_works() {
        let s = M1NoPrecomp::new(256, 32, &mut test_rng());
        let points = (0..30)
            .map(|_| Fr::rand(&mut test_rng()))
            .collect::<Vec<_>>();
        let polys = vec![DensePolynomial::<Fr>::rand(50, &mut test_rng())];
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
