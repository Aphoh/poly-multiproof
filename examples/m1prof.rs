use ark_bls12_381::{Bls12_381, Fr};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
use ark_std::UniformRand;
use merlin::Transcript;
use poly_multiproof::method1::{precompute, Setup};
use rand::thread_rng;

fn main() {
    let points = (0..250)
        .map(|_| Fr::rand(&mut thread_rng()))
        .collect::<Vec<_>>();
    let inner = Setup::new(256, 256, &mut thread_rng());
    let s = precompute::Setup::<Bls12_381>::new(inner, vec![points.clone()])
        .expect("Failed to construct");
    let polys = (0..20)
        .map(|_| DensePolynomial::<Fr>::rand(255, &mut thread_rng()))
        .collect::<Vec<_>>();
    let evals: Vec<Vec<_>> = polys
        .iter()
        .map(|p| points.iter().map(|x| p.evaluate(x)).collect())
        .collect();
    let coeffs = polys.iter().map(|p| p.coeffs.clone()).collect::<Vec<_>>();
    let commits = coeffs
        .iter()
        .map(|p| s.inner.commit(p).expect("Commit failed"))
        .collect::<Vec<_>>();
    let mut transcript = Transcript::new(b"testing");
    let open = s
        .open(&mut transcript, &evals, &coeffs, 0)
        .expect("Open failed");

    for _ in 0..1_000 {
        let mut transcript = Transcript::new(b"testing");
        assert_eq!(
            Ok(true),
            s.verify(&mut transcript, &commits, 0, &evals, &open)
        );
    }
}
