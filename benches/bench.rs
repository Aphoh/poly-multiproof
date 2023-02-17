use ark_bls12_381::{Bls12_381, Fr};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};
use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use merlin::Transcript;
use poly_multiproof::{
    method1,
    method1::precompute as m1_precomp,
    method2,
    method2::precompute as m2_precomp,
    traits::{Committer, PolyMultiProof},
    Commitment,
};
use rand::thread_rng;

type M1 = method1::M1NoPrecomp<Bls12_381>;
type M1Pc = method1::precompute::M1Precomp<Bls12_381>;
type M2 = method2::M2NoPrecomp<Bls12_381>;
type M2Pc = method2::precompute::M2Precomp<Bls12_381>;
#[cfg(feature = "blst")]
type M1Blst = poly_multiproof::m1_blst::M1NoPrecomp;
#[cfg(feature = "blst")]
type M1BlstPc = poly_multiproof::m1_blst::precompute::M1Precomp;

const WIDTH: usize = 4096;
const HEIGHT: usize = 256;
// This won't let the block width go above 128
const WIDTH_STEP: usize = 128 / 8;
const HEIGHT_STEP: usize = HEIGHT / 32;

fn run_verify<'a, E: Pairing, P: PolyMultiProof<E>, M: Measurement>(
    pmp: &P,
    grid: &TestGrid<E::ScalarField>,
    group: &mut BenchmarkGroup<'a, M>,
    commits: &[Commitment<E>],
    id_str: String,
    n_poly: usize,
) {
    let open = pmp
        .open(&mut Transcript::new(b"bench"), &grid.evals, &grid.coeffs, 0)
        .unwrap();
    group.bench_with_input(BenchmarkId::new(id_str, n_poly), &n_poly, |b, _i| {
        b.iter(|| {
            let mut transcript = Transcript::new(b"bench");
            assert!(pmp
                .verify(&mut transcript, &commits[..n_poly], 0, &grid.evals, &open)
                .unwrap());
        })
    });
}

fn run_open<'a, E: Pairing, P: PolyMultiProof<E>, M: Measurement>(
    pmp: &P,
    grid: &TestGrid<E::ScalarField>,
    group: &mut BenchmarkGroup<'a, M>,
    id_str: String,
    n_poly: usize,
) {
    group.bench_with_input(BenchmarkId::new(id_str, n_poly), &n_poly, |b, _i| {
        b.iter(|| {
            pmp.open(&mut Transcript::new(b"bench"), &grid.evals, &grid.coeffs, 0)
                .unwrap();
        })
    });
}

fn verify_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("verify");
    let m1 = M1::new(WIDTH, WIDTH.into(), &mut thread_rng());
    #[cfg(feature = "blst")]
    let m1_blst = M1Blst::new_from_affine(&m1.powers_of_g1, &m1.powers_of_g2);
    let m2 = M2::new_from_affine(
        m1.powers_of_g1.clone(),
        m1.powers_of_g2[0],
        m1.powers_of_g2[1],
    );
    let grid = TestGrid::<Fr>::gen_grid(WIDTH, HEIGHT);
    let commits = grid
        .coeffs
        .iter()
        .map(|c| m1.commit(c).unwrap())
        .collect::<Vec<_>>();

    for n_pts in (1..WIDTH).step_by(WIDTH_STEP).filter(|i| i < &128) {
        let subg_pts = grid.trim_pts(n_pts);
        let m1_pc = M1Pc::from_inner(m1.clone(), vec![subg_pts.points.clone()])
            .expect("Failed to construct m1_pc");
        #[cfg(feature = "blst")]
        let m1_blst_pc = M1BlstPc::from_inner(m1_blst.clone(), vec![subg_pts.points.clone()])
            .expect("Failed to construct m1_blst_pc");
        let m2_pc = M2Pc::from_inner(m2.clone(), vec![subg_pts.points.clone()])
            .expect("Failed to construct m1_pc");
        for n_poly in (1..HEIGHT).step_by(HEIGHT_STEP) {
            let subgrid = subg_pts.trim_poly(n_poly);
            // Method 1 with precomputes
            run_verify(
                &m1_pc,
                &subgrid,
                &mut group,
                &commits,
                format!("m1_pc_{}", n_pts),
                n_poly,
            );
            #[cfg(feature = "blst")]
            run_verify(
                &m1_blst_pc,
                &subgrid,
                &mut group,
                &commits,
                format!("m1blst_pc_{}", n_pts),
                n_poly,
            );
            run_verify(
                &m2_pc,
                &subgrid,
                &mut group,
                &commits,
                format!("m2_pc_{}", n_pts),
                n_poly,
            );
        }
    }
}

fn open_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("open");
    let m1 = M1::new(WIDTH, WIDTH.into(), &mut thread_rng());
    #[cfg(feature = "blst")]
    let m1_blst = M1Blst::new_from_affine(&m1.powers_of_g1, &m1.powers_of_g2);
    let m2 = M2::new_from_affine(
        m1.powers_of_g1.clone(),
        m1.powers_of_g2[0],
        m1.powers_of_g2[1],
    );
    let grid = TestGrid::<Fr>::gen_grid(WIDTH, HEIGHT);

    for n_pts in (1..WIDTH).step_by(WIDTH_STEP).filter(|i| i < &128) {
        let subg_pts = grid.trim_pts(n_pts);
        let m1_pc = m1_precomp::M1Precomp::from_inner(m1.clone(), vec![subg_pts.points.clone()])
            .expect("Failed to construct m1_pc");
        #[cfg(feature = "blst")]
        let m1_blst_pc = M1BlstPc::from_inner(m1_blst.clone(), vec![subg_pts.points.clone()])
            .expect("Failed to construct m1_blst_pc");
        let m2_pc = m2_precomp::M2Precomp::from_inner(m2.clone(), vec![subg_pts.points.clone()])
            .expect("Failed to construct m1_pc");
        for n_poly in (1..HEIGHT).step_by(HEIGHT_STEP) {
            let subgrid = subg_pts.trim_poly(n_poly);
            // Method 1 with precomputes
            run_open(
                &m1_pc,
                &subgrid,
                &mut group,
                format!("m1_pc_{}", n_pts),
                n_poly,
            );
            #[cfg(feature = "blst")]
            run_open(
                &m1_blst_pc,
                &subgrid,
                &mut group,
                format!("m1blst_pc_{}", n_pts),
                n_poly,
            );
            run_open(
                &m2_pc,
                &subgrid,
                &mut group,
                format!("m2_pc_{}", n_pts),
                n_poly,
            );
        }
    }
}

struct TestGrid<F: Clone> {
    coeffs: Vec<Vec<F>>,
    evals: Vec<Vec<F>>,
    points: Vec<F>,
}

impl<F: PrimeField> TestGrid<F> {
    fn gen_grid(width: usize, height: usize) -> Self {
        let degree = width - 1;
        let ev = Radix2EvaluationDomain::<F>::new(width).unwrap();
        let points = ev.elements().into_iter().collect::<Vec<_>>();
        assert_eq!(points.len(), width);
        let coeffs = (0..height)
            .map(|_| DensePolynomial::<F>::rand(degree, &mut thread_rng()).coeffs)
            .collect::<Vec<_>>();
        let evals: Vec<Vec<_>> = coeffs.iter().map(|p| ev.fft(&p)).collect();
        Self {
            coeffs,
            points,
            evals,
        }
    }
    fn trim_pts(&self, n_pts: usize) -> Self {
        Self {
            coeffs: self.coeffs.clone(),
            evals: self.evals.iter().map(|ev| ev[..n_pts].to_vec()).collect(),
            points: self.points[..n_pts].to_vec(),
        }
    }
    fn trim_poly(&self, n_poly: usize) -> Self {
        Self {
            coeffs: self.coeffs[..n_poly].to_vec(),
            evals: self.evals[..n_poly].to_vec(),
            points: self.points.clone(),
        }
    }
}

criterion_group!(benches, open_benchmark, verify_benchmark);
criterion_main!(benches);
