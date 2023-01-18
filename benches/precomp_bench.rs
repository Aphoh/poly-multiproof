use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::PrimeField;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use merlin::Transcript;
use poly_multiproof::{
    method1, method1::precompute as m1_precomp, method2, method2::precompute as m2_precomp,
};
use rand::thread_rng;

const MAX_LOG_SIZE: u32 = 10;
const MAX_SIZE: usize = 2usize.pow(MAX_LOG_SIZE);
const STEP_SIZE: usize = MAX_SIZE / 8;

fn verify_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("verify");
    let m1 = method1::Setup::<Bls12_381>::new(MAX_SIZE, MAX_SIZE, &mut thread_rng());
    let m2: method2::Setup<Bls12_381> = m1.clone().try_into().unwrap();
    let grid = TestGrid::<Fr>::gen_grid(MAX_LOG_SIZE);
    let commits = grid
        .coeffs
        .iter()
        .map(|c| m1.commit(c).unwrap())
        .collect::<Vec<_>>();

    for n_pts in (1..MAX_SIZE).step_by(STEP_SIZE) {
        let subg_pts = grid.trim_pts(n_pts);
        let m1_pc = m1_precomp::Setup::new(m1.clone(), vec![subg_pts.points.clone()])
            .expect("Failed to construct m1_pc");
        let m2_pc = m2_precomp::Setup::new(m2.clone(), vec![subg_pts.points.clone()])
            .expect("Failed to construct m1_pc");
        for n_poly in (1..MAX_SIZE).step_by(STEP_SIZE) {
            let subgrid = subg_pts.trim_poly(n_poly);
            let subcommits = &commits[..n_poly];
            // Method 1 with precomputes
            {
                let open = m1_pc
                    .open(
                        &mut Transcript::new(b"bench"),
                        &subgrid.evals,
                        &subgrid.coeffs,
                        0,
                    )
                    .unwrap();
                group.bench_with_input(
                    BenchmarkId::new(format!("m1_pc_{}", n_pts), n_poly),
                    &n_poly,
                    |b, _i| {
                        b.iter(|| {
                            let mut transcript = Transcript::new(b"bench");
                            assert!(m1_pc
                                .verify(&mut transcript, &subcommits, 0, &subgrid.evals, &open,)
                                .unwrap());
                        })
                    },
                );
            }
            // Method 2 with precomputes
            {
                let open = m2_pc
                    .open(
                        &mut Transcript::new(b"bench"),
                        &subgrid.evals,
                        &subgrid.coeffs,
                        0,
                    )
                    .unwrap();
                group.bench_with_input(
                    BenchmarkId::new(format!("m2_pc_{}", n_pts), n_poly),
                    &n_poly,
                    |b, _i| {
                        b.iter(|| {
                            let mut transcript = Transcript::new(b"bench");
                            assert!(m2_pc
                                .verify(&mut transcript, &subcommits, 0, &subgrid.evals, &open,)
                                .unwrap());
                        })
                    },
                );
            }
        }
    }
}

struct TestGrid<F: Clone> {
    coeffs: Vec<Vec<F>>,
    evals: Vec<Vec<F>>,
    points: Vec<F>,
}

impl<F: PrimeField> TestGrid<F> {
    fn gen_grid(max_log: u32) -> Self {
        let size = 2usize.pow(max_log);
        let degree = size - 1;
        let ev = Radix2EvaluationDomain::<F>::new(size).unwrap();
        let points = ev.elements().into_iter().collect::<Vec<_>>();
        assert_eq!(points.len(), size);
        let coeffs = (0..size)
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

criterion_group!(benches, verify_benchmark);
criterion_main!(benches);
