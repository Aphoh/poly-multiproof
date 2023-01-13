use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::PrimeField;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kzg_multiproof::{method1, method2, MultiOpenKzg};
use merlin::Transcript;
use rand::thread_rng;

const MAX_LOG_SIZE: u32 = 8;
const MAX_SIZE: usize = 2usize.pow(MAX_LOG_SIZE);

fn open_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("open");
    let m1 = method1::Setup::<Bls12_381>::new(MAX_SIZE, MAX_SIZE, &mut thread_rng());
    let m2: method2::Setup<Bls12_381> = m1.clone().try_into().unwrap();
    let grid = TestGrid::<Fr>::gen_grid(MAX_LOG_SIZE);
    for n_pts in powers_of_2(MAX_LOG_SIZE - 1) {
        for n_poly in powers_of_2(MAX_LOG_SIZE - 1) {
            let subgrid = grid.trim(n_poly, n_pts);
            group.bench_with_input(
                BenchmarkId::new(format!("m1_{}", n_pts), n_poly),
                &n_poly,
                |b, _i| {
                    b.iter(|| {
                        let mut transcript = Transcript::new(b"bench");
                        m1.open(
                            &mut transcript,
                            &subgrid.evals,
                            &subgrid.coeffs,
                            &subgrid.points,
                        )
                        .unwrap();
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("m2_{}", n_pts), n_poly),
                &n_poly,
                |b, _i| {
                    b.iter(|| {
                        let mut transcript = Transcript::new(b"bench");
                        m2.open(
                            &mut transcript,
                            &subgrid.evals,
                            &subgrid.coeffs,
                            &subgrid.points,
                        )
                        .unwrap();
                    })
                },
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
    fn gen_grid(max_log: u32) -> Self {
        let size = 2usize.pow(max_log);
        let ev = Radix2EvaluationDomain::<F>::new(size).unwrap();
        let points = ev.elements().into_iter().collect::<Vec<_>>();
        assert_eq!(points.len(), size);
        let coeffs = (0..size)
            .map(|_| DensePolynomial::<F>::rand(50, &mut thread_rng()).coeffs)
            .collect::<Vec<_>>();
        let evals: Vec<Vec<_>> = coeffs.iter().map(|p| ev.fft(&p)).collect();
        Self {
            coeffs,
            points,
            evals,
        }
    }
    fn trim(&self, n_poly: usize, n_pts: usize) -> Self {
        Self {
            coeffs: self.coeffs[..n_poly].to_vec(),
            evals: self.coeffs[..n_poly]
                .iter()
                .map(|ev| ev[..n_pts].to_vec())
                .collect(),
            points: self.points[..n_pts].to_vec(),
        }
    }
}

fn powers_of_2(max_log: u32) -> Vec<usize> {
    (0..=max_log).map(|i| 2usize.pow(i)).collect()
}

criterion_group!(benches, open_benchmark);
criterion_main!(benches);
