use ark_ec::pairing::Pairing;
use merlin::Transcript;

use crate::{Commitment, Error};

pub trait Committer<E: Pairing> {
    fn commit(&self, poly: impl AsRef<[E::ScalarField]>) -> Result<Commitment<E>, Error>;
}

pub trait PolyMultiProof<E: Pairing>: Sized {
    type Proof: Clone;

    fn open(
        &self,
        transcript: &mut Transcript,
        evals: &[impl AsRef<[E::ScalarField]>],
        polys: &[impl AsRef<[E::ScalarField]>],
        point_set_index: usize,
    ) -> Result<Self::Proof, Error>;

    fn verify(
        &self,
        transcript: &mut Transcript,
        commits: &[Commitment<E>],
        point_set_index: usize,
        evals: &[impl AsRef<[E::ScalarField]>],
        proof: &Self::Proof,
    ) -> Result<bool, Error>;
}

pub trait PolyMultiProofNoPrecomp<E: Pairing>: Sized {
    type Proof: Clone;

    fn open(
        &self,
        transcript: &mut Transcript,
        evals: &[impl AsRef<[E::ScalarField]>],
        polys: &[impl AsRef<[E::ScalarField]>],
        points: &[E::ScalarField],
    ) -> Result<Self::Proof, Error>;

    fn verify(
        &self,
        transcript: &mut Transcript,
        commits: &[Commitment<E>],
        points: &[E::ScalarField],
        evals: &[impl AsRef<[E::ScalarField]>],
        proof: &Self::Proof,
    ) -> Result<bool, Error>;
}
