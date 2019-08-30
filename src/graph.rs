extern crate rand;
extern crate rand_chacha;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::HashSet;
use std::hash::Hash;

use std::fmt;

// Graph holds the parameters and the edges of the graph.
#[derive(Debug)]
pub struct Graph {
    // cardinality of the graph
    size: usize,
    // parents holds all parents of all nodes. If j = parents[i][u], then there
    // is an edge (j -> i+1), i.e. j is the parent of i. i+1 because the first
    // node 0 never have parents so the array is never filled for the first cell.
    parents: Vec<Vec<usize>>,
    rng: ChaCha20Rng,
    algo: DRGAlgo,
}

// DRGAlgo represents which algorithm can be used to create the edges so a Graph is
// a Depth Robust Graph
#[derive(Debug)]
pub enum DRGAlgo {
    // BucketSample is the regular bucket sampling algorithm with degree 2
    BucketSample,
    // MetaBucket is the meta-graph construction with a specified degree
    MetaBucket(usize),
}

impl Graph {
    // new returns a new graph instance from the given parameters.
    // The graph's edges are not generated yet, call fill_drg to compute the edges.
    pub fn new(size: usize, seed: [u8; 32], algo: DRGAlgo) -> Graph {
        let cha = ChaCha20Rng::from_seed(seed);
        let mut g = Graph {
            size,
            algo,
            parents: Vec::with_capacity(size),
            rng: cha,
        };
        match g.algo {
            DRGAlgo::BucketSample => g.bucket_sample(),
            DRGAlgo::MetaBucket(degree) => g.meta_bucket(degree),
        }
        g
    }

    // Implementation of the first algorithm BucketSample on page 22 of the
    // porep paper : https://web.stanford.edu/~bfisch/porep_short.pdf
    // It produces a degree-2 graph which is asymptotically depth-robust.
    fn bucket_sample(&mut self) {
        for node in 0..self.size {
            let mut parents = Vec::with_capacity(2);
            match node {
                // no parents for the first node
                0 => continue,
                // second node only has the first node as parent
                1 => {
                    parents.push(0);
                }
                _ => {
                    // push the direct parent of i, i.e. (i-1 -> i)
                    parents.push(node - 1);

                    // choose a bucket index
                    let max_bucket = (node as f32).log2().floor() as usize;
                    let i: usize = self.rng.gen_range(1, max_bucket + 1);
                    // get a node from that bucket, i.e. from [2^i-1, 2^i[
                    // exclusif because otherwise a parent can be the same
                    // as its child
                    let min = 1 << (i - 1);
                    let max = 1 << i;
                    let random_parent = self.rng.gen_range(min, max);
                    assert!(random_parent < node);
                    parents.push(random_parent);
                }
            }

            remove_duplicate(&mut parents);
            self.parents.push(parents);
        }
    }

    // Implementation of the meta-graph construction algorithm described in page 22
    // of the porep paper https://web.stanford.edu/~bfisch/porep_short.pdf
    // It produces a degree-d graph on average.
    fn meta_bucket(&mut self, degree: usize) {
        let m = degree - 1;
        for node in 0..self.size {
            let mut parents = Vec::with_capacity(degree);
            match node {
                // no parents for the first node
                0 => continue,
                // second node only has the first node as parent
                1 => {
                    parents.push(0);
                }
                _ => {
                    // push the direct parent of i, i.e. (i-1 -> i)
                    parents.push(node - 1);

                    // similar to bucket_sample but we select m parents instead
                    // of just one
                    for _ in 0..m {
                        // meta_idx represents a meta node in the meta graph
                        // each node is represented m times, so we always take the
                        // first node index to not fall on the same final index
                        let meta_idx = node * m;
                        let max_bucket = (meta_idx as f32).log2().floor() as usize;
                        // choose bucket index {1 ... log2(idx)}
                        let i: usize = self.rng.gen_range(1, max_bucket + 1);
                        // choose parent in range [2^(i-1), 2^i[
                        let min = 1 << (i - 1);
                        let max = 1 << i;
                        let meta_parent = self.rng.gen_range(min, max);
                        let real_parent = meta_parent / degree;
                        assert!(meta_parent < meta_idx);
                        assert!(real_parent < node);
                        parents.push(real_parent);
                    }
                }
            }
            // filtering duplicate parents
            remove_duplicate(&mut parents);
            self.parents.push(parents);
        }
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "G(size:{}, drg: ", self.size)?;
        match self.algo {
            DRGAlgo::BucketSample => write!(f, "bucket, ")?,
            DRGAlgo::MetaBucket(d) => write!(f, "meta-bucket (degree {}), ", d)?,
        }
        write!(f, "parents: {:?}", self.parents)
    }
}

fn remove_duplicate<T: Hash + Eq>(elements: &mut Vec<T>) {
    let set: HashSet<_> = elements.drain(..).collect();
    elements.extend(set.into_iter());
}

#[cfg(test)]
mod tests {

    use super::*;

    static TEST_SEED: [u8; 32] = [1; 32];

    #[test]
    fn graph_new() {
        let size = 100;
        let g1 = Graph::new(size, TEST_SEED, DRGAlgo::BucketSample);
        assert_eq!(g1.size, size);
        assert_eq!(g1.parents.capacity(), size);
    }

    #[test]
    fn graph_bucket_sample() {
        let g1 = Graph::new(10, TEST_SEED, DRGAlgo::BucketSample);
        g1.parents.iter().enumerate().for_each(|(i, parents)| {
            // test there's at least a parent
            assert!(parents.len() >= 1 && parents.len() <= 2);
            // test there's at least the direct parent
            // == i since first cell is for node 1
            assert!(parents.iter().find(|x| **x == i).is_some());
            // test the other parent is less
            if parents.len() == 2 {
                assert!(parents.iter().find(|x| **x < i).is_some());
            }
        });
    }

    #[test]
    fn graph_meta_sample() {
        let degree = 3;
        let g1 = Graph::new(10, TEST_SEED, DRGAlgo::MetaBucket(3));
        g1.parents.iter().enumerate().for_each(|(i, parents)| {
            // test there's at least a parent
            assert!(parents.len() >= 1 && parents.len() <= degree);
            // test there's at least the direct parent
            // == i since first cell is for node 1
            assert!(parents.iter().find(|x| **x == i).is_some());
            // test all the other parents are less
            if parents.len() > 1 {
                assert_eq!(
                    parents
                        .iter()
                        .filter(|x| **x < i)
                        .collect::<Vec<&usize>>()
                        .len(),
                    parents.len() - 1
                );
            }
        });
    }
}
