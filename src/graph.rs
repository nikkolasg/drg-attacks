use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::HashSet;
use std::hash::Hash;

use std::fmt;

// Graph holds the parameters and the edges of the graph. This is a special type
// of graph that has a *proper* labelling: for each edge (i,j), we have i < j.
#[derive(Debug)]
pub struct Graph {
    // parents holds all parents of all nodes. If j = parents[i][u], then there
    // is an edge (j -> i), i.e. j is the parent of i.
    // The capacity of the graph is the size of the vector - Some nodes may be
    // absent when counted, i.e. node i may not have any parent and may not be
    // the parent of any other node. In that case, it is not included in the graph G
    parents: Vec<Vec<usize>>,
    seed: [u8; 32],
    algo: DRGAlgo,
}

// DRGAlgo represents which algorithm can be used to create the edges so a Graph is
// a Depth Robust Graph
#[derive(Debug, Copy, Clone)]
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
        let mut g = Graph {
            algo,
            seed,
            parents: Vec::with_capacity(size),
        };
        match g.algo {
            DRGAlgo::BucketSample => g.bucket_sample(),
            DRGAlgo::MetaBucket(degree) => g.meta_bucket(degree),
        }
        g
    }

    // depth returns the longest depth found in the graph
    fn depth(&self) -> usize {
        self.parents
            .iter()
            .fold(Vec::new(), |mut acc, parents| {
                // take the depth of each parents + 1 then take the max of it
                match parents.iter().map(|&p| acc[p] + 1).max() {
                    Some(depth) => acc.push(depth),
                    None => acc.push(0),
                };
                acc
            })
            .into_iter()
            .max()
            .unwrap()
    }

    // remove returns a new graph with the specified nodes removed
    // TODO inefficient at the moment; may be faster ways.
    fn remove(&self, nodes: Vec<usize>) -> Graph {
        let mut out = Vec::with_capacity(self.parents.len());
        for i in 0..self.parents.len() {
            let parents = self.parents.get(i).unwrap();
            let new_parents = if nodes.contains(&i) {
                // no parent for a deleted node
                Vec::new()
            } else {
                // only take parents which are not in the list of nodes
                parents
                    .into_iter()
                    .filter(|&parent| !nodes.contains(parent))
                    .map(|&p| p)
                    .collect::<Vec<usize>>()
            };
            out.push(new_parents);
        }

        Graph {
            parents: out,
            algo: self.algo,
            seed: self.seed,
        }
    }

    // Implementation of the first algorithm BucketSample on page 22 of the
    // porep paper : https://web.stanford.edu/~bfisch/porep_short.pdf
    // It produces a degree-2 graph which is asymptotically depth-robust.
    fn bucket_sample(&mut self) {
        let mut rng = self.rng();
        for node in 0..self.parents.len() {
            let mut parents = Vec::new();
            match node {
                // no parents for the first node
                0 => {}
                // second node only has the first node as parent
                1 => {
                    parents.push(0);
                }
                _ => {
                    // push the direct parent of i, i.e. (i-1 -> i)
                    parents.push(node - 1);

                    // choose a bucket index
                    let max_bucket = (node as f32).log2().floor() as usize;
                    let i: usize = rng.gen_range(1, max_bucket + 1);
                    // get a node from that bucket, i.e. from [2^i-1, 2^i[
                    // exclusif because otherwise a parent can be the same
                    // as its child
                    let min = 1 << (i - 1);
                    let max = 1 << i;
                    let random_parent = rng.gen_range(min, max);
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
        let mut rng = self.rng();
        let m = degree - 1;
        for node in 0..self.parents.len() {
            let mut parents = Vec::with_capacity(degree);
            match node {
                // no parents for the first node
                0 => {}
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
                        let i: usize = rng.gen_range(1, max_bucket + 1);
                        // choose parent in range [2^(i-1), 2^i[
                        let min = 1 << (i - 1);
                        let max = 1 << i;
                        let meta_parent = rng.gen_range(min, max);
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

    fn rng(&self) -> ChaCha20Rng {
        ChaCha20Rng::from_seed(self.seed)
    }

    pub fn parents(&self) -> &Vec<Vec<usize>> {
        &self.parents
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "G(cap:{}, drg: ", self.parents.len())?;
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
pub mod tests {

    use super::*;

    static TEST_SEED: [u8; 32] = [1; 32];

    #[test]
    fn graph_new() {
        let size = 100;
        let g1 = Graph::new(size, TEST_SEED, DRGAlgo::BucketSample);
        assert_eq!(g1.parents.len(), 0); // no nodes generated yet
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
            assert!(parents.iter().find(|x| **x == i - 1).is_some());
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
            assert!(parents.iter().find(|x| **x == i - 1).is_some());
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

    #[test]
    fn graph_remove() {
        // graph 1 - 5 nodes
        // 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4
        // 0 -> 2, 2 -> 4
        // Remove nodes 1 and 3
        // -> final graph 0 -> 2, 2 -> 4
        let p1 = vec![vec![], vec![0], vec![0, 1], vec![2], vec![2, 3]];
        let g1 = Graph {
            parents: p1,
            seed: TEST_SEED,
            algo: DRGAlgo::BucketSample,
        };

        let nodes = vec![1, 3];
        let g3 = g1.remove(nodes);
        let expected = vec![vec![], vec![], vec![0], vec![], vec![2]];
        assert_eq!(g3.parents.len(), 5);
        assert_eq!(g3.parents, expected);
    }

    #[test]
    fn graph_depth() {
        let p1 = vec![vec![], vec![0], vec![1], vec![2], vec![3]];
        assert_eq!(graph_from(p1).depth(), 4);

        let p2 = vec![vec![], vec![], vec![0], vec![2], vec![2, 3], vec![3]];
        assert_eq!(graph_from(p2).depth(), 3);
    }

    pub fn graph_from(parents: Vec<Vec<usize>>) -> Graph {
        Graph {
            parents: parents,
            seed: TEST_SEED,
            algo: DRGAlgo::BucketSample,
        }
    }
}
