use std::collections::HashSet;
use std::iter::FromIterator;

use crate::graph::Graph;
use crate::utils;

/*pub enum DepthReduceSet {*/
//ValiantBasic(usize),
//}

//pub fn depth_reduce(g: &Graph, drs: DepthReduceSet) -> Graph {
//match drs {
//DepthReduceSet::ValiantBasic(target) => valiant_basic_depth(g, target),
//}
//}

// valiant_basic returns a set S such that depth(G - S) < target.
// It implements the algo 8 in the https://eprint.iacr.org/2018/944.pdf paper.
fn valiant_basic(g: &Graph, target: usize) -> HashSet<usize> {
    let partitions = valiant_partitions(g);
    // TODO replace by a simple bitset or boolean vec
    let mut chosen: Vec<usize> = Vec::new();
    let mut s = HashSet::new();
    let mut reduced = g.remove(&s);
    // returns the smallest next partition unchosen
    // mut is required because it changes chosen which is mut
    let mut find_next = || -> &HashSet<Edge> {
        match partitions
            .iter()
            .enumerate()
            // only take partitions with edges in it
            .filter(|&(_, values)| values.len() > 0)
            // only take the ones we didn't choose before
            .filter(|&(i, _)| !chosen.contains(&i))
            // take the smallest one
            .min_by_key(|&(_, values)| values.len())
        {
            Some((i, val)) => {
                chosen.push(i);
                val
            }
            None => panic!("no more partitions to use"),
        }
    };
    while reduced.depth() > target {
        let partition = find_next();
        // add the origin node for each edges in the chosen partition
        s.extend(partition.iter().fold(Vec::new(), |mut acc, edge| {
            acc.push(edge.0);
            acc
        }));
        reduced = reduced.remove(&s);
    }

    return s;
}

// Edge holds the origin and endpoint of an edge.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct Edge(usize, usize);
// FIXME: Move outside of this module.

// valiant_partitions returns the sets E_i and S_i from the given graph
// according to the definition algorithm 8 from
// https://eprint.iacr.org/2018/944.pdf .
fn valiant_partitions(g: &Graph) -> Vec<HashSet<Edge>> {
    let bs = utils::node_bitsize();
    let mut eis = Vec::with_capacity(bs);
    for _ in 0..bs {
        eis.push(HashSet::new());
    }

    for (v, parents) in g.parents().iter().enumerate() {
        for &u in parents.iter() {
            let bit = utils::msbd(u, v);
            assert!(bit < bs);
            // edge j -> i differs at the nth bit
            (&mut eis[bit]).insert(Edge(u, v));
        }
    }
    eis
}

#[cfg(test)]
mod test {

    use super::super::graph;
    use super::*;

    // graph 0->1->2->3->4->5->6->7
    // + 0->2 , 2->4, 4->6

    lazy_static! {
        static ref TEST_PARENTS: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![1, 0],
            vec![2],
            vec![3, 2],
            vec![4],
            vec![5, 4],
            vec![6],
            // FIXME: Always have all immediate predecessors as parents
            // by default to simplify manual construction.
        ];
    }

    #[test]
    fn test_valiant_reduce() {
        let graph = graph::tests::graph_from(TEST_PARENTS.to_vec());
        let set = valiant_basic(&graph, 2);
        assert_eq!(set, HashSet::from_iter(vec![0, 2, 3, 4, 6].iter().cloned()));
        // FIXME: With a previous ordering of the edges and `E_i`s the
        // basic Valiant attack outputted a set `S` of 6 elements
        // `{4, 1, 2, 3, 0, 5}`, instead of this new set of only 5.
        // Both are correct in the sense that keep the depth at 2,
        // but we'd expect Valiant to always return the smallest set
        // necessary (hence `valiant_basic` sorts by `E_i` size). This
        // might just be a product of an unstable search (due in part
        // to the small graph size) but should be investigated further.
    }

    #[test]
    fn test_valiant_partitions() {
        let graph = graph::tests::graph_from(TEST_PARENTS.to_vec());
        let edges = valiant_partitions(&graph);
        assert_eq!(edges.len(), utils::node_bitsize());
        edges
            .into_iter()
            .enumerate()
            .for_each(|(i, edges)| match i {
                0 => {
                    assert_eq!(
                        edges,
                        HashSet::from_iter(
                            vec![Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7)]
                                .iter()
                                .cloned()
                        )
                    );
                }
                1 => {
                    assert_eq!(
                        edges,
                        HashSet::from_iter(
                            vec![Edge(0, 2), Edge(1, 2), Edge(4, 6), Edge(5, 6)]
                                .iter()
                                .cloned()
                        )
                    );
                }
                2 => {
                    assert_eq!(
                        edges,
                        HashSet::from_iter(vec![Edge(2, 4), Edge(3, 4)].iter().cloned())
                    );
                }
                _ => {}
            });
    }
}
