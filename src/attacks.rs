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

// greedy_reduce implements the Algorithm 5 of https://eprint.iacr.org/2018/944.pdf
fn greedy_reduce(g: &Graph, target: usize) -> HashSet<usize> {
    let s = HashSet::new();
    //let nodes_radius: Vec<usize> = Vec::new();
    let reduced = g.remove(&s);
    while reduced.depth() > target {}
    s
}

// append_removal is an adaptation of "SelectRemovalNodes" function in Algorithm 6
// of https://eprint.iacr.org/2018/944.pdf. Instead of returning the set of nodes
// to remove, it simply adds them to the given set.
fn append_removal(
    g: &Graph,
    set: &mut HashSet<usize>,
    topk: Vec<usize>,
    nodes_radius: &mut HashSet<usize>,
    d: usize,
    radius: usize,
    k: usize,
) {
    if radius == 0 {
        // take the node with the highest number of incident path
        set.insert(*topk.iter().max().unwrap());
        return;
    }

    topk.iter()
        .filter(|&node| nodes_radius.contains(node))
        .for_each(|&node| {
            set.insert(node);
            // Add node in radius
        });
}

// update_radius_set fills the given inradius set with nodes that inside a radius
// of the given node. Size of the radius is given radius.
fn update_radius_set(g: &Graph, node: usize, inradius: &mut HashSet<usize>, radius: usize) {
    let add_direct_nodes = |v: usize, closests: &mut Vec<usize>| {
        // add all direct parent
        g.parents()[v]
            .iter()
            .for_each(|&parent| closests.push(parent));

        // add all direct children
        // TODO: compute once children graph and use it again as in C#
        g.parents()
            .iter()
            .enumerate()
            // if node i has v as parent then it's good
            .filter(|&(i, parents)| parents.contains(&v))
            .for_each(|(i, _)| {
                closests.push(i);
            });
    };
    // insert first the given node and then add the close nodes
    inradius.insert(node);
    let mut tosearch = vec![node];
    // do it recursively "radius" times
    for i in 0..radius {
        let mut closests = Vec::new();
        // grab all direct nodes of those already in radius "i"
        for &v in tosearch.iter() {
            add_direct_nodes(v, &mut closests);
        }
        tosearch = closests.clone();
        inradius.extend(closests);
    }
}

// count_paths implements the CountPaths method in Algo. 5 for the greedy algorithm
// It returns:
// 1. the number of incident paths of the given length for each node.
//      Index is the the index of the node, value is the paths count.
// 2. the top k nodes indexes that have the higest incident paths
fn count_paths(g: &Graph, s: &HashSet<usize>, length: usize, k: usize) -> (Vec<usize>, Vec<usize>) {
    // dimensions are [n][depth]
    let mut ending_paths = Vec::new();
    let mut starting_paths = Vec::new();
    // initializes the tables with 1 for nodes present in G - S
    g.parents().iter().enumerate().for_each(|(i, _)| {
        // TODO slow checking in O(n) - consider changing to bitset of size n
        let mut length_vec = vec![0; length + 1];
        if s.contains(&i) {
            ending_paths.push(length_vec.clone());
            starting_paths.push(length_vec);
        } else {
            length_vec[0] = 1;
            ending_paths.push(length_vec.clone());
            starting_paths.push(length_vec);
        }
    });
    // counting phase of all starting/ending paths of all length
    for d in 1..=length {
        g.parents().iter().enumerate().for_each(|(i, parents)| {
            // no need to check if included in S like in C# ref. code
            // since everything is 0 and of the right size by default
            //
            // checking each parents (vs only checking direct + 1parent in C#)
            ending_paths[i][d] = parents
                .iter()
                .fold(0, |acc, &parent| acc + ending_paths[parent][d - 1]);

            // difference vs the pseudo code: like in C#, increase parent count
            // instead of iterating over children of node i
            parents.iter().for_each(|&parent| {
                if s.contains(&parent) {
                    return;
                }
                starting_paths[parent][d] += starting_paths[i][d - 1];
            });
        });
    }

    // counting how many incident paths of length d there is for each node
    let mut incidents = vec![0; g.cap()];
    // the indexes of the nodes with the highest number of incident paths
    let mut topk_idx = vec![0; k];
    // the incident paths number - to compare
    let mut topk_val = vec![0; k];
    for i in 0..g.cap() {
        for d in 0..=length {
            incidents[i] += starting_paths[i][d] * ending_paths[i][length - d];
        }
        let (idx, val) = topk_val.iter().enumerate().min_by_key(|(_, &v)| v).unwrap();
        if *val < incidents[i] {
            topk_val[idx] = incidents[i];
            topk_idx[idx] = i;
        }
    }
    //topk_idx.sort();
    (incidents, topk_idx)
}

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
        static ref GREEDY_PARENTS: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![1, 0],
            vec![2, 1],
            vec![3, 2, 0],
            vec![4],
        ];
    }

    #[test]
    fn test_update_radius() {
        let graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        let node = 2;
        let mut inradius = HashSet::new();

        update_radius_set(&graph, node, &mut inradius, 1);
        assert_eq!(inradius, HashSet::from_iter(vec![0, 1, 2, 3, 4]));
        update_radius_set(&graph, node, &mut inradius, 2);
        assert_eq!(inradius, HashSet::from_iter(vec![0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_count_paths() {
        let graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        let target_length = 2;
        // test with empty set to remove
        let s = HashSet::new();
        let k = 3;
        let (counts, topk) = count_paths(&graph, &s, target_length, k);
        assert_eq!(counts, vec![5, 5, 7, 6, 7, 3]);
        // order is irrelevant
        assert_eq!(topk, vec![3, 4, 2]);
        // TODO test with a a non empty s
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
