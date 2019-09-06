use std::collections::HashSet;

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

struct GreedyParams {
    // how many k nodes do we "remove" at each iteration in append_removal
    k: usize,
    // the radius for the heuristic to delete as well close nodes within a
    // radius of a selected node.
    radius: usize,
    #[allow(dead_code)]
    // maximum lenth of the path - heuristic for the table produced by count_paths
    // see paragraph below equation 8.
    // TODO: NOT (really) IMPLEMENTED YET - careful work is required
    length: usize,
}
// greedy_reduce implements the Algorithm 5 of https://eprint.iacr.org/2018/944.pdf
fn greedy_reduce(g: &Graph, target: usize, p: GreedyParams) -> HashSet<usize> {
    let mut s = HashSet::new();
    let mut inradius: HashSet<usize> = HashSet::new();
    let mut reduced = g.remove(&s);
    println!("graph: {:?}", g);
    let mut count = 0;
    while reduced.depth() > target {
        println!(" ---- new iteration ----");
        // TODO use p.length when more confidence in the trick
        let (counts, topk) = count_paths(g, &s, target, p.k);
        append_removal(g, &mut s, &topk, &mut inradius, p.radius);
        reduced = reduced.remove(&s);
        println!("-> counts {:?}", counts);
        println!("-> topk   {:?}", topk);
        println!("-> s      {:?}", s);
        println!("-> radius {:?}", inradius);
        count += 1;
        if count > 3 {
            panic!("aie");
        }
    }
    s
}

// append_removal is an adaptation of "SelectRemovalNodes" function in Algorithm 6
// of https://eprint.iacr.org/2018/944.pdf. Instead of returning the set of nodes
// to remove, it simply adds them to the given set.
fn append_removal(
    g: &Graph,
    set: &mut HashSet<usize>,
    topk: &Vec<Pair>,
    inradius: &mut HashSet<usize>,
    radius: usize,
) {
    if radius == 0 {
        // take the node with the highest number of incident path
        set.insert(topk.iter().max_by_key(|pair| pair.1).unwrap().0);
        return;
    }

    let mut unseen = topk
        .iter()
        // take the nodes that are not yet in the inradius set
        .filter(|&pair| !inradius.contains(&pair.0))
        .collect::<Vec<&Pair>>();

    // if all nodes are already in the radius set, then at least take
    // the first one.
    // https://github.com/filecoin-project/drg-attacks/issues/2
    // for more details.
    if unseen.len() == 0 {
        unseen.push(&topk[0]);
    }

    for &pair in unseen.iter() {
        set.insert(pair.0);
        update_radius_set(g, pair.0, inradius, radius);
    }
}

// update_radius_set fills the given inradius set with nodes that inside a radius
// of the given node. Size of the radius is given radius. It corresponds to the
// under-specified function "UpdateNodesInRadius" in algo. 6 of
// https://eprint.iacr.org/2018/944.pdf
fn update_radius_set(g: &Graph, node: usize, inradius: &mut HashSet<usize>, radius: usize) {
    let add_direct_nodes = |v: usize, closests: &mut Vec<usize>| {
        // add all direct parent
        g.parents()[v]
            .iter()
            .for_each(|&parent| closests.push(parent));

        // add all direct children
        // TODO: compute once children graph and use it again as in C# instead
        // of searching linearly
        g.parents()
            .iter()
            .enumerate()
            // if node i has v as parent then it's good
            .filter(|&(_, parents)| parents.contains(&v))
            .for_each(|(i, _)| {
                closests.push(i);
            });
    };
    // insert first the given node and then add the close nodes
    inradius.insert(node);
    let mut tosearch = vec![node];
    // do it recursively "radius" times
    for _ in 0..radius {
        let mut closests = Vec::new();
        // grab all direct nodes of those already in radius "i"
        for &v in tosearch.iter() {
            add_direct_nodes(v, &mut closests);
        }
        tosearch = closests.clone();
        inradius.extend(closests);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Pair(usize, usize);

// count_paths implements the CountPaths method in Algo. 5 for the greedy algorithm
// It returns:
// 1. the number of incident paths of the given length for each node.
//      Index is the the index of the node, value is the paths count.
// 2. the top k nodes indexes that have the higest incident paths
//      The number of incident path is not given.
fn count_paths(g: &Graph, s: &HashSet<usize>, length: usize, k: usize) -> (Vec<usize>, Vec<Pair>) {
    // dimensions are [n][depth]
    let mut ending_paths = Vec::new();
    let mut starting_paths = Vec::new();
    // initializes the tables with 1 for nodes present in G - S
    g.parents().iter().enumerate().for_each(|(i, _)| {
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
            // checking each parents (vs only checking direct + 1parent in C#)
            ending_paths[i][d] = parents
                .iter()
                // no ending path for node i if the parent is contained in S
                // since G - S doesn't have this parent
                .filter(|p| !s.contains(p))
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
    // counting the top k node wo have the greatest number of incident paths
    // NOTE: difference with the C# that recomputes that vector separately.
    // Since topk is directly correlated to incidents[], we can compute both
    // at the same time and remove one O(n) iteration.
    let mut topk = vec![Pair(0, 0); k];
    let mut inserted = 0;
    for i in 0..g.cap() {
        for d in 0..=length {
            incidents[i] += starting_paths[i][d] * ending_paths[i][length - d];
        }
        let (idx, pair) = topk
            .iter()
            .cloned()
            .enumerate()
            .min_by_key(|(_, pair)| pair.1)
            .unwrap();

        // replace if the minimum number of incident paths in topk is smaller
        // than the one computed for node i in this iteration
        if pair.1 < incidents[i] {
            topk[idx] = Pair(i, incidents[i]);
            inserted += 1;
        }
    }
    assert!(inserted > 0);
    (incidents, topk)
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
    use std::iter::FromIterator;

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
    fn test_greedy() {
        let graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        let params = GreedyParams {
            k: 1,
            radius: 0,
            length: 2,
        };
        let s = greedy_reduce(&graph, 2, params);
        //assert_eq!(s, HashSet::from_iter(vec![3, 4]));
        println!("{:?}", s);
    }

    #[test]
    fn test_append_removal_node() {
        let graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        let mut s = HashSet::new();
        let mut inradius = HashSet::new();
        let k = 3;
        let target_length = 2;
        let (_, topk) = count_paths(&graph, &s, target_length, k);
        let radius = 0;
        append_removal(&graph, &mut s, &topk, &mut inradius, radius);
        assert!(s.contains(&2)); // 4 is valid but (2,7) is last
        assert_eq!(inradius.len(), 0);

        let radius = 1;
        let (_, topk) = count_paths(&graph, &s, target_length, k);
        append_removal(&graph, &mut s, &topk, &mut inradius, radius);
        // 2,3,4,5 because
        // (1) node 2 was inserted at the previous call (prev. line)
        // (2) then the next top 3 are node 1 3 4
        assert_eq!(s, HashSet::from_iter(vec![2, 1, 4, 3]));
        // the whole graph because the neighbors of the set S(2,3,4,5)
        // with a radius of 1 contains 0 and 1 (thanks to node 2)
        assert_eq!(inradius, HashSet::from_iter((0..6).collect::<Vec<usize>>()));
        // TODO probably more tests with larger graph
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
        let mut s = HashSet::new();
        let k = 3;
        let (counts, topk) = count_paths(&graph, &s, target_length, k);
        assert_eq!(counts, vec![5, 5, 7, 6, 7, 3]);
        // order is irrelevant so we keep vec
        assert_eq!(topk, vec![Pair(3, 6), Pair(4, 7), Pair(2, 7)]);
        s.insert(4);
        let (counts, topk) = count_paths(&graph, &s, target_length, k);
        println!("counts {:?}", counts);
        println!("topk: {:?}", topk);
        assert_eq!(counts, vec![3, 3, 3, 3, 0, 0]);
        assert_eq!(topk, vec![Pair(0, 3), Pair(1, 3), Pair(2, 3)]);
    }

    #[test]
    fn test_valiant_reduce() {
        let graph = graph::tests::graph_from(TEST_PARENTS.to_vec());
        let set = valiant_basic(&graph, 2);
        assert_eq!(set, HashSet::from_iter(vec![0, 2, 3, 4, 6]));
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
                        HashSet::from_iter(vec![Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7)])
                    );
                }
                1 => {
                    assert_eq!(
                        edges,
                        HashSet::from_iter(vec![Edge(0, 2), Edge(1, 2), Edge(4, 6), Edge(5, 6)])
                    );
                }
                2 => {
                    assert_eq!(edges, HashSet::from_iter(vec![Edge(2, 4), Edge(3, 4)]));
                }
                _ => {}
            });
    }
}
