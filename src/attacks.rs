use std::collections::HashSet;

use crate::graph::Graph;
use crate::utils;

#[derive(Debug)]
pub enum DepthReduceSet {
    // depth of the resulting G-S graph desired
    ValiantDepth(usize),
    // size of the resulting S desired
    ValiantSize(usize),
    // depth of the resulting G-S graph and some specific parameters
    Greedy(usize, GreedyParams),
}

pub fn depth_reduce(g: &mut Graph, drs: DepthReduceSet) -> HashSet<usize> {
    match drs {
        DepthReduceSet::ValiantDepth(_) => valiant_reduce(g, drs),
        DepthReduceSet::ValiantSize(_) => valiant_reduce(g, drs),
        DepthReduceSet::Greedy(target, p) => greedy_reduce(g, target, p),
    }
}

// GreedyParams holds the different parameters to choose for the greedy algorithm
// such as the radius from which to delete nodes and the heuristic length.
#[derive(Debug)]
pub struct GreedyParams {
    // how many k nodes do we "remove" at each iteration in append_removal
    pub k: usize,
    // the radius for the heuristic to delete as well close nodes within a
    // radius of a selected node.
    pub radius: usize,
    //#[allow(dead_code)]
    // maximum lenth of the path - heuristic for the table produced by count_paths
    // see paragraph below equation 8.
    // TODO: NOT (really) IMPLEMENTED YET - careful work is required
    // pub length: usize,
    // test field to look at the impact of reseting the inradius set between
    // iterations or not.
    pub reset: bool,
}
// greedy_reduce implements the Algorithm 5 of https://eprint.iacr.org/2018/944.pdf
fn greedy_reduce(g: &mut Graph, target: usize, p: GreedyParams) -> HashSet<usize> {
    let mut s = HashSet::new();
    g.children_project();
    let mut inradius: HashSet<usize> = HashSet::new();
    while g.depth_exclude(&s) > target {
        // TODO use p.length when more confidence in the trick
        //let (_, topk) = count_paths(g, &s, target, p.k);
        let (counts, topk) = count_paths(g, &s, target, p.k);
        println!(
            "main loop: depth {} > {}\n\t-> counts.len {:?}",
            g.depth_exclude(&s),
            target,
            counts.len(),
        );
        append_removal(g, &mut s, &mut inradius, &topk, p.radius);
        // TODO
        // 1. Find what should be the normal behavior: clearing or continue
        // updating the inradius set
        // 2. In the latter case, optimization to not re-allocate each time
        // since could be quite big with large k and radius
        if p.reset {
            inradius.clear();
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
    inradius: &mut HashSet<usize>,
    topk: &Vec<Pair>,
    radius: usize,
) {
    if radius == 0 {
        // take the node with the highest number of incident path
        set.insert(topk.iter().max_by_key(|pair| pair.1).unwrap().0);
        return;
    }

    let mut count = 0;
    for node in topk.iter() {
        if inradius.contains(&node.0) {
            // difference with previous insertion is that we only include
            // nodes NOT in the radius set
            continue;
        }
        set.insert(node.0);
        update_radius_set(g, node.0, inradius, radius);
        count += 1;
        println!(
            "\t-> iteration {} : node {} inserted -> inradius {:?}",
            count, node.0, inradius,
        );
    }
    // if all nodes are already in the radius set, then take the
    // the one with the maximum incident paths.
    if count == 0 {
        let node = &topk.iter().max_by_key(|n| n.1).unwrap();
        set.insert(node.0);
        update_radius_set(g, node.0, inradius, radius);
    }

    println!("\t-> topk {:?}", topk);
    println!("\t-> added {} nodes in S: {:?}", count, set);
}

// update_radius_set fills the given inradius set with nodes that inside a radius
// of the given node. Size of the radius is given radius. It corresponds to the
// under-specified function "UpdateNodesInRadius" in algo. 6 of
// https://eprint.iacr.org/2018/944.pdf
fn update_radius_set(g: &Graph, node: usize, inradius: &mut HashSet<usize>, radius: usize) {
    let add_direct_nodes = |v: usize, closests: &mut HashSet<usize>, _: &HashSet<usize>| {
        // add all direct parent
        g.parents()[v]
            .iter()
            // no need to continue searching with that parent since it's
            // already in the radius, i.e. it already has been searched
            // FIXME see if it works and resolves any potential loops
            //.filter(|&parent| !rad.contains(parent))
            .for_each(|&parent| {
                closests.insert(parent);
            });

        // add all direct children
        g.children()[v]
            .iter()
            // no need to continue searching with that parent since it's
            // already in the radius, i.e. it already has been searched
            //.filter(|&child| !rad.contains(child))
            .for_each(|&child| {
                closests.insert(child);
            });
        println!(
            "\t add_direct node {}: at most {} parents and {} children",
            v,
            g.parents()[v].len(),
            g.children()[v].len()
        );
    };
    // insert first the given node and then add the close nodes
    inradius.insert(node);
    let mut tosearch = HashSet::new();
    tosearch.insert(node);
    // do it recursively "radius" times
    for i in 0..radius {
        let mut closests = HashSet::new();
        // grab all direct nodes of those already in radius "i"
        for &v in tosearch.iter() {
            add_direct_nodes(v, &mut closests, inradius);
        }
        tosearch = closests.clone();
        inradius.extend(closests);
        println!(
            "update radius {}: {} new nodes, total {}",
            i,
            tosearch.len(),
            inradius.len()
        );
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
    let mut ending_paths = vec![vec![0; length + 1]; g.cap()];
    let mut starting_paths = vec![vec![0; length + 1]; g.cap()];
    // counting phase of all starting/ending paths of all length
    for d in 1..=length {
        g.parents().iter().enumerate().for_each(|(i, parents)| {
            if !s.contains(&i) {
                // initializes the tables with 1 for nodes present in G - S
                ending_paths[i][0] = 1;
                starting_paths[i][0] = 1;
            }
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

fn valiant_reduce(g: &Graph, d: DepthReduceSet) -> HashSet<usize> {
    match d {
        // valiant_reduce returns a set S such that depth(G - S) < target.
        // It implements the algo 8 in the https://eprint.iacr.org/2018/944.pdf paper.
        DepthReduceSet::ValiantDepth(depth) => {
            valiant_reduce_main(g, &|set: &HashSet<usize>| g.depth_exclude(set) > depth)
        }
        DepthReduceSet::ValiantSize(size) => {
            valiant_reduce_main(g, &|set: &HashSet<usize>| set.len() < size)
        }
        _ => panic!("that should not happen"),
    }
}

fn valiant_reduce_main(g: &Graph, f: &Fn(&HashSet<usize>) -> bool) -> HashSet<usize> {
    let partitions = valiant_partitions(g);
    // TODO replace by a simple bitset or boolean vec
    let mut chosen: Vec<usize> = Vec::new();
    let mut s = HashSet::new();
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
    while f(&s) {
        let partition = find_next();
        // add the origin node for each edges in the chosen partition
        s.extend(partition.iter().fold(Vec::new(), |mut acc, edge| {
            acc.push(edge.0);
            acc
        }));
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
        let mut graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        graph.children_project();
        let params = GreedyParams {
            k: 1,
            radius: 0,
            reset: false,
        };
        let s = greedy_reduce(&mut graph, 2, params);
        assert_eq!(s, HashSet::from_iter(vec![3, 2]));
        let params = GreedyParams {
            k: 1,
            radius: 1,
            reset: false,
        };
        let s = greedy_reduce(&mut graph, 2, params);
        // 1st iteration : counts = [5, 5, 7, 6, 7, 3]
        // 2nd iteration : counts =  [2, 2, 0, 3, 3, 2]
        // so first index 2 then index 3 (takes the minimum in the list)
        assert_eq!(s, HashSet::from_iter(vec![3, 2]));
        println!("\n\n\n ------\n\n\n");
        let params = GreedyParams {
            k: 2,
            radius: 1,
            reset: false,
        };
        let s = greedy_reduce(&mut graph, 2, params);
        // main loop: depth 5 > 2
        //         -> counts [5, 5, 7, 6, 7, 3]
        //         -> iteration 1 : node 2 inserted -> inradius {2, 0, 1, 3, 4}
        //         -> topk [Pair(2, 7), Pair(4, 7)]
        //         -> added 1 nodes in S: {2}
        // main loop: depth 4 > 2
        //         -> counts [2, 2, 0, 3, 3, 2]
        //         -> topk [Pair(3, 3), Pair(4, 3)]
        //         -> added 1 nodes in S: {4, 2} <-- thanks to the rule
        //         when all nodes are in the radius, we take the highest one
        //         and max_by_key returns the latest.
        assert_eq!(s, HashSet::from_iter(vec![2, 4]));
    }

    #[test]
    fn test_append_removal_node() {
        let mut graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        graph.children_project();
        let mut s = HashSet::new();
        let k = 3;
        let target_length = 2;
        let (_, topk) = count_paths(&graph, &s, target_length, k);
        let radius = 0;
        let mut inradius = HashSet::new();
        append_removal(&graph, &mut s, &mut inradius, &topk, radius);
        assert!(s.contains(&2)); // 4 is valid but (2,7) is last

        let radius = 1;
        let (counts, topk) = count_paths(&graph, &s, target_length, k);
        println!("counts {:?}", counts);
        append_removal(&graph, &mut s, &mut inradius, &topk, radius);
        // counts [2, 2, 0, 3, 3, 2]
        // -> topk [Pair(4, 3), Pair(1, 2), Pair(3, 3)]
        // -> iteration 1 : node 4 inserted -> inradius {5, 0, 3, 2, 4}
        // -> iteration 2 : node 1 inserted -> inradius {5, 0, 1, 3, 2, 4}
        // 2 is there from the previous call to append_removal
        assert_eq!(s, HashSet::from_iter(vec![2, 4, 1]));
        // TODO probably more tests with larger graph
    }

    #[test]
    fn test_update_radius() {
        let mut graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        graph.children_project();
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
        assert_eq!(counts, vec![3, 3, 3, 3, 0, 0]);
        assert_eq!(topk, vec![Pair(0, 3), Pair(1, 3), Pair(2, 3)]);
    }

    #[test]
    fn test_valiant_reduce_depth() {
        let graph = graph::tests::graph_from(TEST_PARENTS.to_vec());
        let set = valiant_reduce(&graph, DepthReduceSet::ValiantDepth(2));
        assert_eq!(set, HashSet::from_iter(vec![0, 2, 3, 4, 6]));
    }

    #[test]
    fn test_valiant_reduce_size() {
        let graph = graph::tests::graph_from(TEST_PARENTS.to_vec());
        let set = valiant_reduce(&graph, DepthReduceSet::ValiantSize(3));
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
