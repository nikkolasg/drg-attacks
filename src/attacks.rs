use super::graph::Graph;
use super::utils;

/*pub enum DepthReduceSet {*/
//ValiantBasic(usize),
//}

//pub fn depth_reduce(g: &Graph, drs: DepthReduceSet) -> Graph {
//match drs {
//DepthReduceSet::ValiantBasic(target) => valiant_basic_depth(g, target),
//}
//}

// greedy_reduce implements the Algorithm 5 of https://eprint.iacr.org/2018/944.pdf
fn greedy_reduce(g: &Graph, target: usize) -> Vec<usize> {
    let s: Vec<usize> = Vec::new();
    //let nodes_radius: Vec<usize> = Vec::new();
    let reduced = g.remove(&s);
    while reduced.depth() > target {}
    s
}

// count_paths implements the CountPaths method in Algo. 5 for the greedy algorithm
// It returns the number of incident paths of the given length for each node.
fn count_paths(g: &Graph, s: &Vec<usize>, length: usize) -> Vec<usize> {
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
    for i in 0..g.cap() {
        for d in 0..=length {
            incidents[i] += starting_paths[i][d] * ending_paths[i][length - d];
        }
    }
    incidents
}

// valiant_basic returns a set S such that depth(G - S) < target.
// It implements the algo 8 in the https://eprint.iacr.org/2018/944.pdf paper.
fn valiant_basic(g: &Graph, target: usize) -> Vec<usize> {
    let partitions = valiant_partitions(g);
    // TODO replace by a simple bitset or boolean vec
    let mut chosen: Vec<usize> = Vec::new();
    let mut s: Vec<usize> = Vec::new();
    let mut reduced = g.remove(&s);
    // returns the smallest next partition unchosen
    // mut is required because it changes chosen which is mut
    let mut find_next = || -> &Vec<Edge> {
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
#[derive(Debug, PartialEq)]
struct Edge(usize, usize);

// valiant_partitions returns the sets E_i and S_i from the given graph
// according to the definition algorithm 8 from
// https://eprint.iacr.org/2018/944.pdf .
fn valiant_partitions(g: &Graph) -> Vec<Vec<Edge>> {
    let bs = utils::node_bitsize();
    let mut eis = Vec::with_capacity(bs);
    for _ in 0..bs {
        eis.push(Vec::new());
    }

    for (i, parents) in g.parents().iter().enumerate() {
        for &j in parents.iter() {
            let bit = utils::msbd(j, i);
            assert!(bit < bs);
            // edge j -> i differs at the nth bit
            (&mut eis[bit]).push(Edge(j, i));
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
            vec![0, 1],
            vec![2],
            vec![2, 3],
            vec![4],
            vec![4, 5],
            vec![6],
        ];
    }

    #[test]
    fn test_count_paths() {
        let parents = vec![
            vec![],
            vec![0],
            vec![1, 0],
            vec![2, 1],
            vec![3, 2, 0],
            vec![4],
        ];
        let graph = graph::tests::graph_from(parents);
        let target_length = 2;
        // test with empty set to remove
        let s = Vec::new();
        let result = count_paths(&graph, &s, target_length);
        assert_eq!(result, vec![5, 5, 7, 6, 7, 3]);
    }

    #[test]
    fn test_valiant_reduce() {
        let graph = graph::tests::graph_from(TEST_PARENTS.to_vec());
        let set = valiant_basic(&graph, 2);
        assert_eq!(set, vec![2, 3, 0, 1, 4, 5]);
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
                63 => {
                    let exp = vec![Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7)];
                    assert_eq!(edges, exp);
                }
                62 => {
                    let exp = vec![Edge(0, 2), Edge(1, 2), Edge(4, 6), Edge(5, 6)];
                    assert_eq!(edges, exp);
                }
                61 => {
                    let exp = vec![Edge(2, 4), Edge(3, 4)];
                    assert_eq!(edges, exp);
                }
                _ => {}
            });
    }
}
