mod attacks;
pub mod graph;
mod utils;
use attacks::{depth_reduce, DepthReduceSet, GreedyParams};
use graph::{DRGAlgo, Graph};
use rand::Rng;
use std::env;
use std::time::Instant;

// used by test module...
#[macro_use]
extern crate lazy_static;

fn attack(g: &mut Graph, r: DepthReduceSet) {
    println!("Attack with {:?}", r);
    let start = Instant::now();
    let set = depth_reduce(g, r);
    let duration = start.elapsed();
    let depth = g.depth_exclude(&set);
    println!(
        "\t-> |S| = {} = {:.4}n",
        set.len(),
        (set.len() as f32) / (g.cap() as f32)
    );
    println!(
        "\t-> depth(G-S) = {} = {:.4}n",
        depth,
        (depth as f32) / (g.cap() as f32)
    );
    println!("\t-> time elapsed: {:?}", duration);
}

fn porep_comparison() {
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 20;
    let size = (2 as usize).pow(n);
    println!("Comparison with porep short paper with n = {}", size);
    let deg = 6;
    let fname = format!("porep_n{}_d{}.json", n, deg);

    let mut g1 = Graph::load_or_create(&fname, size, random_bytes, DRGAlgo::MetaBucket(deg));
    //let mut g1 = Graph::new(size, random_bytes, DRGAlgo::MetaBucket(deg));

    let depth = (0.25 * (size as f32)) as usize;
    println!("{}", g1.stats());
    println!("Trial #1 with target depth = 0.25n = {}", depth);
    //attack(&mut g1, DepthReduceSet::ValiantDepth(depth));

    //let set_size = (0.30 * (size as f32)) as usize;
    //println!(
    //"Trial #2 with target size set = 0.30n = {} (G-S = 0.7n)",
    //set_size
    //);
    //attack(&mut g1, DepthReduceSet::ValiantSize(set_size));

    //println!(
    //"Trial #3 with Valiant AB16, target depth = 0.25n = {}",
    //depth
    //);
    /*attack(&mut g1, DepthReduceSet::ValiantAB16(depth));*/

    println!("Trial #4 with Greedy DRS, target depth = 0.25n = {}", depth);
    attack(
        &mut g1,
        DepthReduceSet::Greedy(
            depth,
            GreedyParams {
                k: GreedyParams::k_ratio(n as usize),
                radius: 5,
                length: 16,
                reset: true,
                iter_topk: true,
                ..GreedyParams::default()
            },
        ),
    );

    // Comparison with porep short paper with n = 1048576
    // graph stats: size=1048576, min parents=1, max children=26
    // Trial #1 with target depth = 0.25n = 262144
    // Attack with ValiantDepth(262144)
    //         -> size 344275 = 0.3283n
    //         -> depth(G-S) 234005 = 0.2232n
    //         -> time elapsed: 54.654373484s
    // Trial #2 with target size set = 0.30n = 314572 (G-S = 0.7n)
    // Attack with ValiantSize(314572)
    //         -> size 344275 = 0.3283n
    //         -> depth(G-S) 234005 = 0.2232n
    //         -> time elapsed: 36.29261127s
    // Trial #3 with Valiant AB16, target depth = 0.25n = 262144
    // Attack with ValiantAB16(262144)
    //         -> size 319204 = 0.3044n
    //         -> depth(G-S) 247292 = 0.2358n
    //         -> time elapsed: 97.742500864s

    // NOTE: AB16 seems slower and less performant than the ValiantDepth
}

fn greedy_attacks() {
    println!("Greedy Attacks parameters");
    println!("DRG graph generation");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 20;
    let size = (2 as usize).pow(n);
    let deg = 6;
    let depth = (0.25 * size as f64) as usize;
    let fname = format!("greedy_n{}_deg{}.json", n, deg);
    let mut g1 = Graph::load_or_create(&fname, size, random_bytes, DRGAlgo::MetaBucket(deg));
    println!(
        "Greedy attacks tests with size = {}, depth(G-S) <= {}",
        size, depth
    );

    //attack(&mut g1, DepthReduceSet::ValiantDepth(depth));

    let mut greed_params = GreedyParams {
        k: GreedyParams::k_ratio(n as usize),
        radius: 4,
        reset: true,
        // length influences the number of points taken from topk in one iteration
        // if it is too high, then too many nodes will be in the radius so we'll
        // only take the first entry in topk but not the rest (since they'll be in
        // the radius set)
        length: 8,
        iter_topk: true,
        use_degree: false,
    };

    attack(&mut g1, DepthReduceSet::Greedy(depth, greed_params.clone()));
    greed_params.use_degree = true;
    attack(&mut g1, DepthReduceSet::Greedy(depth, greed_params.clone()));

    greed_params.iter_topk = false;
    attack(&mut g1, DepthReduceSet::Greedy(depth, greed_params.clone()));
    // k_ratio seems to give XXX
    greed_params.k = 300; // normally 2^(n-18)/2 * 400 -> take the minimum and reduce
    attack(&mut g1, DepthReduceSet::Greedy(depth, greed_params.clone()));
    // reset seems to give a slightly worse result
    greed_params.reset = false;
    attack(&mut g1, DepthReduceSet::Greedy(depth, greed_params.clone()));
    // higher radius seems to give XXX
    greed_params.radius = 8;
    attack(&mut g1, DepthReduceSet::Greedy(depth, greed_params.clone()));
    // higher length seems to give XXX
    greed_params.length = 32;
    attack(&mut g1, DepthReduceSet::Greedy(depth, greed_params.clone()));
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        match args[1].to_lowercase().trim() {
            "greedy" => greedy_attacks(),
            "porep" => porep_comparison(),
            _ => panic!("command not understood: choose greedy or porep"),
        }
    } else {
        porep_comparison();
    }
}
