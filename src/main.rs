mod attacks;
pub mod graph;
mod utils;
use attacks::{depth_reduce, DepthReduceSet, GreedyParams};
use graph::{DRGAlgo, Graph};
use rand::Rng;
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
    println!("\t-> size {}", set.len());
    println!("\t-> depth(G-S) {}", depth);
    println!("\t-> time elapsed: {:?}", duration);
}

fn porep_comparison() {
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let size = (2 as usize).pow(24);
    println!("Comparison with porep short paper with n = {}", size);
    let deg = 6;
    let mut g1 = Graph::new(size, random_bytes, DRGAlgo::MetaBucket(deg));

    let depth = (0.25 * (size as f32)) as usize;
    println!("{}", g1.stats());
    println!("Trial #1 with target depth = 0.25n = {}", depth);
    attack(&mut g1, DepthReduceSet::ValiantDepth(depth));

    let set_size = (0.30 * (size as f32)) as usize;
    println!(
        "Trial #2 with target size set = 0.30n = {} (G-S = 0.7n)",
        set_size
    );
    attack(&mut g1, DepthReduceSet::ValiantSize(set_size));
    // graph stats: size=16777216, min parents=1, max children=24
    // Trial #1 with target depth = 0.25n = 4194304
    // Attack with ValiantDepth(4194304)
    //         -> size 4876208
    //         -> depth(G-S) 3947807
    //         -> time elapsed: 851.764374598s
    // Trial #2 with target size set = 0.30n = 5033165 (G-S = 0.7n)
    // Attack with ValiantSize(5033165)
    //         -> size 6907804
    //         -> depth(G-S) 1869564
    //         -> time elapsed: 531.527749373s
}

fn large_graphs() {
    println!("Large graph scenario");
    println!("DRG graph generation");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let size = (2 as usize).pow(10);
    let deg = 6;
    let depth = (2 as usize).pow(7);
    let mut g1 = Graph::new(size, random_bytes, DRGAlgo::MetaBucket(deg));
    attack(&mut g1, DepthReduceSet::ValiantDepth(depth));

    attack(
        &mut g1,
        DepthReduceSet::Greedy(
            depth,
            GreedyParams {
                k: 30,
                radius: 5,
                reset: false,
            },
        ),
    );
}

fn small_graph() {
    println!("Large graph scenario");
    println!("DRG graph generation");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let size = 100;
    let deg = 4;
    let depth = 50;
    let mut g1 = Graph::new(size, random_bytes, DRGAlgo::MetaBucket(deg));
    attack(&mut g1, DepthReduceSet::ValiantDepth(depth));

    attack(
        &mut g1,
        DepthReduceSet::Greedy(
            depth,
            GreedyParams {
                k: 1,
                radius: 0,
                reset: false,
            },
        ),
    );
}

fn main() {
    //small_graph();
    //large_graphs();
    porep_comparison();
}
