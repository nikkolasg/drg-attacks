mod attacks;
pub mod graph;
mod utils;
use attacks::{depth_reduce, DepthReduceSet, GreedyParams};
use graph::{DRGAlgo, Graph};
use rand::Rng;
use std::time::{Duration, Instant};

// used by test module...
#[macro_use]
extern crate lazy_static;

fn attack(g: &mut Graph, r: DepthReduceSet) {
    println!("Attack with {:?}", r);
    let start = Instant::now();
    let set = depth_reduce(g, r);
    let duration = start.elapsed();
    println!("\t-> size {}", set.len());
    println!("\t-> time elapsed: {:?}", duration);
}

fn main() {
    println!("DRG graph generation");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let size = (2 as usize).pow(10);
    let deg = 6;
    let depth = (2 as usize).pow(7);
    let mut g1 = Graph::new(size, random_bytes, DRGAlgo::MetaBucket(deg));
    attack(&mut g1, DepthReduceSet::Valiant(depth));

    attack(
        &mut g1,
        DepthReduceSet::Greedy(5, GreedyParams { k: 3, radius: 4 }),
    );
}
