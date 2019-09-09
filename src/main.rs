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

fn main() {
    println!("DRG graph generation");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let size = (2 as usize).pow(20);
    let deg = 6;
    let depth = (2 as usize).pow(8);
    let g1 = Graph::new(size, random_bytes, DRGAlgo::MetaBucket(deg));
    println!("Attack graph with valiant");
    let start = Instant::now();
    let valiant = depth_reduce(&g1, DepthReduceSet::Valiant(depth));
    let duration = start.elapsed();
    println!("\t-> valiant: size {}", valiant.len());
    println!("\t-> time elapsed for valiant is: {:?}", duration);

    let start = Instant::now();
    println!("Attack graph with greedy");
    let greedy = depth_reduce(
        &g1,
        DepthReduceSet::Greedy(5, GreedyParams { k: 3, radius: 2 }),
    );
    let duration = start.elapsed();
    println!("\t-> time elapsed for greedy is: {:?}", duration);
    println!("\t-> greedy:  size {}", greedy.len());
}
