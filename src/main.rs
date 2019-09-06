mod attacks;
pub mod graph;
mod utils;
use attacks::{depth_reduce, DepthReduceSet, GreedyParams};
use graph::{DRGAlgo, Graph};
use rand::Rng;

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
    let valiant = depth_reduce(&g1, DepthReduceSet::Valiant(depth));
    println!("Attack graph with greedy");
    let greedy = depth_reduce(
        &g1,
        DepthReduceSet::Greedy(5, GreedyParams { k: 3, radius: 2 }),
    );
    println!("valiant: size {}", valiant.len());
    println!("greedy:  size {}", greedy.len());
}
