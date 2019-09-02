mod attacks;
pub mod graph;
mod utils;
use graph::*;
use rand::Rng;

fn main() {
    println!("DRG graph generation");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let g1 = Graph::new(16, random_bytes, DRGAlgo::BucketSample);
    println!("{}", g1);

    let g2 = Graph::new(16, random_bytes, DRGAlgo::MetaBucket(3));
    println!("{}", g2);
}
