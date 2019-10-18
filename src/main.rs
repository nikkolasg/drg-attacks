#![feature(test)]
mod attacks;
pub mod graph;
mod utils;
use attacks::{attack, attack_with_profile, AttackProfile, DepthReduceSet, GreedyParams};
use graph::{DRGAlgo, Graph, GraphSpec};
use rand::Rng;
use serde_json::Result;
use std::env;
// used by test module...
#[macro_use]
extern crate lazy_static;
extern crate rayon;
extern crate test;

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
        DepthReduceSet::GreedySize(
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
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 8;
    let size = (2 as usize).pow(n);
    let deg = 6;
    let target_size = (0.30 * size as f64) as usize;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: DRGAlgo::MetaBucket(deg),
    };
    let runs = 10;
    //attack(&mut g1, DepthReduceSet::ValiantDepth(depth));

    let mut greed_params = GreedyParams {
        k: 50,
        radius: 4,
        reset: true,
        // length influences the number of points taken from topk in one iteration
        // if it is too high, then too many nodes will be in the radius so we'll
        // only take the first entry in topk but not the rest (since they'll be in
        // the radius set)
        length: 8,
        iter_topk: true,
        use_degree: false,
        parallel: false,
    };

    let mut profile = AttackProfile::from_attack(
        DepthReduceSet::GreedySize(target_size, greed_params.clone()),
        size,
    );
    // FIXME: Build the profile in one statement instead of making it mutable.
    profile.runs = runs;
    profile.range.start = 0.2;
    profile.range.end = 0.5;
    profile.range.interval = 0.1;

    let res1 = attack_with_profile(spec, &profile);

    greed_params.length = 16;
    let mut profile = AttackProfile::from_attack(
        DepthReduceSet::GreedySize(target_size, greed_params.clone()),
        size,
    );
    profile.runs = runs;
    profile.range.start = 0.2;
    profile.range.end = 0.5;
    profile.range.interval = 0.1;

    let res2 = attack_with_profile(spec, &profile);
    println!("\n\n------------------");
    let json = serde_json::to_string_pretty(&vec![res1, res2]).expect("can't serialize to json");
    println!("{}", json);
}

fn baseline_valiant() {
    println!("Baseline VALIANT computation for target size [0.10,0.20,0.30]");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 20;
    let size = (2 as usize).pow(n);
    let deg = 6;
    let target_size = (0.30 * size as f64) as usize;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: DRGAlgo::MetaBucket(deg),
    };

    // target depth
    let mut profile = AttackProfile::from_attack(DepthReduceSet::ValiantDepth(target_size), size);
    profile.runs = 3;
    profile.range.start = 0.15;
    profile.range.end = 0.26;
    profile.range.interval = 0.05;

    let res1 = attack_with_profile(spec, &profile);
    // target size
    let mut profile = AttackProfile::from_attack(DepthReduceSet::ValiantSize(target_size), size);
    profile.runs = 3;
    profile.range.start = 0.15;
    profile.range.end = 0.31;
    profile.range.interval = 0.05;

    let res2 = attack_with_profile(spec, &profile);
    let json = serde_json::to_string_pretty(&vec![res1,res2]).expect("can't serialize to json");
    println!("{}", json);
}

fn theoretical_limit() {
    let ts = 0.0000115;
    let td = 0.03;
    println!("Comparing against theoretical limit:");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 20;
    let size = (2 as usize).pow(n);
    let deg = 6;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: DRGAlgo::MetaBucket(deg),
    };


    let mut greed_params = GreedyParams {
        k: GreedyParams::k_ratio(n as usize),
        radius: 4,
        reset: true,
        length: 10,
        iter_topk: true,
        use_degree: true,
        parallel: true,
    };
     
    let mut profile = AttackProfile::from_attack(
        DepthReduceSet::GreedySize((ts * size as f64) as usize, greed_params.clone()),
        size,
    );
    profile.runs = 3;
    profile.range.start = 0.0000115;
    profile.range.end = 0.0000116;
    profile.range.interval = 0.1;

    let res0 = attack_with_profile(spec, &profile);
    println!("json: {}",serde_json::to_string_pretty(&res0).expect("can't serialize to json"));

    let mut profile = AttackProfile::from_attack(
        DepthReduceSet::GreedyDepth((td * size as f64) as usize, greed_params.clone()),
        size,
    );
    profile.runs = 3;
    profile.range.start = 0.03;
    profile.range.end = 0.04;
    profile.range.interval = 0.01;

    let res1 = attack_with_profile( spec, &profile);
    println!("json: {}",serde_json::to_string_pretty(&vec![res0,res1]).expect("can't serialize to json"));



}

fn baseline_greedy() {
    println!("Baseline computation for target size [0.10,0.20,0.30]");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 20;
    let size = (2 as usize).pow(n);
    let deg = 6;
    let target_depth = (0.001 * size as f64) as usize;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: DRGAlgo::MetaBucket(deg),
    };

    let mut greed_params = GreedyParams {
        k: GreedyParams::k_ratio(n as usize),
        radius: 4,
        reset: true,
        length: 10,
        iter_topk: true,
        use_degree: true,
        parallel: true,
    };
     
    let mut profile = AttackProfile::from_attack(
        DepthReduceSet::GreedyDepth(target_depth, greed_params.clone()),
        size,
    );
    profile.runs = 3;
    profile.range.start = 0.15;
    profile.range.end = 0.26;
    profile.range.interval = 0.05;


    let res1 = attack_with_profile(spec, &profile);

    let mut profile = AttackProfile::from_attack(
        DepthReduceSet::GreedySize(target_depth, greed_params.clone()),
        size,
    );
    profile.runs = 3;
    profile.range.start = 0.15;
    profile.range.end = 0.31;
    profile.range.interval = 0.05;
    let res2 = attack_with_profile(spec, &profile);

    let json = serde_json::to_string_pretty(&vec![res1,res2]).expect("can't serialize to json");
    println!("{}", json);
}

fn main() {
    pretty_env_logger::init_timed();
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        match args[1].to_lowercase().trim() {
            "greedy" => greedy_attacks(),
            "porep" => porep_comparison(),
            "baseline_greedy" => baseline_greedy(),
            "baseline_valiant" => baseline_valiant(),
            "theoretical" => theoretical_limit(),
            _ => panic!("command not understood: choose greedy or porep"),
        }
    } else {
        porep_comparison();
    }
}
