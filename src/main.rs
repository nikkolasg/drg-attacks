#![deny(warnings)]
mod attacks;
pub mod graph;
mod utils;
use attacks::{attack, attack_with_profile, AttackProfile, DepthReduceSet, GreedyParams};
use graph::{DRGAlgo, Graph, GraphSpec};
use rand::Rng;

#[macro_use]
#[cfg(test)]
extern crate lazy_static;
extern crate rayon;

use clap::{value_t, App, Arg, SubCommand};
#[cfg(feature = "cpu-profile")]
use gperftools::profiler::PROFILER;

/// Start profile (currently use for the Greedy attack) and dump the file in
/// the current directory. It can later be analyzed with `pprof`, e.g.,
/// ```text
/// cargo run --release --features cpu-profile  -- -n 14 greedy
/// REV=$(git rev-parse --short HEAD)
/// pprof --lines --dot target/release/drg-attacks greedy.profile > profile-$REV.dot && xdot profile-$REV.dot &
/// ```
#[cfg(feature = "cpu-profile")]
#[inline(always)]
fn start_profile(stage: &str) {
    PROFILER
        .lock()
        .unwrap()
        .start(format!("./{}.profile", stage))
        .unwrap();
}
#[cfg(feature = "cpu-profile")]
#[inline(always)]
fn stop_profile() {
    PROFILER.lock().unwrap().stop().unwrap();
}
#[cfg(not(feature = "cpu-profile"))]
#[inline(always)]
fn start_profile(_stage: &str) {}
#[cfg(not(feature = "cpu-profile"))]
#[inline(always)]
fn stop_profile() {}

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

fn greedy_attacks(n: usize) {
    println!("Greedy Attacks parameters");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let size = (2 as usize).pow(n as u32);
    let deg = 6;
    let target_size = (0.30 * size as f64) as usize;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: DRGAlgo::MetaBucket(deg),
    };
    let runs = 10;
    //attack(&mut g1, DepthReduceSet::ValiantDepth(depth));

    let greed_params = GreedyParams {
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
    profile.runs = runs;
    profile.range.start = 0.2;
    profile.range.end = 0.5;
    profile.range.interval = 0.1;

    start_profile("greedy");
    let res = attack_with_profile(spec, &profile);
    // FIXME: Turn this into a JSON output.
    println!("\n\n------------------");
    println!("Attack finished: {:?}", profile);
    stop_profile();
    let json = serde_json::to_string_pretty(&res).expect("can't serialize to json");
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


    let greed_params = GreedyParams {
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

    let greed_params = GreedyParams {
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
    let matches = App::new("DRG Attacks")
        .version("1.0")
        .arg(
            Arg::with_name("size")
                .short("n")
                .long("size-n")
                .help("Size of graph expressed as a power of 2")
                .default_value("10")
                .takes_value(true),
        )
        .subcommand(SubCommand::with_name("greedy").about("Greedy attack"))
        .subcommand(SubCommand::with_name("porep"))
        .subcommand(SubCommand::with_name("baseline_greedy"))
        .subcommand(SubCommand::with_name("baseline_valiant"))
        .subcommand(SubCommand::with_name("theory"))
        .get_matches();

    let n = value_t!(matches, "size", usize).unwrap();
    assert!(n < 50, "graph size is too big (2^{})", n);
    // FIXME: Use this argument for all attacks, not just Greedy (different
    // attacks may use different default values).

    if let Some(_) = matches.subcommand_matches("greedy") {
        greedy_attacks(n);
    } else if let Some(_) = matches.subcommand_matches("porep") {
        porep_comparison();
    } else if let Some(_) = matches.subcommand_matches("theory") {
        theoretical_limit();
    } else if let Some(_) = matches.subcommand_matches("baseline_greedy") {
        baseline_greedy();
    } else if let Some(_) = matches.subcommand_matches("baseline_valiant") {
        baseline_valiant();
    } else {
        eprintln!("No subcommand entered, running `porep_comparison`");
        porep_comparison();
    }
    // FIXME: Can this be structured with a `match`?
}
