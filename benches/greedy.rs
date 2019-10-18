use criterion::{black_box, criterion_group, criterion_main, Criterion};
use drg::attacks::{count_paths, update_radius_set, GreedyParams};
use drg::graph::*;
use rand::Rng;

fn bench_update_radius(c: &mut Criterion) {
    let seed = rand::thread_rng().gen::<[u8; 32]>();
    let size = 10000;
    let deg = 4;
    let radius = 6; // points covered ~= 3^4
    let mut graph = Graph::new(size, seed, DRGAlgo::MetaBucket(deg));
    graph.children_project();
    let node = size / 2;
    let mut inradius = NodeSet::default();
    let mut p = GreedyParams {
        radius: radius,
        parallel: false,
        ..GreedyParams::default()
    };
    c.bench_function("update_radius sequential", |b| {
        b.iter(|| update_radius_set(&graph, node, &mut inradius, &p))
    });
    p.parallel = true;
    c.bench_function("update_radius parallel", |b| {
        b.iter(|| update_radius_set(&graph, node, &mut inradius, &p))
    });
}

fn bench_count_paths(c: &mut Criterion) {
    let seed = rand::thread_rng().gen::<[u8; 32]>();
    let size = (2 as u32).pow(16) as usize;
    let degree = 4;
    let graph = Graph::new(size, seed, DRGAlgo::MetaBucket(degree));
    let length = 10;
    let k = 400;
    let s = ExclusionSet::new(&graph);
    let mut p = GreedyParams {
        k: k,
        length: length,
        parallel: false,
        ..GreedyParams::default()
    };
    c.bench_function("count_paths sequential", |b| {
        b.iter(|| count_paths(&graph, &s, &p))
    });
    p.parallel = true;
    c.bench_function("count_paths parallel", |b| {
        b.iter(|| count_paths(&graph, &s, &p))
    });
}

criterion_group!(benches, bench_count_paths, bench_update_radius);
criterion_main!(benches);
