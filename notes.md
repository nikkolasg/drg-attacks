# Notes from implementation

## Bucket Sample

### Modulo vs Division

In the porep [paper](https://web.stanford.edu/~bfisch/porep_short.pdf), the
bucket sampling algorithm uses the modular reduction to create an edge whose
endpoints fits into `[0, n[`. The problem is that the modular reduction does not
guarantee the inequality `i < j` for an edge `(i,j)`.
This needs to be an integer division (taking the floor result) instead.

### Logarithm floor

When using the floor(logarithm2(idx)) to derive the bucket index, we are losing
some potential nodes. Example:
```
1. node = 4
   floor(log2(node)) = 2
   range = [2, 4[

2. node = 5
   floor(log2(node)) = 2
   range = [2, 4[  // should it be [2, 5[ here ?

3. node = 6 
   floor(log2(node)) = 2
   range = [2, 4] // should it be [2, 6[ here ?

   ... 
```

UPDATE: the DRSample algorithm in https://acmccs.github.io/papers/p1001-alwenA.pdf 
uses `log2().floor() + 1` to AVOID that scenario. Implementation is now following 
that algorithm instead of the one in the porep paper.

### unproportionate chances

see: https://github.com/filecoin-project/research/issues/136
It happens within my implementation and exhibits strongly with 1000 nodes
already.
```
add_direct node 0: at most 0 parents and 771 children
```
means the first has 771 children !

### Bucket Sample porep vs CCS vs code

+ For the bucket sample algorithm, so far the code does a mix: it compute the random bucket and random index like the DRSample algo in the CCS paper (which is DIFFERENT than the regular algo in the porep paper), but uses the same general structure as the porep paper algorithm BucketSample.

### DRSample Bucket construction

+ in the paper, we want to select parent v such that 2^(i-1) <= dist(u,v) <= 2^i
with i >= 2, so that means we have dist(u,v) >= 2 always, that means we can't
have u == v or u == v-1  -> as shown in the last line of the algo:
r <-- [max(g/2, 2), g] so r >= 2, so u - r != u

+ in C# code:
```c#
// floor(log(i))
int logi = (int)Math.Floor(Math.Log(i, 2.0));
// j <- [0, floor(log(i)) - 1]
int j = r.Next() % logi;
int jj = Math.Min(i, 1 << (j + 1));

int backDist = r.Next(Math.Max(jj >> 1, 2), jj + 1);
DAG[i] = i - backDist;
```
which is DIFFERENT than the pseudo code in the CCS paper since it uses a
different upper bound for selecting the bucket index:
    - ccs: j <-- [1, floor(log(i)) + 1]
    - c# : j <-- [0, floor(log(i)) - 1] // because of modulo

### Difference between r2 and drg-attacks

Difference between r2 and drg-attacks are two folds:
    1. different way to compute the bucket index
    2. different way to compute the final value


#### Bucket Index

+ In the paper algorithm (DRSample, Algorithm 1 in CCS [paper](https://acmccs.github.io/papers/p1001-alwenA.pdf), we have:
```
g <-- [1,floor(log2(v))+1]
```

+ In r2, we have :
```rust
let logi = ((node * m) as f32).log2().floor() as usize;
let j = rng.gen::<usize>() % logi;
```
=> r2 uses the floor of the logarithm without doing a +1 when selecting the
value. It gives rises to high differences in bucket index, as show below:
```
v = 18, log2(v) = 4.16... 
// DRSample
floor(log2(v)) = 4
g <-- [1, 5]
// R2
g <-- [1, 4[ <=> [1, 3]
```

+ In drg-attacks, we have:
```rust
let meta_idx = node * m;
// ceil instead of floor() + 1
let max_bucket = (meta_idx as f32).log2().ceil() as usize;
let i: usize = rng.gen_range(1, max_bucket + 1);
```
That follows the paper algorithm.

#### Final Index 

+ In the paper algorith, we have:
```
g <-- min(v, 2^g')
r <-- [max(g/2, 2) , g]
return v - r
```

+ In r2 we have
```
// min(v, 2^g')
let jj = cmp::min(node * m + k, 1 << (j + 1));
// r <-- [ max(jj/2, 2), jj ]
let back_dist = rng.gen_range(cmp::max(jj >> 1, 2), jj + 1);
// v - r
let out = (node * m + k - back_dist) / m;
```

+ In drg-attacks we have:
```
// g = min(v, 2^g')
let max = std::cmp::min(meta_idx, 1 << i);
// max(2, g / 2)
let min = std::cmp::max(2, max >> 1);
// return v - [ max(2, g/2), g ]
let meta_parent = node * m - rng.gen_range(min, max); // no "+ k" here
let real_parent = meta_parent / degree; 
```

The reason for drg-attacks to NOT include the "+k" is because it prevents
falling on the same "real" node when we divide by the degree. Basically,
drg-attacks always choose the minimum index for the "bucket of meta nodes".  For
example, with degree = 3, node 2 gets mapped to "{6,7,8}", so to find the parent
for node 2, I can take 3 times the meta index 6 so that I'll always find a number
below 6. In r2, for the third iteration, the meta index would be 8 so I may end
up selecting the meta-node 8 - 2 = 6, which corresponds to the same index in the
basic graph ! From what I understand of the paper algorithm, this should not be
possible and is not desirable: for a given child u, we always select a parent
node such that 2^(i-1) <= dist(u,v) <= 2^i , so dist(u,v) > 1. In the code they
even go further since the minimum to choose r is 2, so v - r <= v - 2 always.

## Greedy Attacks

+ Should we reset the `inradius` set after one iteration of the loop or not ?
+ SHould we *always* add the first node of the "top k" ? or only if all of them
    are in the radius, then only then, we add the first one ? Why not a random one

### Observations

+ Changing k doesn't seem to improve results but improve computation time when
  reducing k

## Results

// degree = 6, 2^24 -> (0.30n, 0.24n)
graph stats: size=16777216, min parents=1, max children=25
Trial #1 with target depth = 0.25n = 4194304
Attack with ValiantDepth(4194304)
        -> |S| = 4876295 = 0.2906n
        -> depth(G-S) = 4072751 = 0.2428n
        -> time elapsed: 862.685810048s


graph created and saved at porep_n24_d2.json
graph stats: size=16777216, min parents=2, max children=11
Trial #1 with target depth = 0.25n = 4194304
Attack with ValiantDepth(4194304)
        -> |S| = 1361719 = 0.0812n
        -> depth(G-S) = 3806177 = 0.2269n
        -> time elapsed: 319.69523059s


### Greedy attacks

Greedy attacks tests with size = 4096, depth(G-S) <= 1024
Attack with Greedy(1024, GreedyParams { k: 10, radius: 3, length: 5, reset: true, iter_t
opk: true })
-> |S| = 1050 = 0.2563n
        -> depth(G-S) = 1007 = 0.2458n
        -> time elapsed: 19.119084952s
Attack with Greedy(1024, GreedyParams { k: 10, radius: 3, length: 5, reset: true, iter_t
opk: false })
        -> |S| = 1079 = 0.2634n
        -> depth(G-S) = 998 = 0.2437n
        -> time elapsed: 38.19792473s


---
master - "latest tests" 
graph created and saved at porep_n20_d5.json
graph stats: size=1048576, min parents=1, max children=20
Trial #1 with target depth = 0.25n = 262144
Attack with ValiantDepth(262144)
	-> |S| = 295265 = 0.2816n
	-> depth(G-S) = 227680 = 0.2171n
	-> time elapsed: 35.231161026s
Trial #4 with Greedy DRS, target depth = 0.25n = 262144

graph created and saved at porep_n20_d6.json
graph stats: size=1048576, min parents=1, max children=22
Trial #1 with target depth = 0.25n = 262144
Attack with ValiantDepth(262144)
	-> |S| = 343263 = 0.3274n
	-> depth(G-S) = 242762 = 0.2315n
	-> time elapsed: 46.823593713s
---------------------
Greedy attacks tests with size = 1048576, depth(G-S) <= 262144
Attack with Greedy(262144, GreedyParams { k: 800, radius: 4, length: 8, reset: true, iter_topk: true, use_degree: false })
	-> |S| = 235200 = 0.2243n
	-> depth(G-S) = 261760 = 0.2496n
	-> time elapsed: 45369.448506162s

Attack with Greedy(262144, GreedyParams { k: 800, radius: 4, length: 8, reset: true, iter_topk: true, use_degree: true })
	-> |S| = 251200 = 0.2396n
	-> depth(G-S) = 260755 = 0.2487n
	-> time elapsed: 20630.204970272s


## Comparisons

In the [porep paper](https://web.stanford.edu/~bfisch/porep_short.pdf), figure
4.1 shows the different configurations and results of the attacks are shown.

We see for example, that for n=2^20, with degree = 6 (m = 5), we have different
points.  One point used to characterize the depth robustness of the graph is 
(e ~= 0.32 * 10^6, d ~= 2.7 * 10^5).

+ Compare for the same e, the depth you find, and for the same depth, the length
  of e you find.



### Optimization to compute incidents paths

```rust

    ntr = node to remove
    for parents in node.parents() {
        for d in 0..d {
            starting_paths[parent][d] -= starting_paths[ntr][d-1]
        }
    }
    for child in node.children() {
        for d in 0..d {
            ending_paths[child][d] -= ending_paths[ntr][d-1]
        }
    }

// SUM (0 -> length) (start[n][d] - (x1 = SUM(ctr: children TO REMOVE) start[ctr][d-1])
                \* (end[n][d] - (y1 = SUM(ptr: parents TO REMOVE) end[ptr][d-1]))

// from A = a1*b1 + a2*b2 + ... TO
// ([a1 - x1]* [b1 - y1]) + ([a2 - x2]* [b2 - y2]) * ...
// (a1*b1 - a1*y1 -x1*b1 - x1*y1) + (...)
// A - (x1(b1+y1) + a1) - (x2(b2 + y2) + a2) - .... 
// A - 
    
--------
    for d in 1..=length {
        g.for_each_edge(|e| {
            // checking each parents (vs only checking direct + 1parent in C#)
            // no ending path for node i if the parent is contained in S
            // since G - S doesn't have this parent
            if !s.contains(&e.parent) {
                ending_paths[e.child][d] += ending_paths[e.parent][d - 1];

                // difference vs the pseudo code: like in C#, increase parent count
                // instead of iterating over children of node i
                starting_paths[e.parent][d] += starting_paths[e.child][d - 1];
            }
        });
    }

    let mut incidents = Vec::with_capacity(g.size());
    g.for_each_node(|&node| {
        incidents.push(Pair(
            node,
            (0..=length)
                .map(|d| (starting_paths[node][d] * ending_paths[node][length - d]) as usize)
                .sum(),
        ));
    });
```
<<<<<<< HEAD
--------
Baseline VALIANT computation for target size [0.10,0.20,0.30]
Attack (run 0) target (Depth(0.2999992370605469) = 0.15), with ValiantDepth(157286)
        -> |S| = 0.4604
        -> depth(G-S) = 0.1067
        -> time elapsed: 35.568549116s
Attack (run 0) target (Depth(0.2999992370605469) = 0.18), with ValiantDepth(188743)
        -> |S| = 0.4604
        -> depth(G-S) = 0.1067
        -> time elapsed: 33.784741412s
Attack (run 0) target (Depth(0.2999992370605469) = 0.21), with ValiantDepth(220200)
        -> |S| = 0.4604
        -> depth(G-S) = 0.1067
        -> time elapsed: 34.424781515s
Attack (run 0) target (Depth(0.2999992370605469) = 0.24), with ValiantDepth(251658)
        -> |S| = 0.3308
        -> depth(G-S) = 0.2293
        -> time elapsed: 30.378717294s
Attack (run 1) target (Depth(0.2999992370605469) = 0.15), with ValiantDepth(157286)
        -> |S| = 0.4563
        -> depth(G-S) = 0.1060
        -> time elapsed: 34.801943912s
Attack (run 1) target (Depth(0.2999992370605469) = 0.18), with ValiantDepth(188743)
        -> |S| = 0.4563
        -> depth(G-S) = 0.1060
        -> time elapsed: 33.041768581s
Attack (run 1) target (Depth(0.2999992370605469) = 0.21), with ValiantDepth(220200)
        -> |S| = 0.4563
        -> depth(G-S) = 0.1060
        -> time elapsed: 32.827424262s
Attack (run 1) target (Depth(0.2999992370605469) = 0.24), with ValiantDepth(251658)
        -> |S| = 0.3265
        -> depth(G-S) = 0.2226
        -> time elapsed: 30.043105239s
Attack (run 2) target (Depth(0.2999992370605469) = 0.15), with ValiantDepth(157286)
        -> |S| = 0.4604
        -> depth(G-S) = 0.0986
        -> time elapsed: 32.90691631s
Attack (run 2) target (Depth(0.2999992370605469) = 0.18), with ValiantDepth(188743)
        -> |S| = 0.4604
        -> depth(G-S) = 0.0986
        -> time elapsed: 32.762877913s
Attack (run 2) target (Depth(0.2999992370605469) = 0.21), with ValiantDepth(220200)
        -> |S| = 0.4604
        -> depth(G-S) = 0.0986
        -> time elapsed: 32.865584661s
Attack (run 2) target (Depth(0.2999992370605469) = 0.24), with ValiantDepth(251658)
        -> |S| = 0.3279
        -> depth(G-S) = 0.2236
        -> time elapsed: 29.035486593s


------------------
Depth Attack finished: AttackProfile { runs: 3, target: Depth(0.2999992370605469),
range: TargetRange { start: 0.15, interval: 0.03, end: 0.25 }, attack: ValiantDepth
(314572) }
{
  "results": [
    {
      "target": 0.15,
      "mean_depth": 0.10377883911132812,
      "mean_size": 0.4590330123901367
    },
    {
      "target": 0.18,
      "mean_depth": 0.10377883911132812,
      "mean_size": 0.4590330123901367
    },
    {
      "target": 0.21,
      "mean_depth": 0.10377883911132812,
      "mean_size": 0.4590330123901367
    },
    {
      "target": 0.24,
      "mean_depth": 0.22513930002848306,
      "mean_size": 0.3283672332763672
    }
  ],
  "attack": {
    "ValiantDepth": 314572
  }
}


------------------
Attack (run 0) target (Size(0.2999992370605469) = 0.1), with ValiantSize(104857)
        -> |S| = 0.1775
        -> depth(G-S) = 0.4748
        -> time elapsed: 16.255118889s
Attack (run 0) target (Size(0.2999992370605469) = 0.2), with ValiantSize(209715)
        -> |S| = 0.3308
        -> depth(G-S) = 0.2293
        -> time elapsed: 15.239742931s
Attack (run 0) target (Size(0.2999992370605469) = 0.30000000000000004), with Valian
tSize(314572)
        -> |S| = 0.3308
        -> depth(G-S) = 0.2293
        -> time elapsed: 15.072645141s
Attack (run 1) target (Size(0.2999992370605469) = 0.1), with ValiantSize(104857)
        -> |S| = 0.1776
        -> depth(G-S) = 0.4627
        -> time elapsed: 15.269347989s
Attack (run 1) target (Size(0.2999992370605469) = 0.2), with ValiantSize(209715)
        -> |S| = 0.3265
        -> depth(G-S) = 0.2226
        -> time elapsed: 15.31680873s
Attack (run 1) target (Size(0.2999992370605469) = 0.30000000000000004), with Valian
tSize(314572)
        -> |S| = 0.3265
        -> depth(G-S) = 0.2226
        -> time elapsed: 15.168382539s
Attack (run 2) target (Size(0.2999992370605469) = 0.1), with ValiantSize(104857)
        -> |S| = 0.1771
        -> depth(G-S) = 0.4817
        -> time elapsed: 14.704309067s
Attack (run 2) target (Size(0.2999992370605469) = 0.2), with ValiantSize(209715)
        -> |S| = 0.3279
        -> depth(G-S) = 0.2236
        -> time elapsed: 15.124037994s
Attack (run 2) target (Size(0.2999992370605469) = 0.30000000000000004), with Valian
tSize(314572)
        -> |S| = 0.3279
        -> depth(G-S) = 0.2236
        -> time elapsed: 14.787900706s


------------------
Size Attack finished: AttackProfile { runs: 3, target: Size(0.2999992370605469), ra
nge: TargetRange { start: 0.1, interval: 0.1, end: 0.31 }, attack: ValiantSize(3145
72) }
{
  "results": [
    {
      "target": 0.1,
      "mean_depth": 0.4730873107910156,
      "mean_size": 0.1774123509724935
    },
    {
      "target": 0.2,
      "mean_depth": 0.22513930002848306,
      "mean_size": 0.3283672332763672
    },
    {
      "target": 0.30000000000000004,
      "mean_depth": 0.22513930002848306,
      "mean_size": 0.3283672332763672
    }
  ],
  "attack": {
    "ValiantSize": 314572
  }
}
=======

{
    "spec": {
      "size": 1048576,
      "algo": {
        "MetaBucket": 6
      }
    },
    "runs": 3,
    "attack": {
      "GreedySize": [
        1048,
        {
          "k": 800,
          "radius": 4,
          "length": 10,
          "parallel": true,
          "reset": true,
          "iter_topk": true,
          "use_degree": true
        }
      ]
    },
    "results": [
      {
        "target": 0.15,
        "mean_depth": 0.4692646662394206,
        "mean_size": 0.150299072265625
      },
      {
        "target": 0.2,
        "mean_depth": 0.33088525136311847,
        "mean_size": 0.200653076171875
      },
      {
        "target": 0.25,
        "mean_depth": 0.21710904439290366,
        "mean_size": 0.250244140625
      },
      {
        "target": 0.3,
        "mean_depth": 0.12705866495768228,
        "mean_size": 0.30059814453125
      }
    ]
  }

>>>>>>> master
