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

### MetaBucket construction

+ Difference between r2 and drg-attacks are two folds:
    1. different index as in DRSample
    2. different way to compute the final value

For 2. , r2 does `let out = (node * m + k - back_dist) / m;` 
drg-attacks 
```
let meta_parent = node * m - rng.gen_range(min, max); // no "+ k" here
let real_parent = meta_parent / degree; 
```

Reason for drg-attacks is because it always choose the minimum index for the
"bucket of meta nodes". For example, with degree = 3, node 2 gets mapped to
"{6,7,8}", so to find the parent for node 2, I can always take the meta index
6 so that I'll always find a number below 6. In r2, for the third iteration,
the meta index would be 8 so I may end up selecting the meta-node 8 - 2 = 6, 
which corresponds to the same index in the basic graph !

## Greedy Attacks

+ Should we reset the `inradius` set after one iteration of the loop or not ?
+ SHould we *always* add the first node of the "top k" ? or only if all of them
    are in the radius, then only then, we add the first one ? Why not a random one


## Comparisons

In the [porep paper](https://web.stanford.edu/~bfisch/porep_short.pdf), figure
4.1 shows the different configurations and results of the attacks are shown.

We see for example, that for n=2^20, with degree = 6 (m = 5), we have different
points.  One point used to characterize the depth robustness of the graph is 
(e ~= 0.32 * 10^6, d ~= 2.7 * 10^5).

+ Compare for the same e, the depth you find, and for the same depth, the length
  of e you find.
