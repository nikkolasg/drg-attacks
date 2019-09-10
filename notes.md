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
