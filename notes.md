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
that algorithm.

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

