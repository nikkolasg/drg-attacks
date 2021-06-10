[![CircleCI](https://circleci.com/gh/filecoin-project/drg-attacks.svg?style=svg)](https://circleci.com/gh/filecoin-project/drg-attacks)

# Attacks on DRG Graph 

This repository is a library for generating *Depth Robust Graphs* as well
contains code for *attacking* those graphs. This repository holds as well the
code to generate results based on the attacks and is the baseline for the
challenge.

## Depth Robust Graphs

A depth robust graph is a directed graph where the label of a node depends on
its parent and with the following property:
> After removing any subset of up to e nodes (and adjacent edges) there remains
> a directed path of length d.

One usage of these graphs is to enforce a long sequential computation even
against adversarial behavior thanks to this directed path.

You can find the code to generate multiple kind of such graphs in `src/graph.rs`
file.

## Attacks

The theoretical bounds for the DRG properties are unfortunately very low, and
can't be used to derive practical applications of DRGs. There are
empirical attacks that thrive to find the smallest set of nodes to remove to
reach a given depth (or find the smallest depth for a given size of nodes to
remove): these attacks are called *depth-reducing set* attacks.

You can find multiple implementations of different such attacks in
`src/attacks.rs`. In particular there are two general "kinds" of attacks
implemented:
- **Valiant based attacks**: these attacks rely on the Valiant Lemma to iteratively
  construct the set of nodes to remove which results in an graph which has
  half of the size at each steps. 
- **Greedy attacks**: these attacks are trying to find the best set of nodes
  possible according to some heuristics.

## Resource

The most comprehensive resource is from Blocki et al. from 2019: [paper](https://eprint.iacr.org/2018/944.pdf)
Alwen et al. first showed the DRSample algorithm and the valiant based depth
reducing attacks in this [CCS paper](https://eprint.iacr.org/2018/944.pdf)
