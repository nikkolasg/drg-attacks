# Results

Notation:
- S is the set of node to remove from the graph G (average)
- d is the longest depth found on the graph G - S (average)
- target is the desired size of either S or d for when the algorithm should stop

All values are given in proportion of the size of the graph.

## Valiant results

### Depth

We target a specific depth:
(target, |S|,d):
(0.15, 0.45, 0.09) -> (0.55n, 0.09n)-DRG
(0.20, 0.45, 0.09) -> (0.55n, 0.09n)-DRG
(0.25, 0.32, 0.22) -> (0.68n, 0.22n)-DRG

### Set Size

We target a specific set size:
(target, S,d):
(0.15, 0.17, 0.46) -> (0.83n, 0.46n)-DRG
(0.2, 0.32, 0.22) -> (0.68n, 0.22n)-DRG
(0.25, 0.32, 0.22) -> (0.68n, 0.22n)-DRG
(0.3, 0.32, 0.22) -> (0.68n, 0.22n)-DRG

Valiant is *unprecise* in targeting a specific depth since it adds set with
drastic size difference. The *target* depth is the depth indicated when the
algorithm should stop. The actual depth is often lower since the last steps of
the protocol add a lot to the set S.


