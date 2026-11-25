# ID-GNN per-slice audit, rel-stack/post-post-related

MAP@100 x100.

| slice | users | ID-GNN | recent popularity |
|---|---|---|---|
| overall | 258 | 10.89 | 0.58 |
| repeat | 23 | 93.48 | 0.49 |
| cf_nbr | 6 | 6.02 | 6.25 |
| novel | 234 | 3.28 | 0.46 |
| cold_user | 166 | 3.14 | 0.62 |
| new_item | 78 | 2.71 | 0.00 |
| head10 | 97 | 14.05 | 2.14 |

## Mechanical scoring

- HOLDS — J1 reproduction within 1.0: MAP 10.89 vs published 10.83
- HOLDS — J4 novel MAP < half overall: novel 3.28 vs overall 10.89
