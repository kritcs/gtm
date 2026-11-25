# ID-GNN per-slice audit, rel-stack/user-post-comment

MAP@100 x100.

| slice | users | ID-GNN | recent popularity |
|---|---|---|---|
| overall | 758 | 13.38 | 0.02 |
| repeat | 132 | 31.77 | 0.10 |
| cf_nbr | 16 | 12.41 | 0.00 |
| novel | 681 | 9.50 | 0.00 |
| cold_user | 138 | 17.69 | 0.00 |
| new_item | 354 | 11.52 | 0.00 |
| head10 | 94 | 4.82 | 0.15 |

## Mechanical scoring

- HOLDS — J1 reproduction within 1.0: MAP 13.38 vs published 12.72
- FAILS — J4 novel MAP < half overall: novel 9.50 vs overall 13.38
- HOLDS — J3 novel MAP >= 1.0 and > 2x best heuristic: novel 9.50 vs repeat_then_recent_pop 0.00
