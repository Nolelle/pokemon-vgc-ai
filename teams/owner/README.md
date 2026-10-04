# Owner teams

Six Reg M-C teams supplied by the owner. Each has a Showdown export (`<name>.txt`, `EVs`
lines are Champions Stat Points) and a packed version (`<name>.packed.txt`, made with
`./pokemon-showdown pack-team` from the Showdown checkout, run under Node 22). All six pass
`./pokemon-showdown validate-team gen9championsvgc2026regmc` (exit 0, no output).

| Team | Pokepaste |
| --- | --- |
| `psyspam_sand` | https://pokepast.es/056e780a38dac900 |
| `gardevoir_psyspam` | https://pokepast.es/e6497a1a671dac90 |
| `hatterene_tr` | https://pokepast.es/7b073199fc857b04 |
| `coaching_baxcalibur` | https://pokepast.es/b4465e52a2df6d1e |
| `terrain_pulse_blastoise` | https://pokepast.es/81427a109e744097 |
| `salamence_tw` | https://pokepast.es/fe12c887687cf59c |

**LLM-harness testing starts with `psyspam_sand` and `salamence_tw`** (see
`docs/llm_test_protocol.md`); the other four are held in reserve.

Use the packed files with `offline/evaluate_own_spread_pool.py --our-teams`.
