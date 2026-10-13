# Verifier vs A-J judge: checkpoints and metrics

Two 400-step runs of the recipe that differ only in the reward:

| Arm | Directory | Reward |
|---|---|---|
| Pairwise verifier | `verifier/` | `reward_mode=verifier`: 3 criteria x 4 slot-swapped repeats, round robin, A-T expected grade |
| Absolute A-J judge | `judge_aj3/` | `reward_mode=judge`: one written A-J letter per criterion on the same 3 criteria, averaged |

Both used `max_steps=400`, 8 prompts x 5 rollouts per step, and the same seed,
prompts and policy settings. Held-out: 50 prompts x 5 paintings per checkpoint,
graded by `moonshotai/Kimi-K2.6` on the three criteria (A-J, normalized to 0-1).

Each arm's directory holds:

| File | Content |
|---|---|
| `checkpoints.jsonl` | one row per saved checkpoint (every 5 steps, plus `final`): Tinker `state_path` (resume training) and `sampler_path` (sample), the Tinker run id, which API key env var owns it, and that checkpoint's held-out score, rendered-only score, render rate, best-of-5 and pass@5 |
| `config.json` | the run's full training config |
| `metrics.jsonl` | training metrics per step |
| `heldout_metrics.jsonl` | held-out metrics per evaluated checkpoint |

Checkpoints were saved with `ttl_seconds=null`, so they are kept on Tinker
indefinitely; load them with the API key named in `tinker_api_key_env`.
Pass@5 is the share of held-out prompts where at least one of the five
paintings averages 8/10 or better (0.78 on 0-1); best-of-5 is the mean over
prompts of the best painting's score. A painting that fails to render scores 0.

| Held-out (mean over checkpoints) | Verifier | A-J judge |
|---|---|---|
| Score, steps 150-400 | 0.660 | 0.627 |
| Score, rendered paintings only, steps 150-400 | 0.669 | 0.635 |
| Pass@5, steps 150-400 | 65.5% | 60.3% |
| Score at step 400 | 0.712 | 0.674 |

![held-out score, rendered paintings only](heldout_rendered_only.png)
