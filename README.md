# paint_rl: RL for painting with code, rewarded by a multimodal LLM-as-a-Verifier

`Inkling-Small` learns to paint watercolors by writing
[p5.brush](https://github.com/acamposuribe/p5.brush) sketches. Each sketch is rendered
headlessly to a PNG. The paintings in a rollout group are compared against each other by
a multimodal verifier, and group-centred rewards update the policy on Tinker. The task
follows ["Training AI to Paint with Code"](https://surya.website/rling-qwen-to-paint-with-code);
the reward follows [LLM-as-a-Verifier](https://github.com/llm-as-a-verifier/llm-as-a-verifier).

![paintings early to late](results/examples_early_to_late.png)

*Two training prompts as they came round during the run, and one held-out prompt (never
trained on) at three checkpoints. Each cell is the best of that step's five paintings by
Gemini score.*

## Models

| Role | Model | How |
|---|---|---|
| Policy | `thinkingmachines/Inkling-Small` | LoRA rank 32 on Tinker, `tml_v0` renderer, thinking effort 0.7, 8k-token budget |
| Training reward | `thinkingmachines/Inkling-Small` | Self-verification with `llm_verifier.compare` over Tinker's OpenAI-compatible endpoint, thinking effort 0.2, token logprobs |
| Held-out evaluator | `gemini-3.8-flash` | Absolute 0-10 scores on held-out prompts; never part of the reward |

## Quickstart

```bash
git clone https://github.com/jackyk02/paint_tinker.git && cd paint_tinker
cp .env.example .env && $EDITOR .env   # TINKER_API_KEY, GEMINI_API_KEY
set -a && . ./.env && set +a
./setup.sh                             # vendored cookbook + paint-rl extra + Chromium

# Train
python -m tinker_cookbook.recipes.paint_rl.train log_path=/tmp/paint_rl/verifier

# In a second shell: evaluate every checkpoint on the held-out prompts as it appears
python -m tinker_cookbook.recipes.paint_rl.eval_checkpoints log_path=/tmp/paint_rl/verifier
```

Every knob is in
[`train.py`](tinker-cookbook/tinker_cookbook/recipes/paint_rl/train.py); the recipe's
[README](tinker-cookbook/tinker_cookbook/recipes/paint_rl/README.md) covers the design.

## Defaults

| | |
|---|---|
| Steps | 300, each 8 prompts x 5 rollouts |
| Optimiser | importance-sampling policy gradient, group-centred advantages, lr 4e-5 |
| Reward | `0.05 * compiled + 0.05 * length_ok + 0.90 * score` |
| Verifier | 3 criteria x 2 slot-swapped repeats, full round robin: 60 calls per group, 480 in flight |
| Prompts | 256 training, 16 held-out (8 novel-subject, 8 novel-combination) |
| Checkpoints | every 5 steps, kept on Tinker indefinitely |
| Held-out evaluation | separate process, base model + every checkpoint, Gemini only |

## How it works

1. **Prompt.** For example, `Paint a teal heron standing in shallow water in wet-on-wet
   watercolor wash.` The pool is 48 subjects x 8 colours x 4 styles, drawn from what
   watercolorists commonly paint: twelve flowers (sunflower, lotus on a still pond,
   wisteria over a garden gate), twelve animals, twelve still-life objects (teapot, cup of
   coffee, bowl of lemons) and twelve scenes (lighthouse, mountain lake at dawn, village
   street under a crescent moon), each family spanning three difficulty tiers. The 256
   training prompts are balanced over subjects, colours and styles. The 16 held-out prompts
   are 8 **novel subjects** (one held-out subject per family, never trained on in any form)
   and 8 **novel combinations** of subjects, colours and styles seen in training.
   `prompts_test.py` pins the split.
2. **Sketch.** The policy replies with one `javascript` block that defines `setup()` on a
   512x512 WEBGL canvas, using only an allow-list of about 20 brush calls.
3. **Render.** Headless Chromium with software WebGL and pinned p5 1.11.3 / p5.brush 1.1.4.
   A sketch *compiles* if it throws no error, makes at least three distinct `brush.*`
   drawing calls, and paints a non-blank canvas.
4. **Verify.** Every pair of compiled paintings in a group is compared on one criterion at
   a time (prompt adherence, watercolor technique, composition), twice with the two slots
   swapped so position bias cancels. The verifier grades each painting A-T, and the score
   is the *expected* grade under its token probabilities, not the sampled letter. A
   painting's score is its mean over the comparisons it took part in; sketches that did
   not compile score 0 and sit out.
5. **Update.** Rewards are centred within each group and fed to an importance-sampling
   policy gradient (`tinker_cookbook/rl/train.py`).

## Results

One run with the defaults above, extended past 300 steps (`max_steps=1000`, still
training); numbers are at step 470. Held-out: 16 prompts never trained on x 5 paintings
per checkpoint, scored by Gemini, which never sees training. Gemini gives each painting an
integer 0-10 on each of the three criteria; a painting's score is the mean of the three
(hence 4.7, 6.3), and the curves average 80 paintings per checkpoint.

![held-out curves](results/heldout_curves.png)

![held-out prompts across checkpoints](results/heldout_checkpoints.png)

*Two held-out prompts at six checkpoints. Each cell is the best of that checkpoint's five
paintings by Gemini score, with the mean of all five below it.*

| Held-out | step 0 (base) | step 470 |
|---|---|---|
| Gemini score, all paintings (failed sketch = 0) | 3.24 / 10 | 5.35 / 10 |
| Gemini score, compiled paintings only | 4.25 / 10 | 5.35 / 10 |
| Compile rate | 76% | 100% |
| Novel subjects / novel combinations | 3.30 / 3.17 | 5.36 / 5.35 |
| Tier 1 / 2 / 3 (single subject / in a setting / two elements) | 3.61 / 3.30 / 2.67 | 5.59 / 5.46 / 4.88 |

- **Quality kept rising after compiling was solved.** The compile rate reaches ~100% by
  step 100; the compiled-only Gemini score is flat until then and keeps climbing after it
  (4.25 -> 5.35), so the later gain is painting quality, not fewer failures.
- **It generalises to unseen subjects.** Novel-subject prompts improve as much as novel
  combinations of seen ones.
- **The verifier stayed healthy.** 0% of verifier calls failed over the run. Its own grade
  of compiled paintings rose from 0.75 to 0.91 (first vs last 10 steps), moving in the same
  direction as Gemini.
- **Sketches got longer.** Mean code length grew from ~1.7k to ~6.7k characters (the
  length gate allows up to 8k), and a step now takes ~5 min instead of ~2. Policy entropy
  fell from 0.39 to 0.26.

`results/run/` holds the run's `config.json`, training `metrics.jsonl` and
`heldout_metrics.jsonl`.

## Outputs

Under `<log_path>/`:

| Path | Content |
|---|---|
| `metrics.jsonl` | training metrics per step |
| `checkpoints.jsonl` | every saved checkpoint, with its Tinker `sampler_path` |
| `paintings/train/step_XXXX/<prompt id>/` | each rollout's `.png`, `.js` and raw response, plus `group.json` with every verifier score |
| `paintings/index.jsonl` | one row per rollout: reward, score, compile result, paths |
| `iteration_XXXXXX/` | the cookbook's per-step HTML report and rollout summaries |
| `heldout_eval/metrics.jsonl` | held-out metrics per evaluated checkpoint; headline `test/env/all/eval_strong/score` |
| `heldout_eval/paintings/` | the held-out paintings, filed by checkpoint step |

Judge progress by the held-out Gemini score: the training reward is relative to the other
paintings in a group, so it is not comparable across steps. `gallery.py`, `progression.py`
and `compare_runs.py` turn the outputs into an HTML gallery, per-step best-painting sheets
and comparison curves.

## Repository layout

| Path | Content |
|---|---|
| `tinker-cookbook/` | [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook), vendored from upstream `e6fa1cc`, with the recipe and a `paint-rl` dependency extra added |
| `tinker-cookbook/tinker_cookbook/recipes/paint_rl/` | the recipe |
| `setup.sh` | installs the vendored cookbook with the `paint-rl` extra (`llm-verifier`, `openai`, `google-genai`, `playwright`) and Chromium |
| `.env.example` | the credentials the run needs |
| `results/` | figures, metrics and config of the run above |
