"""Test-time best-of-n: sample n paintings, keep the one the verifier ranks first.

    # Paint your own requests with a trained checkpoint
    python -m tinker_cookbook.recipes.paint_rl.best_of_n \\
        model_path=tinker://<run id>/sampler_weights/000400 \\
        "prompts.0=Paint a red fox curled up in the snow, in loose watercolor washes." \\
        out_dir=/tmp/paint_rl/best_of_n

    # ... or the run's 50 held-out prompts
    python -m tinker_cookbook.recipes.paint_rl.best_of_n \\
        model_path=tinker://<run id>/sampler_weights/000400 heldout=True out_dir=...

    # Score the selection rule on paintings a held-out evaluation already made
    # (no sampling): pick from each prompt's saved paintings and report the
    # Kimi score of the pick against the average painting and the best one.
    python -m tinker_cookbook.recipes.paint_rl.best_of_n \\
        from_heldout=/tmp/paint_rl/verifier/heldout_eval/paintings/test/step_0400 out_dir=...

The trained policy (``model_path``, a LoRA checkpoint of Inkling-Small)
samples ``n`` sketches per request, each is rendered headlessly, and the
paintings that render are ranked by the *base* Inkling-Small as a pairwise
verifier: the same round-robin tournament on the same three criteria that
rewarded training (``VerifierConfig``). Sketches that fail to render are never
picked; a request where none renders has no pick.

Under ``out_dir/<request id>/``: every candidate's ``.js`` and ``.png``,
``best.png``, and ``result.json`` with the candidates' verifier scores.
``out_dir/results.jsonl`` has one row per request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import chz
import tinker
from PIL import Image

from tinker_cookbook.recipes.paint_rl.prompts import build_prompt_splits, system_prompt
from tinker_cookbook.recipes.paint_rl.render import RenderResult, extract_code, get_shared_renderer
from tinker_cookbook.recipes.paint_rl.verifier import VerifierConfig, get_shared_scorer
from tinker_cookbook.renderers import Message, get_renderer, get_text_content
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.git_rev import recipe_user_metadata

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    # Painting requests, in plain text: prompts.0=... prompts.1=... (a request
    # may contain commas).
    prompts: tuple[str, ...] = ()
    # Also paint the run's 50 held-out prompts (the training split's seed and sizes).
    heldout: bool = False
    # The trained policy: a Tinker sampler checkpoint of base_model.
    model_path: str | None = None
    base_model: str = "thinkingmachines/Inkling-Small"
    renderer_name: str = "tml_v0"
    n: int = 5
    # Sampling as in training.
    policy_effort: float = 0.7
    max_tokens: int = 8192
    temperature: float = 1.0
    canvas_size: int = 512
    render_concurrency: int = 40
    render_timeout_s: float = 120.0
    # The selector: base Inkling-Small as the pairwise verifier, as in training.
    verifier: VerifierConfig = chz.field(default_factory=VerifierConfig)
    # Instead of sampling, select among the paintings a held-out evaluation
    # saved for one checkpoint (its ``.../paintings/test/step_XXXX`` folder).
    from_heldout: str | None = None
    out_dir: str = "/tmp/paint_rl/best_of_n"
    seed: int = 0
    base_url: str | None = None


@dataclass
class Candidate:
    index: int
    code: str | None
    render: RenderResult | None
    truncated: bool
    # Set when the candidate comes from a held-out evaluation: Kimi's score.
    eval_score: float | None = None

    @property
    def ok(self) -> bool:
        return self.render is not None and self.render.ok


def request_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]


async def select(
    cfg: Config, problem: str, candidates: list[Candidate]
) -> tuple[int | None, list[float | None]]:
    """Index of the verifier's pick among the candidates that rendered, and every
    candidate's tournament score (``None`` for one that did not render)."""
    ok: list[Candidate] = []
    images: list[Image.Image] = []
    for c in candidates:
        image = c.render.image() if c.render is not None and c.render.ok else None
        if image is not None:
            ok.append(c)
            images.append(image)
    scores: list[float | None] = [None] * len(candidates)
    if not ok:
        return None, scores
    group = await get_shared_scorer(cfg.verifier).score_group(problem, images, seed=cfg.seed)
    for c, s in zip(ok, group.scores, strict=True):
        scores[c.index] = s
    # Ties go to the earliest sample.
    best = max(ok, key=lambda c: (scores[c.index], -c.index))
    return best.index, scores


class BestOfN:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.renderer = get_renderer(cfg.renderer_name, get_tokenizer(cfg.base_model))
        self.system = system_prompt(cfg.canvas_size)
        self._sampler: tinker.SamplingClient | None = None

    async def sampler(self) -> tinker.SamplingClient:
        if self._sampler is None:
            if self.cfg.model_path is None:
                raise ValueError("model_path is required to sample (or pass from_heldout=...)")
            service = tinker.ServiceClient(
                base_url=self.cfg.base_url,
                user_metadata=recipe_user_metadata("recipe_paint_rl_best_of_n"),
            )
            self._sampler = await service.create_sampling_client_async(
                model_path=self.cfg.model_path
            )
        return self._sampler

    async def sample(self, problem: str) -> list[Candidate]:
        messages = [
            Message(role="system", content=self.system),
            Message(role="user", content=problem),
        ]
        if isinstance(self.renderer, TmlV0Renderer):
            prompt = self.renderer.build_generation_prompt(messages, effort=self.cfg.policy_effort)
        else:
            prompt = self.renderer.build_generation_prompt(messages)
        result = await (await self.sampler()).sample_async(
            prompt=prompt,
            num_samples=self.cfg.n,
            sampling_params=tinker.SamplingParams(
                stop=self.renderer.get_stop_sequences(),
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            ),
        )
        sketch = get_shared_renderer(
            canvas_size=self.cfg.canvas_size,
            max_concurrency=self.cfg.render_concurrency,
            timeout_s=self.cfg.render_timeout_s,
        )

        async def one(i: int, seq: tinker.SampledSequence) -> Candidate:
            message, termination = self.renderer.parse_response(seq.tokens)
            truncated = seq.stop_reason == "length" or not termination.is_stop_sequence
            code = None if truncated else extract_code(get_text_content(message))
            render = await sketch.render(code) if code is not None else None
            return Candidate(index=i, code=code, render=render, truncated=truncated)

        return list(await asyncio.gather(*(one(i, s) for i, s in enumerate(result.sequences))))

    def save(
        self,
        rid: str,
        problem: str,
        candidates: list[Candidate],
        best: int | None,
        scores: list[float | None],
    ) -> dict:
        d = self.out / rid
        d.mkdir(parents=True, exist_ok=True)
        for c in candidates:
            if c.code is not None:
                (d / f"candidate_{c.index}.js").write_text(c.code, encoding="utf-8")
            if c.render is not None and c.render.png is not None:
                (d / f"candidate_{c.index}.png").write_bytes(c.render.png)
        if best is not None:
            shutil.copy(d / f"candidate_{best}.png", d / "best.png")
        row = {
            "id": rid,
            "prompt": problem,
            "best": best,
            "verifier_scores": scores,
            "rendered": [c.ok for c in candidates],
            "truncated": [c.truncated for c in candidates],
        }
        if any(c.eval_score is not None for c in candidates):
            row["eval_scores"] = [c.eval_score for c in candidates]
        (d / "result.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
        return row

    async def paint(self, rid: str, problem: str) -> dict:
        candidates = await self.sample(problem)
        best, scores = await select(self.cfg, problem, candidates)
        row = self.save(rid, problem, candidates, best, scores)
        logger.info(
            "%s: %d/%d rendered, pick %s (%s)",
            rid, sum(c.ok for c in candidates), len(candidates), best, problem,
        )  # fmt: skip
        return row

    async def pick_saved(self, group_json: Path) -> dict:
        """Select among one prompt's paintings saved by a held-out evaluation."""
        g = json.loads(group_json.read_text())
        root = group_json.parents[3]  # .../paintings
        candidates = []
        for i, r in enumerate(g["rollouts"]):
            png = (root / r["png_path"]).read_bytes() if r.get("png_path") else None
            render = RenderResult(
                ok=bool(r["compiled"]) and png is not None,
                png=png,
                error=r.get("render_error"),
                n_brush_calls=r.get("brush_calls", 0),
                blank=False,
                duration_s=0.0,
            )
            code_path = root / r["js_path"] if r.get("js_path") else None
            candidates.append(
                Candidate(
                    index=i,
                    code=code_path.read_text() if code_path and code_path.exists() else None,
                    render=render,
                    truncated=bool(r.get("truncated")),
                    # A sketch that fails to render scores 0, as in the held-out metric.
                    eval_score=(r.get("eval_score") or 0.0) if r["compiled"] else 0.0,
                )
            )
        problem = g["prompt"]["text"]
        best, scores = await select(self.cfg, problem, candidates)
        return self.save(g["prompt"]["id"], problem, candidates, best, scores)

    async def run(self) -> list[dict]:
        if self.cfg.from_heldout:
            groups = sorted(Path(self.cfg.from_heldout).glob("*/group.json"))
            rows = list(await asyncio.gather(*(self.pick_saved(g) for g in groups)))
        else:
            requests = [(request_id(p), p) for p in self.cfg.prompts]
            if self.cfg.heldout:
                _, test = build_prompt_splits(seed=self.cfg.seed)
                requests += [(p.id, p.text) for p in test]
            if not requests:
                raise ValueError("nothing to paint: pass prompts.0=... and/or heldout=True")
            await self.sampler()  # one sampling client, before the requests fan out
            rows = list(await asyncio.gather(*(self.paint(rid, p) for rid, p in requests)))
        with (self.out / "results.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        if self.cfg.from_heldout:
            summary = selection_summary(rows)
            (self.out / "summary.json").write_text(json.dumps(summary, indent=1))
            logger.info("selection on saved paintings: %s", summary)
        return rows


def selection_summary(rows: list[dict]) -> dict[str, float]:
    """Kimi score of the verifier's pick vs the average painting (a random pick)
    and the best painting (an oracle pick), averaged over prompts."""
    picked, mean, oracle = [], [], []
    for r in rows:
        ev = r["eval_scores"]
        mean.append(sum(ev) / len(ev))
        oracle.append(max(ev))
        picked.append(ev[r["best"]] if r["best"] is not None else 0.0)
    n = len(rows)
    return {
        "prompts": n,
        "random_pick": sum(mean) / n,
        "verifier_pick": sum(picked) / n,
        "oracle_pick": sum(oracle) / n,
        "verifier_pick_is_oracle": sum(p >= o - 1e-9 for p, o in zip(picked, oracle)) / n,
    }


async def main(cfg: Config) -> None:
    await BestOfN(cfg).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(chz.entrypoint(Config)))
