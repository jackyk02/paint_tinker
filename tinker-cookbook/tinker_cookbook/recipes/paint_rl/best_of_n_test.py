import asyncio
import io

import pytest
from PIL import Image

from tinker_cookbook.recipes.paint_rl import best_of_n
from tinker_cookbook.recipes.paint_rl.render import RenderResult
from tinker_cookbook.recipes.paint_rl.verifier import GroupScores


def _render(ok: bool) -> RenderResult:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return RenderResult(
        ok=ok, png=buf.getvalue(), error=None, n_brush_calls=3, blank=False, duration_s=0.0
    )


class _FakeScorer:
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.n_images: int | None = None

    async def score_group(self, problem, images, seed=0):
        self.n_images = len(images)
        return GroupScores(scores=self.scores)


def test_select_picks_the_top_rendered_candidate(monkeypatch: pytest.MonkeyPatch):
    # Candidate 1 did not render, so the verifier sees candidates 0, 2 and 3 only.
    scorer = _FakeScorer([0.4, 0.7, 0.7])
    monkeypatch.setattr(best_of_n, "get_shared_scorer", lambda config: scorer)
    candidates = [
        best_of_n.Candidate(index=i, code="", render=_render(ok), truncated=False)
        for i, ok in enumerate([True, False, True, True])
    ]
    best, scores = asyncio.run(best_of_n.select(best_of_n.Config(), "p", candidates))
    assert scorer.n_images == 3
    assert scores == [0.4, None, 0.7, 0.7]
    assert best == 2  # a tie goes to the earliest sample


def test_select_has_no_pick_when_nothing_renders(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(best_of_n, "get_shared_scorer", lambda config: _FakeScorer([]))
    candidates = [best_of_n.Candidate(index=0, code=None, render=None, truncated=True)]
    assert asyncio.run(best_of_n.select(best_of_n.Config(), "p", candidates)) == (None, [None])


def test_selection_summary_compares_pick_with_random_and_oracle():
    rows = [
        {"best": 1, "eval_scores": [0.2, 0.8, 0.5]},
        {"best": 0, "eval_scores": [0.4, 0.6, 0.0]},
        {"best": None, "eval_scores": [0.0, 0.0, 0.0]},
    ]
    s = best_of_n.selection_summary(rows)
    assert s["prompts"] == 3
    assert s["verifier_pick"] == pytest.approx((0.8 + 0.4 + 0.0) / 3)
    assert s["random_pick"] == pytest.approx((0.5 + 1 / 3 + 0.0) / 3)
    assert s["oracle_pick"] == pytest.approx((0.8 + 0.6 + 0.0) / 3)
    assert s["verifier_pick_is_oracle"] == pytest.approx(2 / 3)
