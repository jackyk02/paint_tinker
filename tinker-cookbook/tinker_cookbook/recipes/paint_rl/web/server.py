"""A small web studio for a trained paint_rl checkpoint.

    python -m tinker_cookbook.recipes.paint_rl.web.server \\
        model_path=tinker://<run id>/sampler_weights/000400 \\
        examples_dirs.0=/tmp/paint_rl/best_of_n/demo

Then open http://127.0.0.1:8000. A visitor types a painting request (or picks
one of the examples); the server runs test-time best-of-n (``best_of_n.py``):
the checkpoint samples ``n`` sketches, they are rendered headlessly, and base
Inkling-Small picks one as a pairwise verifier. The page polls the job and
shows every candidate, its verifier score, its code, and the pick.

``examples_dirs`` are ``best_of_n`` output directories (with ``results.jsonl``)
shown as ready-made examples, so they appear instantly. Everything runs on
the server's ``TINKER_API_KEY``; the server listens on 127.0.0.1 by default,
so expose it deliberately (each request costs sampling and verifier calls).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chz
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from tinker_cookbook.recipes.paint_rl import best_of_n

logger = logging.getLogger(__name__)

INDEX = Path(__file__).with_name("index.html")
_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_FILE = re.compile(r"^(best|candidate_[0-9]{1,2})\.(png|js)$")


@chz.chz
class Config:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8000
    n: int = 5
    # best_of_n output directories to show as examples.
    examples_dirs: tuple[str, ...] = ()
    out_dir: str = "/tmp/paint_rl/web"
    # Requests painted at once; the rest wait in line.
    max_concurrent_jobs: int = 4
    max_queued_jobs: int = 32
    max_prompt_chars: int = 300
    model_label: str = "Inkling-Small, RL step 400 (verifier reward)"


@dataclass
class Job:
    id: str
    prompt: str
    created: float
    stage: str = "queued"  # queued -> sketching -> judging -> done | failed
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    row: dict[str, Any] | None = None
    rendered: list[bool] = field(default_factory=list)


class PaintRequest(BaseModel):
    prompt: str


def _example_rows(dirs: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for k, d in enumerate(dirs):
        path = Path(d) / "results.jsonl"
        if not path.exists():
            logger.warning("no results.jsonl in %s", d)
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append({**json.loads(line), "source": k})
    return rows


def _candidates(row: dict[str, Any], url: str) -> list[dict[str, Any]]:
    scores = row["verifier_scores"]
    return [
        {
            "index": i,
            "rendered": ok,
            "score": scores[i],
            "image": f"{url}/candidate_{i}.png" if ok else None,
            "code": f"{url}/candidate_{i}.js",
        }
        for i, ok in enumerate(row["rendered"])
    ]


def build_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="paint_rl studio", docs_url=None, redoc_url=None)
    jobs_dir = Path(cfg.out_dir) / "jobs"
    bon = best_of_n.BestOfN(
        best_of_n.Config(model_path=cfg.model_path, n=cfg.n, out_dir=str(jobs_dir))
    )
    jobs: dict[str, Job] = {}
    slots = asyncio.Semaphore(cfg.max_concurrent_jobs)
    tasks: set[asyncio.Task[None]] = set()

    async def run(job: Job) -> None:
        async with slots:
            job.started = time.time()
            job.stage = "sketching"
            try:
                candidates = await bon.sample(job.prompt)
                job.rendered = [c.ok for c in candidates]
                job.stage = "judging"
                best, scores = await best_of_n.select(bon.cfg, job.prompt, candidates)
                job.row = bon.save(job.id, job.prompt, candidates, best, scores)
                job.stage = "done"
            except Exception as e:  # report it to the page instead of hanging the job
                logger.exception("job %s failed", job.id)
                job.error = f"{type(e).__name__}: {e}"
                job.stage = "failed"
            finally:
                job.finished = time.time()

    @app.on_event("startup")
    async def _startup() -> None:
        await bon.sampler()  # fail fast on a bad model_path or key

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX.read_text(encoding="utf-8"))

    @app.get("/api/info")
    async def info() -> dict[str, Any]:
        examples = _example_rows(cfg.examples_dirs)  # re-read, so new examples appear
        return {
            "model": cfg.model_label,
            "n": cfg.n,
            "max_prompt_chars": cfg.max_prompt_chars,
            "examples": [
                {
                    "id": r["id"],
                    "prompt": r["prompt"],
                    "best": r["best"],
                    "candidates": _candidates(r, f"/examples/{r['source']}/{r['id']}"),
                }
                for r in examples
                if r["best"] is not None
            ],
        }

    @app.post("/api/paint")
    async def paint(req: PaintRequest) -> dict[str, str]:
        prompt = " ".join(req.prompt.split())
        if not 3 <= len(prompt) <= cfg.max_prompt_chars:
            raise HTTPException(400, f"a request is 3 to {cfg.max_prompt_chars} characters")
        waiting = sum(j.stage in ("queued", "sketching", "judging") for j in jobs.values())
        if waiting >= cfg.max_queued_jobs:
            raise HTTPException(429, "the studio is busy; try again in a few minutes")
        job = Job(id=uuid.uuid4().hex[:12], prompt=prompt, created=time.time())
        jobs[job.id] = job
        task = asyncio.create_task(run(job))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {"id": job.id}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        ahead = sum(j.stage == "queued" and j.created < job.created for j in jobs.values())
        out: dict[str, Any] = {
            "id": job.id,
            "prompt": job.prompt,
            "stage": job.stage,
            "queue_position": ahead if job.stage == "queued" else 0,
            "elapsed": (job.finished or time.time()) - (job.started or job.created),
            "rendered": job.rendered,
            "error": job.error,
        }
        if job.row is not None:
            out["best"] = job.row["best"]
            out["candidates"] = _candidates(job.row, f"/files/{job.id}")
        return out

    def _send(root: Path, key: str, name: str) -> FileResponse:
        if not _SAFE.match(key) or not _FILE.match(name):
            raise HTTPException(404)
        path = root / key / name
        if not path.is_file():
            raise HTTPException(404)
        media = "image/png" if name.endswith(".png") else "text/plain; charset=utf-8"
        return FileResponse(path, media_type=media)

    @app.get("/files/{job_id}/{name}")
    async def job_file(job_id: str, name: str) -> FileResponse:
        return _send(jobs_dir, job_id, name)

    @app.get("/examples/{source}/{example_id}/{name}")
    async def example_file(source: int, example_id: str, name: str) -> FileResponse:
        if not 0 <= source < len(cfg.examples_dirs):
            raise HTTPException(404)
        return _send(Path(cfg.examples_dirs[source]), example_id, name)

    return app


def main(cfg: Config) -> None:
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(chz.entrypoint(Config))
