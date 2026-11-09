"""Closed-loop rollouts for a learner (ppo.py): a campaign's training arms,
and a batch of episodes run in UML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from expliccit_env import campaign
from expliccit_env.spec import CampaignSpec, Condition, EpisodeSpec, PathSpec, paths_of
from expliccit_env.uml import UmlBuild


def arms(campaign_path: Path, rng: np.random.Generator) -> list[tuple[PathSpec, Condition]]:
    """Every (path, condition) of a campaign, as its expansion pairs them."""
    spec = CampaignSpec.parse(campaign_path.read_text())
    return [(p, c) for p in paths_of(spec.sweep, rng) for c in spec.conditions]


def run(
    out: Path, episodes: list[EpisodeSpec], build: UmlBuild, workers: int, scratch: Path | None
) -> dict[str, Any]:
    """Run ``episodes`` as one shard into ``out``, keeping the steps table;
    the campaign's summary (episodes ok and failed)."""
    campaign.write_episodes(out, episodes, {"name": out.name})
    for _ in campaign.run_shard(
        out,
        build,
        campaign.Shard(0, 1),
        workers=workers,
        scratch=scratch,
        tables=frozenset({"steps"}),
    ):
        pass
    return campaign.summary(out)
