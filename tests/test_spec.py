"""Spec parsing laws: invalid combinations are unrepresentable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from expliccit_env.guest_plan import TcpCrossPlan, UdpCrossPlan
from expliccit_env.spec import BaseRtt, CampaignSpec, EpisodeSpec, MahakMrtt

EXAMPLES = Path(__file__).parents[1] / "examples"


def episode(delay: object, **extra: object) -> str:
    return json.dumps(
        {
            "path": {
                "trace": {"kind": "mahak_step", "bw_mbps": 10, "step": 1.0},
                "delay": delay,
                "buffer": {"packets": 100},
            },
            "cc": {"kind": "stock", "name": "cubic"},
            **extra,
        }
    )


def test_delay_is_exactly_one_kind() -> None:
    assert EpisodeSpec.parse(episode({"mahak_compat_mrtt_ms": 20})).path.delay == MahakMrtt(
        mahak_compat_mrtt_ms=20
    )
    assert EpisodeSpec.parse(episode({"base_rtt_ms": 40})).path.delay == BaseRtt(base_rtt_ms=40)
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(episode({"mahak_compat_mrtt_ms": 20, "base_rtt_ms": 40}))
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(episode({}))


def test_mahak_mrtt_is_one_way() -> None:
    a = EpisodeSpec.parse(episode({"mahak_compat_mrtt_ms": 20}))
    b = EpisodeSpec.parse(episode({"base_rtt_ms": 40}))
    assert a.path.delay.one_way_ns == b.path.delay.one_way_ns == 20_000_000


def test_rejects_unknown_fields_and_bad_values() -> None:
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(episode({"base_rtt_ms": 40}, surprise=1))
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(episode({"base_rtt_ms": 0}))
    bad_buffer = json.loads(episode({"base_rtt_ms": 40}))
    bad_buffer["path"]["buffer"] = {"packets": 0, "bytes": 0}
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(json.dumps(bad_buffer))


def test_episode_id_is_a_content_hash() -> None:
    a = EpisodeSpec.parse(episode({"base_rtt_ms": 40}, seed=3))
    assert a.episode_id() == EpisodeSpec.parse(a.canonical_json()).episode_id()
    assert a.episode_id() != EpisodeSpec.parse(episode({"base_rtt_ms": 40}, seed=4)).episode_id()


def test_campaign_expansion_is_deterministic() -> None:
    spec = CampaignSpec.parse((EXAMPLES / "campaign-icnp25-sample.json").read_text())
    a, b = spec.episodes(), spec.episodes()
    assert len(a) == 2000
    assert [e.canonical_json() for e in a] == [e.canonical_json() for e in b]
    assert len({e.episode_id() for e in a}) == len(a)
    smoke = CampaignSpec.parse((EXAMPLES / "campaign-smoke.json").read_text()).episodes()
    assert len(smoke) == 8


def test_examples_parse() -> None:
    EpisodeSpec.parse((EXAMPLES / "episode-mahak.json").read_text()).guest_plan()


def test_loss_and_cross_reach_the_guest_plan() -> None:
    spec = json.loads(episode({"base_rtt_ms": 40}, duration_ms=5000))
    spec["path"]["loss"] = {"downlink": 0.01, "uplink": 0.5}
    spec["cross"] = [
        {"kind": "tcp", "cc": "bbr3", "start_ms": 1000},
        {"kind": "udp_cbr", "rate_mbps": 2.5, "start_ms": 0, "stop_ms": 4000},
    ]
    plan = EpisodeSpec.parse(json.dumps(spec)).guest_plan()
    assert plan.downlink.loss == round(0.01 * 2**32)
    assert plan.uplink.loss == 2**31
    assert plan.cross == (TcpCrossPlan("bbr3", 1000, 5000), UdpCrossPlan(2_500_000, 0, 4000))


@pytest.mark.parametrize(
    ("path_extra", "cross"),
    [
        ({"loss": {"downlink": 1.0}}, []),  # certain loss: no transfer at all
        ({"loss": {"downlink": -0.1}}, []),
        ({}, [{"kind": "tcp", "cc": "cubic", "start_ms": 5000}]),  # starts at the end
        ({}, [{"kind": "tcp", "cc": "cubic", "stop_ms": 6000}]),  # outlives the episode
        ({}, [{"kind": "udp_cbr", "rate_mbps": 0}]),
        ({}, [{"kind": "udp_cbr", "rate_mbps": 1, "cc": "cubic"}]),
        ({}, [{"kind": "tcp", "cc": "agent"}]),
        ({}, [{"kind": "tcp", "cc": "cubic"}] * 17),
    ],
)
def test_rejects_bad_loss_and_cross(path_extra: dict[str, object], cross: object) -> None:
    spec = json.loads(episode({"base_rtt_ms": 40}, duration_ms=5000))
    spec["path"] |= path_extra
    spec["cross"] = cross
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(json.dumps(spec))


def test_conditions_multiply_a_campaign() -> None:
    base = json.loads((EXAMPLES / "campaign-smoke.json").read_text())
    plain = CampaignSpec.parse(json.dumps(base)).episodes()
    assert plain == CampaignSpec.parse(json.dumps(base | {"conditions": [{}]})).episodes()
    cond = {"loss": {"downlink": 0.02}, "cross": [{"kind": "udp_cbr", "rate_mbps": 1}]}
    wide = CampaignSpec.parse(json.dumps(base | {"conditions": [{}, cond]})).episodes()
    assert len(wide) == 2 * len(plain)
    assert sum(not e.cross and e.path.loss.downlink == 0 for e in wide) == len(plain)
    assert sum(e.path.loss.downlink == 0.02 and len(e.cross) == 1 for e in wide) == len(plain)


def test_ccs_per_point_runs_a_seeded_subset_on_each_point() -> None:
    base = json.loads((EXAMPLES / "campaign-smoke.json").read_text())
    base |= {"conditions": [{}, {"loss": {"downlink": 0.01}}]}
    every = CampaignSpec.parse(json.dumps(base)).episodes()
    n = len(base["ccs"])
    spec = CampaignSpec.parse(json.dumps(base | {"ccs_per_point": n - 1}))
    eps = spec.episodes()
    assert eps == spec.episodes() and len(eps) == len(every) // n * (n - 1)
    by_point: dict[str, set[tuple[str, float]]] = {}
    for e in eps:
        key = json.dumps(e.path.model_dump(mode="json", exclude={"loss"}), sort_keys=True)
        by_point.setdefault(key, set()).add((e.cc.model_dump_json(), e.path.loss.downlink))
    for runs in by_point.values():  # the same n - 1 controllers under both conditions
        ccs = {c for c, _ in runs}
        assert len(ccs) == n - 1 and runs == {(c, x) for c in ccs for x in (0.0, 0.01)}
    with pytest.raises(ValidationError):
        CampaignSpec.parse(json.dumps(base | {"ccs_per_point": n + 1}))


def test_conditions_per_point_runs_a_seeded_subset_of_conditions_on_each_point() -> None:
    base = json.loads((EXAMPLES / "campaign-smoke.json").read_text())
    conds = [{}, {"loss": {"downlink": 0.01}}, {"loss": {"downlink": 0.02}}]
    spec = CampaignSpec.parse(json.dumps(base | {"conditions": conds, "conditions_per_point": 2}))
    eps = spec.episodes()
    assert eps == spec.episodes()
    by_point: dict[str, set[tuple[str, float]]] = {}
    for e in eps:
        key = json.dumps(e.path.model_dump(mode="json", exclude={"loss"}), sort_keys=True)
        by_point.setdefault(key, set()).add((e.cc.model_dump_json(), e.path.loss.downlink))
    picks = set()
    for runs in by_point.values():  # every controller under the same 2 conditions
        losses = {x for _, x in runs}
        assert len(losses) == 2 and len(runs) == 2 * len(base["ccs"])
        picks.add(frozenset(losses))
    assert len(picks) > 1  # the subsets differ between points
    with pytest.raises(ValidationError):
        CampaignSpec.parse(json.dumps(base | {"conditions": conds, "conditions_per_point": 4}))


def test_a_udp_share_becomes_a_rate_on_each_paths_mean_capacity() -> None:
    def path(levels: list[float]) -> dict[str, object]:
        return {
            "trace": {"kind": "periodic_levels", "levels_mbps": levels, "level_ms": 1000},
            "delay": {"mahak_compat_mrtt_ms": 20},
            "buffer": {"packets": 100},
        }

    udp = {"kind": "udp_share", "share": 0.25, "start_ms": 2000, "stop_ms": 9000}
    spec = {
        "name": "u",
        "ccs": [{"kind": "stock", "name": "cubic"}],
        "sweep": {"kind": "paths", "paths": [path([8.0]), path([24.0, 48.0])]},
        "conditions": [{"cross": [udp, {"kind": "tcp", "cc": "reno", "start_ms": 0}]}],
        "duration_ms": 10_000,
    }
    eps = CampaignSpec.parse(json.dumps(spec)).episodes()
    rates = [e.cross[0].rate_mbps for e in eps if e.cross[0].kind == "udp_cbr"]
    assert rates == [2.0, 9.0]  # a quarter of 8 and of (24 + 48) / 2
    assert all(
        (e.cross[0].start_ms, e.cross[0].stop_ms, e.cross[1].kind) == (2000, 9000, "tcp")
        for e in eps
    )
    with pytest.raises(ValidationError):
        CampaignSpec.parse(json.dumps(spec | {"conditions": [{"cross": [udp | {"share": 1.5}]}]}))


def test_paths_sweep_takes_any_trace_and_conditions_override_loss() -> None:
    path = {
        "trace": {"kind": "periodic_levels", "levels_mbps": [24, 48], "level_ms": 7000},
        "delay": {"mahak_compat_mrtt_ms": 40},
        "buffer": {"packets": 1000},
        "loss": {"downlink": 0.5},
    }
    spec = {
        "name": "p",
        "ccs": [{"kind": "stock", "name": "vegas"}],
        "sweep": {"kind": "paths", "paths": [path]},
        "conditions": [{}, {"loss": {"downlink": 0.01}}],
        "steps": True,
    }
    eps = CampaignSpec.parse(json.dumps(spec)).episodes()
    assert [e.path.loss.downlink for e in eps] == [0.0, 0.01]
    assert all(e.steps and e.cc.name == "vegas" for e in eps)
    assert eps[0].guest_plan().downlink.trace.sum() == 24_000 * 7 // 12 + 48_000 * 7 // 12
