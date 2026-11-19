"""The grounding library's features and fitted formulas as labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from expliccit_env.dataset import _with_features
from expliccit_env.policy import OBS_DIM
from expliccit_env.train import data, library
from expliccit_env.train.graph import TruthColumns
from expliccit_env.train.lcfs import Labels, columns, known, units
from expliccit_env.train.spec import Net1Spec

# Seed 128's last direction, as decompose printed it.
FORMULA = (
    "4.668585 - log([max_960ms(buffer_fill)]*[mean_240ms(queue_delay_mean_ns_10ms)] + 77.984276)"
)


def _episode(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "buffer_fill": rng.uniform(0, 1, n).astype(np.float32),
        "queue_delay_mean_ns_10ms": rng.uniform(0, 5e7, n).astype(np.float32),
    }


def test_features_read_only_their_own_episodes_past() -> None:
    # Two episodes of a step each value: the window's maximum climbs within
    # an episode and starts again at the second.
    episode = np.repeat([0, 1], 100)
    step = np.tile(np.arange(100), 2)
    fill = np.tile(np.linspace(0, 0.99, 100), 2).astype(np.float32)
    v = library.values(
        ["max_960ms(buffer_fill)", "mean_240ms(buffer_fill)"],
        {"buffer_fill": fill}.__getitem__,
        episode,
        step,
    )
    assert np.allclose(v["max_960ms(buffer_fill)"], fill, atol=1e-6)
    # The mean is on the log scale: the first step is its own value, then
    # it lags the climb.
    mean = v["mean_240ms(buffer_fill)"]
    assert mean[0] == pytest.approx(fill[0]) and mean[100] == pytest.approx(fill[100])
    assert np.all(mean[1:100] < fill[1:100])


def test_a_formula_is_read_as_data() -> None:
    x = {"[a]": np.array([1.0, 2.0])}
    got = library.evaluate("(-1)*1.5/[a] + [a]**2 - log([a])", lambda n: x[f"[{n}]"])
    assert np.allclose(got, -1.5 / x["[a]"] + x["[a]"] ** 2 - np.log(x["[a]"]))
    for bad in ("__import__('os').system('true')", "[a].real", "exp([a])", "lambda: 1"):
        with pytest.raises((ValueError, SyntaxError)):
            library.evaluate(bad, lambda _: np.ones(1))


def test_a_formula_is_an_lcf_without_a_unit() -> None:
    assert known(FORMULA) and units(FORMULA) == (1.0, "")
    assert columns([FORMULA]) == {"buffer_fill", "queue_delay_mean_ns_10ms"}
    assert library.stored([FORMULA, "capacity_bps"]) == (
        "max_960ms(buffer_fill)",
        "mean_240ms(queue_delay_mean_ns_10ms)",
    )
    assert Net1Spec(lcfs=(FORMULA,)).lcfs == (FORMULA,)
    for bad in ("exp([buffer_fill])", "[sum_240ms(buffer_fill)]", "1 +"):
        with pytest.raises(ValueError, match="unknown LCF"):
            Net1Spec(lcfs=(bad,))
    # The live truth has no past: an oracle cannot read a formula.
    assert not TruthColumns.known(FORMULA)


def _shard(d: Path, tables: list[pa.Table]) -> None:
    d.mkdir(parents=True)
    eids = [t["episode_id"][0].as_py() for t in tables]
    pq.write_table(
        pa.table(
            {
                "episode_id": eids,
                "status": ["ok"] * len(eids),
                "spec_json": [json.dumps({"path": {"bw": i}}) for i in range(len(eids))],
                "controller": ["bbr3"] * len(eids),
            }
        ),
        d / "episodes.parquet",
    )
    pq.write_table(pa.concat_tables(tables), d / "steps.parquet")


def _table(eid: str, labels: dict[str, np.ndarray]) -> pa.Table:
    n = len(next(iter(labels.values())))
    return pa.table(
        {
            "episode_id": [eid] * n,
            "step": np.arange(n),
            "obs": pa.FixedSizeListArray.from_arrays(
                pa.array(np.zeros(n * OBS_DIM, np.float32)), OBS_DIM
            ),
            "action": np.zeros(n, np.float32),
            "reward": np.zeros(n, np.float32),
        }
        | {f"label_{k}": v for k, v in labels.items()}
    )


def test_stored_features_give_strided_steps_the_formula_of_every_step(tmp_path: Path) -> None:
    eps = {f"e{i}": _episode(300, i) for i in range(3)}
    full = [_table(e, lab) for e, lab in eps.items()]
    _shard(tmp_path / "full" / "s", full)
    every = data.load([tmp_path / "full"], seed=0, val=0.0, test=0.0)
    want = every.labels[FORMULA][every.step % 6 == 0]
    assert np.isfinite(want).all()

    # As campaign run --features ... --steps-stride 6 writes them.
    features = library.stored([FORMULA])
    strided = [_with_features(t, features) for t in full]
    strided = [t.filter(pa.array(np.asarray(t["step"]) % 6 == 0)) for t in strided]
    _shard(tmp_path / "strided" / "s", strided)
    got = data.load([tmp_path / "strided"], seed=0, val=0.0, test=0.0).labels[FORMULA]
    assert np.allclose(got, want, rtol=1e-5)

    # Without them a strided table cannot give the formula.
    _shard(
        tmp_path / "bare" / "s", [t.filter(pa.array(np.asarray(t["step"]) % 6 == 0)) for t in full]
    )
    bare = data.load([tmp_path / "bare"], seed=0, val=0.0, test=0.0)
    with pytest.raises(KeyError, match="not stored"):
        bare.labels[FORMULA]
    with pytest.raises(KeyError, match="not stored"):
        Labels(eps["e0"])[FORMULA]


def test_a_formula_reads_a_label_in_its_physical_units() -> None:
    bps = np.array([2e6, 5e7], np.float32)
    labels = Labels({"flow_arrival_rate_bps": bps})
    got = labels["[flow_arrival_rate_bps] / 2"]  # Mbit/s, as PySR read it
    assert np.allclose(got, bps * 1e-6 / 2, rtol=1e-5)
