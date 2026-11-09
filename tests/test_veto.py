"""The copies veto (experiments/veto_export.py) and the actor it combines
(recurrent.ConcatActor)."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from expliccit_env.train.recurrent import ConcatActor, EnsembleActor, RecurrentActor  # noqa: E402

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "veto_export.py"
_spec = importlib.util.spec_from_file_location("veto_export", _PATH)
assert _spec is not None and _spec.loader is not None
veto_export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(veto_export)


def test_concat_actor_puts_a_policy_next_to_beliefs() -> None:
    torch.manual_seed(0)
    u = RecurrentActor(5, 8)
    beliefs = EnsembleActor([RecurrentActor(5, 8, outputs=2) for _ in range(3)])
    actor = ConcatActor([u, beliefs])
    assert actor.hidden == 8 + 24
    h = actor.cell(torch.randn(4, 6), torch.zeros(4, actor.hidden))
    out = actor.mean(h)
    assert out.shape == (4, 3)
    hu, hb = torch.split(h, [8, 24], dim=-1)
    assert torch.allclose(out[:, 0], u.mean(hu))
    assert torch.allclose(out[:, 1:], beliefs.mean(hb))


def test_veto_keeps_the_rule_when_copies_are_believed() -> None:
    veto = veto_export.Veto(u_rule=0.25, action_max=3.0, tq=0.0)
    a = torch.tensor([[0.5, 0.0, 1.0], [0.5, 0.0, -1.0], [0.5, 0.0, 0.0]])
    u = veto(a, torch.zeros(3, 1), None)
    assert torch.allclose(u, torch.tensor([0.25, 1.5, 0.875]))
    sharp = veto_export.Veto(u_rule=0.25, action_max=3.0, tq=0.5)
    assert torch.allclose(
        sharp(torch.tensor([[0.5, 0.0, 0.0]]), torch.zeros(1, 1), None), torch.tensor([1.5])
    )
