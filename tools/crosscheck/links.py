"""The cross-check's four links, for the original stack and for palaestra.

The original stack is Ubuntu 16.04, Sage's patched Linux 4.19 and ccBench's
Mahimahi (tools/crosscheck/README.md). Each link is a palaestra PathSpec
built the way palaestra's own experiments build it (experiments/ccbench.py,
experiments/expliccit_case.py), and everything the team needs comes from it:
the trace file is the spec's own timestamps, and the shell arguments are its
delay, loss and droptail limit.

    a  expliCCit's case-study link: 24 and 48 Mbit/s alternating every 7 s,
       40 ms one way, 1000 packets, 45 s
    b  48 Mbit/s, 20 ms one way, 1 BDP (160 packets), 30 s
    c  12 Mbit/s, 40 ms one way, 4 BDP (320 packets), 1% data loss, 30 s
    d  ccBench's solo cell 96 Mbit/s, 80 ms one way, 4 BDP (5120 packets), 30 s

BDP is ccBench's, D * B // 6 packets for one-way delay D ms and B Mbit/s.

Usage: python tools/crosscheck/links.py write DIR
       python tools/crosscheck/links.py plan OUT --sage SAGE_ONNX
write puts DIR/<link>.trace and DIR/links.json. plan writes that kit to
OUT/kit, imports expliCCit's released policy as deployed into
OUT/policies/expliccit (needs the train extra), and plans OUT/c-crosscheck:
every link with Cubic, BBRv1, Vegas, Sage and expliCCit, seeds 1 to 3, with
OUT/aliases.json (agent:SHA12 -> sage or expliccit).
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import ccbench
import expliccit_case
from heldout import controllers

from expliccit_env import campaign
from expliccit_env import mahimahi as mm
from expliccit_env.spec import EpisodeSpec, PathSpec

STOCK = "cubic,bbr,vegas"
SEEDS = (1, 2, 3)
WARMUP_S = 2  # the window starts this long after the flow's first data packet
TAIL_S = 1  # and ends this long before the run's duration after it


@dataclass(frozen=True, slots=True)
class Link:
    name: str
    title: str
    path: PathSpec
    duration_s: int

    @property
    def window_s(self) -> tuple[int, int]:
        """Seconds after the flow's first data packet that the metrics cover."""
        return WARMUP_S, self.duration_s - TAIL_S

    def one_way_ms(self) -> int:
        ns = self.path.delay.one_way_ns
        if ns % 1_000_000:
            raise ValueError(f"link {self.name}: mm-delay takes whole ms")
        return ns // 1_000_000

    def shell(self, log: str, command: str) -> str:
        """The Mahimahi shells that make this link, data entering from outside.

        ponytail: packet limits and one trace for both directions only, which
        is all these links use; add byte limits or an uplink trace if one does.
        """
        p = self.path
        if p.buffer.bytes or p.uplink_trace is not None:
            raise ValueError(f"link {self.name}: only packet limits and one trace")
        loss = "".join(
            f" mm-loss {way} {rate:g}"
            for way, rate in (("downlink", p.loss.downlink), ("uplink", p.loss.uplink))
            if rate
        )
        q = f"packets={p.buffer.packets}"
        t = f"{self.name}.trace"
        return (
            f"mm-delay {self.one_way_ms()}{loss} mm-link {t} {t} --downlink-log={log}"
            f" --uplink-queue=droptail --uplink-queue-args={q}"
            f" --downlink-queue=droptail --downlink-queue-args={q} -- {command}"
        )


def _ccbench(bw: int, delay: int, bdps: int, loss: float = 0.0) -> PathSpec:
    return PathSpec.model_validate(
        ccbench.path(bw, delay, ccbench.buffer(bw, delay, bdps), [bw], loss)
    )


LINKS = (
    Link(
        "a",
        "expliCCit's case-study link: 24 and 48 Mbit/s alternating every 7 s, 40 ms one way,"
        " 1000 packets",
        PathSpec.model_validate(expliccit_case.path(40, 1000)),
        expliccit_case.SECONDS,
    ),
    Link("b", "48 Mbit/s, 20 ms one way, 1 BDP", _ccbench(48, 20, 1), 30),
    Link("c", "12 Mbit/s, 40 ms one way, 4 BDP, 1% data loss", _ccbench(12, 40, 4, 0.01), 30),
    Link("d", "ccBench's solo cell: 96 Mbit/s, 80 ms one way, 4 BDP", _ccbench(96, 80, 4), 30),
)
BY_NAME = {link.name: link for link in LINKS}


def link_of(spec: EpisodeSpec) -> Link | None:
    """The cross-check link an episode ran, if any."""
    for link in LINKS:
        if spec.path == link.path and spec.duration_ms == link.duration_s * 1000:
            return link if not spec.cross else None
    return None


def write(out: Path) -> list[dict[str, object]]:
    """Each link's trace file and its entry in links.json."""
    out.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for link in LINKS:
        ts = link.path.trace.timestamps()
        text = "".join(f"{t}\n" for t in ts.tolist())
        # The file is exactly the spec's trace, as Mahimahi and sch_mmlink read it.
        assert np.array_equal(mm.parse_trace(text), ts)
        trace = out / f"{link.name}.trace"
        trace.write_text(text)
        q = f"packets={link.path.buffer.packets}"
        entries.append(
            {
                "name": link.name,
                "title": link.title,
                "trace": trace.name,
                "trace_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "trace_period_ms": int(ts[-1]),
                # One 1500-byte packet per line: 12 kbit, so 12 * lines / ms in Mbit/s.
                "mean_rate_mbps": round(12.0 * ts.size / float(ts[-1]), 6),
                "mm_delay_ms": link.one_way_ms(),
                "mm_loss": link.path.loss.model_dump(),
                "mm_link": {
                    "uplink_trace": trace.name,
                    "downlink_trace": trace.name,
                    "uplink_queue": "droptail",
                    "uplink_queue_args": q,
                    "downlink_queue": "droptail",
                    "downlink_queue_args": q,
                },
                "duration_s": link.duration_s,
                "window_s": list(link.window_s),
                "shell": link.shell("LOG", "COMMAND"),
                "palaestra_path": link.path.model_dump(mode="json"),
            }
        )
    (out / "links.json").write_text(json.dumps(entries, indent=1) + "\n")
    return entries


def plan(out: Path, sage: Path) -> None:
    from expliccit_env.train import run  # noqa: PLC0415 (needs the train extra)

    write(out / "kit")
    policy = out / "policies" / "expliccit"
    run.import_expliccit(expliccit_case.MODELS, policy, [], as_deployed=True)
    ccs, aliases = controllers(STOCK, [f"sage={sage}", f"expliccit={policy / 'policy.onnx'}"])
    eps = [
        EpisodeSpec.model_validate(
            {
                "path": link.path.model_dump(mode="json"),
                "cc": cc,
                "duration_ms": link.duration_s * 1000,
                "seed": seed,
            }
        )
        for link in LINKS
        for cc in ccs
        for seed in SEEDS
    ]
    campaign.write_episodes(out / "c-crosscheck", eps, {"name": "c-crosscheck"})
    (out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")
    print(f"c-crosscheck: {len(eps)} episodes")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    w = sub.add_parser("write")
    w.add_argument("dir", type=Path)
    pl = sub.add_parser("plan")
    pl.add_argument("out", type=Path)
    pl.add_argument("--sage", type=Path, required=True, help="Sage's released policy (ONNX)")
    a = p.parse_args()
    if a.verb == "write":
        for e in write(a.dir):
            print(json.dumps({k: v for k, v in e.items() if k != "palaestra_path"}))
    else:
        plan(a.out, a.sage)


if __name__ == "__main__":
    main()
