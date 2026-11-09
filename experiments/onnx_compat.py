"""Every released policy on the current guest: each .onnx asset of the
model-* releases (experiments/onnx-compat.tsv) runs one short episode alone
on one link, and the report says which ran, which failed and why, which of
the contract's inputs each graph reads (policy.py: obs, state, truth, cwnd),
and what it did (applied action, utilization).

Usage: python experiments/onnx_compat.py plan TAG OUT
           (download TAG's policies into OUT/policies, write OUT/campaign.json
           and OUT/names.json, each graph's name and inputs)
       python experiments/onnx_compat.py report OUT DIR...
           (OUT/onnx-compat.md from the DIRs' episodes)
"""

import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import pyarrow.parquet as pq

MANIFEST = Path(__file__).with_name("onnx-compat.tsv")
URL = "https://github.com/Radonux/palaestra/releases/download/{tag}/{name}"
# 24 then 12 Mbit/s for 4 s each, 20 ms one way, a 100-packet buffer: a
# policy that reads the path must change its window when the capacity does.
PATH = {
    "trace": {"kind": "periodic_levels", "levels_mbps": [24, 12], "level_ms": 4000},
    "delay": {"mahak_compat_mrtt_ms": 20},
    "buffer": {"packets": 100},
}


def plan(tag: str, out: Path) -> None:
    rows = [r for r in csv.DictReader(MANIFEST.open(), delimiter="\t") if r["tag"] == tag]
    if not rows:
        raise SystemExit(f"{tag}: not in {MANIFEST.name}")
    (out / "policies").mkdir(parents=True, exist_ok=True)
    ccs, names = [], {}
    for r in rows:
        f = out / "policies" / r["name"]
        urllib.request.urlretrieve(URL.format(tag=tag, name=r["name"]), f)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        controller = f"agent:{sha[:12]}"
        if controller in names:  # the same graph under two names runs once
            names[controller]["name"] += f" = {r['name']}"
            continue
        inputs = [i.name for i in onnx.load(str(f), load_external_data=False).graph.input]
        names[controller] = {"name": r["name"], "inputs": inputs}
        ccs.append({"kind": "agent", "policy": {"path": str(f.resolve()), "sha256": sha}})
    camp = {
        "name": tag,
        "ccs": ccs,
        "sweep": {"kind": "paths", "paths": [PATH]},
        "conditions": [{}],
        "duration_ms": 8000,
        "seed": 1,
        "steps": True,
    }
    (out / "campaign.json").write_text(json.dumps(camp, indent=1))
    (out / "names.json").write_text(json.dumps(names, indent=1))


def report(out: Path, dirs: list[Path]) -> None:
    lines = [
        "| Release | Policy | Inputs | Status | Failure | Applied a: mean, sd | Utilization |",
        "|---|---|---|---|---|---|---|",
    ]
    ok = failed = 0
    for d in sorted(dirs):
        names = json.loads((d / "names.json").read_text())
        for shard in sorted((d / "c").glob("shards/*")):
            eps = pq.read_table(shard / "episodes.parquet").to_pylist()
            steps = pq.read_table(shard / "steps.parquet", columns=["episode_id", "applied_action"])
            eid = np.asarray(steps["episode_id"].to_pylist())
            act = np.asarray(steps["applied_action"].to_numpy(zero_copy_only=False), np.float64)
            for e in eps:
                info = names[e["controller"]]
                a = act[eid == e["episode_id"]]
                good = e["status"] == "ok" and a.size > 0 and bool(np.isfinite(a).all())
                ok += good
                failed += not good
                util = e.get("utilization")
                used = "" if util is None else f"{util:.3f}"
                acted = f"{a.mean():+.3f}, {a.std():.3f}" if a.size else "none"
                lines.append(
                    f"| {d.name} | {info['name']} | {', '.join(info['inputs'])} | {e['status']} |"
                    f" {e.get('failure') or ''} | {acted} | {used} |"
                )
    head = [
        f"Released policies on the current guest: {ok} ran, {failed} failed, of {ok + failed}"
        " (each alone for 8 s on 24 then 12 Mbit/s, 20 ms one way, 100 packets).",
        "",
    ]
    text = "\n".join(head + lines) + "\n"
    (out / "onnx-compat.md").write_text(text)
    print(text, end="")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("tag")
    pp.add_argument("out", type=Path)
    pr = sub.add_parser("report")
    pr.add_argument("out", type=Path)
    pr.add_argument("dirs", type=Path, nargs="+")
    a = p.parse_args()
    if a.cmd == "plan":
        plan(a.tag, a.out)
    else:
        report(a.out, a.dirs)


if __name__ == "__main__":
    main()
