# expliccit-env kernel modules

Five out-of-tree modules give the UML guest its bottleneck, its sender
observations, its policy-driven congestion controls, and one record
stream to the host ([docs/architecture.md](../../docs/architecture.md)).

| Module | Role |
|---|---|
| `expl_rec` | The record ring, read through `/dev/expl_rec`. `sch_mmlink`, `expl_tcpobs` and `tcp_agent` write into it, so it loads first. |
| `sch_mmlink` | qdisc `mmlink`: one direction of a trace-driven link and its delay line, with Mahimahi's semantics and optional i.i.d. random loss on entry (`mm-loss`). The path namespace runs two. |
| `sch_flowdelay` | qdisc `flowdelay`: extra delay per TCP destination port, one FIFO per port, on the receiver's device, so a cross flow's ACKs give it a longer base RTT. Loaded only when an episode has such a flow. |
| `expl_tcpobs` | Sender state at every arriving segment, from the `tcp:tcp_probe` tracepoint, for sockets with `SO_MARK == mark`. It reads only the generic TCP socket, so it works for any congestion control. |
| `tcp_agent` | Congestion controls `agent` (holds the window and pacing rate a policy set, Sage-style), `agent_pure` (Sage's kernel as deployed: the window a policy set, and DeepCC pacing at every delivery) and `agent_cubic` (each action sets the window once, then Cubic, vendored from Linux 7.0 in `cubic.h`, grows it, as Orca does; `TCP_CC_INFO` gives Orca's statistics per period), driven through `/dev/expl_agent`. Each write names its socket by the writer's file descriptor, and each socket keeps its own action. |

## Building

`scripts/build-guest.sh build/uml` builds them against the kernel from
`scripts/build-uml.sh`, in a copy, with the guest runner and root. By hand:

    make -C third_party/linux O=$PWD/build/uml ARCH=um M=$PWD/kernel/modules W=1 modules

They build cleanly with `W=1`, and `scripts/check-kernel.sh` runs
checkpatch `--strict` on them. sparse (`C=1`) is UNRUN: Ubuntu 24.04's
sparse is too old for Linux 7.0.

## ABIs

`expliccit_rec.h` (record layouts, `EXPL_REC_ABI` 4, and the agent action,
`EXPL_AGENT_ABI` 2), `sch_mmlink.h` and `sch_flowdelay.h` (netlink
attributes) are shared with the guest runner (`guest/src/recorder.rs`,
`guest/src/netlink.rs`) and the host decoder
(`src/expliccit_env/records.py`). Change them only together, and bump the
ABI for any layout change; `tests/test_records_abi.py` checks the numpy
dtypes against the C structs, offset for offset (it needs a C compiler).
`expl_rec.h` is the in-kernel interface the other modules use to emit
records.

## Other congestion controls

They come from the kernel tree in `third_party/linux`. Cubic, BBRv1 and
Reno are built in. BBRv3 (`tcp_bbr3`, Google's; see
[third_party/README.md](../../third_party/README.md)), BIC, H-TCP and
Westwood are modules of the kernel build. `scripts/build-guest.sh` builds
CDG, HighSpeed, Hybla, Illinois, LP, NV, Scalable, Vegas, Veno and YeAH
from the unmodified in-tree sources as out-of-tree modules, because
enabling them in `kernel/uml.config` changes the kernel image. A
controller built as a module is loaded only in the episodes that use it.
