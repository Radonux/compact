# expliccit-env kernel modules

Four out-of-tree modules give the UML guest its bottleneck, its sender
observations, its policy-driven congestion controls, and one record stream
to the host.

| Module | Role |
|---|---|
| `expl_rec` | The global record ring, read through `/dev/expl_rec`. The other three modules write into it, so load it first. |
| `sch_mmlink` | qdisc `mmlink`: one direction of a trace-driven link plus a delay line, with Mahimahi semantics and optional i.i.d. random loss on entry (`mm-loss`). The path namespace runs two, one per direction. |
| `expl_tcpobs` | Sender state at every arriving segment, from the `tcp:tcp_probe` tracepoint, for sockets with `SO_MARK == mark`. It reads only the generic TCP socket, so it works for any congestion control. |
| `tcp_agent` | Congestion controls `agent` (Sage-style: holds the window and pacing rate the policy set) and `agent_reno` (Orca-style: each action sets the window once, then Reno grows it), driven by policies through `/dev/expl_agent`. Each write names its socket by the writer's file descriptor, and each socket keeps its own action, so any number of agent sockets share a kernel. |

## Building

`scripts/build-guest.sh build/uml` builds them against the kernel from
`scripts/build-uml.sh`, in a copy so the repository stays clean, together
with the guest runner and root. By hand, writing the objects next to the
sources:

    make -C third_party/linux O=$PWD/build/uml ARCH=um M=$PWD/kernel/modules W=1 modules

They build cleanly with `W=1`, and `scripts/check-kernel.sh` runs
checkpatch `--strict` on them. sparse (`C=1`) is UNRUN: Ubuntu 24.04's
sparse is too old for Linux 7.0.

## ABIs

`expliccit_rec.h` (record layouts) and `sch_mmlink.h` (netlink attributes)
are ABIs shared with the guest runner and the host decoder. Change them only
together with `guest/` and `src/expliccit_env/records.py`, and bump
`EXPL_REC_ABI` for any record layout change.
`tests/test_records_abi.py::test_dtypes_match_c_structs` checks that the
numpy dtypes in `records.py` match the C structs, offset for offset (it
needs a C compiler). `guest/src/netlink.rs` mirrors `sch_mmlink.h`.
`expl_rec.h` is the in-kernel interface the other modules use to emit
records.

## Other modules the guest loads

The guest loads these four modules in every episode. Other congestion
controls come from the kernel tree in `third_party/linux`, not from this
directory. Cubic, BBRv1 and Reno are built into the kernel. BBRv3
(`tcp_bbr3`, Google's, see `third_party/README.md`), BIC, H-TCP and
Westwood are modules of the kernel build itself. `scripts/build-guest.sh`
builds CDG, HighSpeed, Hybla, Illinois, LP, NV, Scalable, Vegas, Veno and
YeAH from the unmodified in-tree sources as out-of-tree modules, because
enabling them in `kernel/uml.config` changes the kernel image. A
congestion control built as a module is loaded only in the episodes that
use it.
