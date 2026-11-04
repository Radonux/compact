# Third-party sources

Two upstream trees and one set of released weights are vendored here, so
builds and experiments do not depend on fetching them. Each was imported
unmodified in one commit, and every change we make to it is a later commit,
so `git diff <import> -- <tree>` is exactly what we changed.

| Tree | Import commit | Our changes | License |
|---|---|---|---|
| `linux/` | `e8eadca` | HZ, seeded randomness, Google's BBRv3 | GPL-2.0 (`linux/COPYING`) |
| `mahimahi/` | `ac3c802` | Mahak's metric script | GPL-3.0 (`mahimahi/COPYING`) |
| `expliccit-models/` | the commit that added it | none | none stated; published with its authors' consent |

## linux/

Upstream Linux 7.0, imported unmodified in commit
`e8eadca Import Linux 7.0 source tree, unmodified` from the release tarball
<https://cdn.kernel.org/pub/linux/kernel/v7.x/linux-7.0.tar.xz>
(sha256 `bb7f6d80b387c757b7d14bb93028fcb90f793c5c0d367736ee815a100b3891f0`,
matching kernel.org's `sha256sums.asc`, fetched 2026-10-30). That commit was
checked with `diff -r` against a fresh extraction of the tarball.

To list our kernel changes, or see them as one patch:

    git log --oneline e8eadca.. -- third_party/linux
    git diff e8eadca -- third_party/linux

### The patches we carry

| Commits | What | Why |
|---|---|---|
| `dea41ac` | `um`: HZ 1000 | arch/um hard-codes HZ at 100. TCP's jiffy timers (delayed ACK, RTO, BBR's min RTT and PROBE_RTT stamps) would then be 10x coarser than on an HZ=1000 host kernel. |
| `12562c6` | `um`: `CONFIG_UML_SEEDED_RANDOM` | Bit-reproducible episodes. UML takes host entropy in three places: `setup_arch()`, `arch_get_random_longs()` at every CRNG reseed, and the kernel command line that `random_init_early()` mixes in. The option replaces host entropy with a stream seeded by `um_seed=` and keeps the command line out of the pool. It is not cryptographically secure; it is for simulation kernels only. |
| `3cb3ca7` .. `8eb2345` (12) | BBRv3 core prerequisites | Imported unchanged with `git am` from google/bbr branch `bbr-v3-2026-09-16-01` (commits `05406b11b` .. `c552eebe7`), original authors and messages kept. |
| `44162fe` | `tcp_bbr3` module | Same branch, commit `5dee74110`: BBRv3 as a separate congestion control `bbr3`, leaving BBRv1 as `bbr`. |
| `814b1d9` | 7.0 port of `tcp_bbr3` | The branch targets a newer kernel with a dedicated `cwnd_event_tx_start` op. Linux 7.0 delivers `CA_EVENT_TX_START` through `cwnd_event`, so the port dispatches it there to the same, unchanged handler. Kernels from 7.3 on have the op again; revert this commit when moving past 7.0. |
| `56f58a1` | Opt-in app-limited checks | The series adds app-limited checks on every ACK and write timer, for every socket. That changed the app-limited marks on other congestion controls' rate samples, which BBRv1's bandwidth filter reads. The checks now run only for congestion controls that set `TCP_CONG_BROAD_APP_LIMITED`; `tcp_bbr3` sets it, so every other congestion control behaves as in plain 7.0. |

The 13 commits from google/bbr are listed in `linux-imports.txt`, so
`scripts/check-kernel.sh` does not style-check them. The branch is based on
net-next of early June 2026; all 13 applied to 7.0 without conflicts.

What was checked: with BBRv3 built as a module and not loaded, seeded Cubic
and BBRv1 episodes are byte-identical to those of the kernel without the
series (`docs/validation.md`, M6). The check needs two kernel builds, so it
is not in CI; rerun it when the patch set changes.

### Adding a kernel patch

- Our own change: one commit that touches only `third_party/linux`.
  `scripts/check-kernel.sh` runs checkpatch `--strict` on it.
- An upstream patch: import it with `git am`, keeping its author and
  message (only the project's `Co-Authored-By` trailer is added), list it in
  `linux-imports.txt`, and add it to the table above.

## mahimahi/

Upstream Mahimahi, imported unmodified in commit
`ac3c802 Import Mahimahi source tree, unmodified`: a `git archive` of
<https://github.com/ravinet/mahimahi> at
`f1346c38c415abfc98626dd82450f49a57033603` (2025-04-14). The repository does not
build it. `kernel/modules/sch_mmlink.c` ports its `src/frontend/link_queue.cc`,
and `src/expliccit_env/mahimahi.py` holds an independent Python port of the
same code that the tests use as the oracle for the kernel.

| Commit | What | Why |
|---|---|---|
| `594599d` | `scripts/mm-th-delay` | Mahak's metric script, copied verbatim from <https://github.com/Ppazhooh/mahak> at `5077583` (sha256 `3c49fea6684fb848c696a073e072002bad3ea7818e73b44a50b53aa480ab458f`). It is Mahimahi's `scripts/mm-throughput-graph`, modified to accept drop lines and to print utilization and mean queueing delay, so it is kept as a change to the tree it derives from. `src/expliccit_env/mahimahi.py` is tested against it. |

Comparing the link model with real Mahimahi is UNRUN: it needs a root host,
and Mahak's `highbdp.patch` no longer applies to current Mahimahi and must
be ported first (`docs/validation.md`).

## expliccit-models/

expliCCit's released net1 and actor: the four files of its
`sage_rl/offline/models` that `expliccit train import-expliccit` reads,
copied unchanged on 2026-11-04 from the copy of expliCCit's code we were
given (dated 2025-05-22). The `.pth` files hold only tensors (they load with
`torch.load(..., weights_only=True)`), and the JSON files the min-max
scalers' parameters; the pickled scikit-learn scalers and the unused
checkpoints were left out. expliCCit's code itself stays out of this
repository until expliCCit is published (`AGENTS.md`).

| File | sha256 |
|---|---|
| `best_model.pth` (net1) | `c8cf456f791f112aef9ece8d27f7b777ff8b057be00c043e0af1c7dbc7573a9d` |
| `actor.pth` (net2) | `e6f1032aad21a479be68acd723519ed8f1fbfe728f862a00db3a455f7a55c66b` |
| `x_scaler_params.json` | `456859d70c5a01424f2f84a018f8d5006c7cfaa6dd1f01f6c260e428429c1c5e` |
| `y_scaler_params.json` | `e0eb72cb760729b7986a2da6ae9115c6f4651188e79602714e41a3d367d612bd` |
