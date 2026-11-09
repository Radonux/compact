//! The episode plan the host writes as `plan.json` next to the trace files.
//! `Plan::load` is the only way in: it parses and validates once, so the rest
//! of the runner works with values that cannot be out of range.
//! Mirrors `src/expliccit_env/guest_plan.py` on the host.

use crate::sys::{Context, Result, fail};
use serde::Deserialize;
use std::num::{NonZeroU32, NonZeroU64};
use std::path::{Path, PathBuf};

pub const PLAN_VERSION: u32 = 7;
/// Extra base RTT a cross flow may add (sch_flowdelay's FLOWDELAY_MAX_DELAY_NS).
pub const MAX_EXTRA_DELAY_NS: u64 = 10_000_000_000;
/// Cross-traffic flows per episode (ports 6000 to 6063), all sender to
/// receiver. ponytail: 64 flows; raise it with the guest's memory (each
/// flow is two sockets and up to four threads), or add reverse flows.
pub const MAX_CROSS: usize = 64;
/// Largest trace the qdisc reads from its file (sch_mmlink.h MMLINK_TRACE_MAX):
/// 64 MiB, 18.6 hours at one byte per millisecond.
pub const MAX_TRACE_BYTES: usize = 1 << 26;
/// TCP_CA_NAME_MAX in include/net/tcp.h, including the NUL.
const TCP_CA_NAME_MAX: usize = 16;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPlan {
    plan_version: u32,
    duration_ms: u32,
    start_offset_ms: u32,
    cc: String,
    obs_mark: u32,
    ring_mib: u32,
    downlink: RawLink,
    uplink: RawLink,
    sysctls: Vec<(String, String)>,
    agent: Option<RawAgent>,
    cross: Vec<RawCross>,
    steps: bool,
}

#[derive(Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum RawCross {
    #[serde(rename = "tcp")]
    Tcp { cc: String, start_ms: u32, stop_ms: u32, extra_delay_ns: u64, agent: Option<RawAgent> },
    #[serde(rename = "udp_cbr")]
    UdpCbr { rate_bps: u64, start_ms: u32, stop_ms: u32 },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgent {
    policy_file: String,
    noise_sigma: f64,
    noise_seed: u64,
    noise_hold: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLink {
    delay_ns: u64,
    trace_file: String,
    limit_pkts: u32,
    limit_bytes: u32,
    loss: u32,
}

/// One direction of the path. Direction fixes the stage order.
pub struct Link {
    pub delay_ns: u64,
    /// Opportunities per millisecond; len >= 2, last entry nonzero.
    pub trace: Vec<u8>,
    /// The same bytes' file, as an absolute path: the qdisc reads it itself,
    /// since a netlink attribute holds under 64 KiB (about 65 s of trace).
    pub trace_path: String,
    /// Droptail limits, at least one nonzero (0 means no limit).
    pub limit_pkts: u32,
    pub limit_bytes: u32,
    /// i.i.d. drop probability times 2^32 (0: no loss stage).
    pub loss: u32,
}

/// A competing flow through the same bottleneck, active over
/// [start_ms, stop_ms) of trace time, with stop_ms <= the episode length.
pub enum CrossFlow {
    /// Bulk TCP with its own congestion control, a stock one or an agent
    /// one driven by its own policy. Its ACKs wait `extra_delay_ns` more
    /// than the path's, so its base RTT is that much longer.
    Tcp { cc: String, start_ms: u32, stop_ms: u32, extra_delay_ns: u64, agent: Option<Agent> },
    /// Constant bit rate UDP, counted at the IP layer (1500-byte packets).
    Udp { rate_bps: NonZeroU64, start_ms: u32, stop_ms: u32 },
}

/// A sysctl written in the sender and receiver namespaces. The name is a
/// dotted path of `[a-z0-9_]` components, so it cannot leave /proc/sys.
pub struct Sysctl {
    pub path: String,
    pub value: String,
}

/// Who decides the observed flow's cwnd: the kernel's own congestion
/// control, or an agent.
pub enum Control {
    Stock,
    Agent(Agent),
}

/// A sage-v1 policy driving one socket's tcp_agent controller ("agent" or
/// "agent_reno") through /dev/expl_agent after every step.
pub struct Agent {
    pub policy: PathBuf,
    pub noise: Noise,
}

/// Exploration: Gaussian noise added to every action the policy takes,
/// from a seeded generator, so an online training run is reproducible.
/// Each draw is kept for `hold` steps.
#[derive(Clone, Copy)]
pub struct Noise {
    pub sigma: f64,
    pub seed: u64,
    pub hold: NonZeroU32,
}

pub struct Plan {
    pub duration_ms: NonZeroU32,
    pub start_offset_ms: u32,
    pub cc: String,
    pub obs_mark: NonZeroU32,
    pub ring_mib: NonZeroU32,
    pub downlink: Link,
    pub uplink: Link,
    pub sysctls: Vec<Sysctl>,
    pub control: Control,
    pub cross: Vec<CrossFlow>,
    /// Run Sage's step loop (always, under a policy) and write steps.bin.
    pub steps: bool,
}

impl Plan {
    pub fn load(dir: &Path) -> Result<Plan> {
        let text = std::fs::read_to_string(dir.join("plan.json")).ctx("reading plan.json")?;
        let raw: RawPlan = serde_json::from_str(&text).ctx("parsing plan.json")?;
        if raw.plan_version != PLAN_VERSION {
            return fail(format!("plan_version {} != {PLAN_VERSION}", raw.plan_version));
        }
        cc_name(&raw.cc)?;
        if raw.cross.len() > MAX_CROSS {
            return fail(format!("{} cross flows, at most {MAX_CROSS}", raw.cross.len()));
        }
        let cross = raw
            .cross
            .into_iter()
            .map(|c| CrossFlow::parse(c, raw.duration_ms, dir))
            .collect::<Result<_>>()?;
        let control = match Agent::parse(&raw.cc, raw.agent, dir)? {
            None => Control::Stock,
            Some(a) => Control::Agent(a),
        };
        Ok(Plan {
            steps: raw.steps || matches!(control, Control::Agent(_)),
            control,
            cross,
            duration_ms: nonzero(raw.duration_ms, "duration_ms")?,
            start_offset_ms: raw.start_offset_ms,
            cc: raw.cc,
            obs_mark: nonzero(raw.obs_mark, "obs_mark")?,
            ring_mib: nonzero(raw.ring_mib, "ring_mib")?,
            downlink: Link::load(dir, raw.downlink).ctx("downlink")?,
            uplink: Link::load(dir, raw.uplink).ctx("uplink")?,
            sysctls: raw.sysctls.into_iter().map(Sysctl::parse).collect::<Result<_>>()?,
        })
    }
}

fn cc_name(cc: &str) -> Result<()> {
    let valid = !cc.is_empty()
        && cc.len() < TCP_CA_NAME_MAX
        && cc.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
    if valid { Ok(()) } else { fail(format!("invalid congestion control name {cc:?}")) }
}

impl Agent {
    /// The agent section for a socket running `cc`: present exactly when
    /// `cc` is an agent controller.
    fn parse(cc: &str, raw: Option<RawAgent>, dir: &Path) -> Result<Option<Agent>> {
        let is_agent = matches!(cc, "agent" | "agent_reno");
        match (is_agent, raw) {
            (false, None) => Ok(None),
            (true, Some(a)) => {
                if !(0.0..=2.0).contains(&a.noise_sigma) {
                    return fail(format!("agent.noise_sigma {} outside [0, 2]", a.noise_sigma));
                }
                if a.noise_hold > 1000 {
                    return fail(format!("agent.noise_hold {} above 1000", a.noise_hold));
                }
                Ok(Some(Agent {
                    policy: dir.join(plain_file_name(&a.policy_file)?),
                    noise: Noise {
                        sigma: a.noise_sigma,
                        seed: a.noise_seed,
                        hold: nonzero(a.noise_hold, "agent.noise_hold")?,
                    },
                }))
            }
            (true, None) => fail(format!("cc {cc} needs an agent section")),
            (false, Some(_)) => fail(format!("cc {cc} takes no agent section")),
        }
    }
}

impl CrossFlow {
    /// The flow's agent, when a policy drives it.
    pub fn agent(&self) -> Option<&Agent> {
        match self {
            CrossFlow::Tcp { agent, .. } => agent.as_ref(),
            CrossFlow::Udp { .. } => None,
        }
    }

    fn parse(raw: RawCross, duration_ms: u32, dir: &Path) -> Result<CrossFlow> {
        let window = |start: u32, stop: u32| {
            if start < stop && stop <= duration_ms {
                Ok(())
            } else {
                fail(format!("cross flow window [{start}, {stop}) not inside [0, {duration_ms})"))
            }
        };
        match raw {
            RawCross::Tcp { cc, start_ms, stop_ms, extra_delay_ns, agent } => {
                cc_name(&cc)?;
                window(start_ms, stop_ms)?;
                if extra_delay_ns > MAX_EXTRA_DELAY_NS {
                    return fail(format!("cross flow extra delay {extra_delay_ns} ns above 10 s"));
                }
                let agent = Agent::parse(&cc, agent, dir).ctx("cross flow")?;
                Ok(CrossFlow::Tcp { cc, start_ms, stop_ms, extra_delay_ns, agent })
            }
            RawCross::UdpCbr { rate_bps, start_ms, stop_ms } => {
                window(start_ms, stop_ms)?;
                match NonZeroU64::new(rate_bps) {
                    Some(r) if rate_bps <= 10_000_000_000 => {
                        Ok(CrossFlow::Udp { rate_bps: r, start_ms, stop_ms })
                    }
                    _ => fail(format!("cross rate {rate_bps} bit/s outside (0, 10 Gbit/s]")),
                }
            }
        }
    }
}

/// A file name inside the plan directory: no separators, no leading dot.
fn plain_file_name(name: &str) -> Result<&str> {
    let plain = !name.is_empty()
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        && !name.starts_with('.');
    if plain { Ok(name) } else { fail(format!("{name:?} must be a plain file name")) }
}

fn nonzero(v: u32, what: &str) -> Result<NonZeroU32> {
    NonZeroU32::new(v).map_or_else(|| fail(format!("{what} must be nonzero")), Ok)
}

impl Link {
    fn load(dir: &Path, raw: RawLink) -> Result<Link> {
        let file = dir.join(plain_file_name(&raw.trace_file)?);
        let trace = std::fs::read(&file).ctx(format!("reading {}", raw.trace_file))?;
        let absolute = std::fs::canonicalize(&file).ctx(format!("resolving {}", raw.trace_file))?;
        let Some(trace_path) = absolute.to_str().map(str::to_owned) else {
            return fail(format!("{} is not a UTF-8 path", absolute.display()));
        };
        if trace.len() < 2 || trace.len() > MAX_TRACE_BYTES {
            return fail(format!("trace has {} bytes, need 2..={MAX_TRACE_BYTES}", trace.len()));
        }
        if trace.last() == Some(&0) {
            return fail("trace must end with a nonzero entry");
        }
        if raw.limit_pkts == 0 && raw.limit_bytes == 0 {
            return fail("droptail needs a packet or byte limit");
        }
        Ok(Link {
            delay_ns: raw.delay_ns,
            trace,
            trace_path,
            limit_pkts: raw.limit_pkts,
            limit_bytes: raw.limit_bytes,
            loss: raw.loss,
        })
    }
}

impl Sysctl {
    fn parse((name, value): (String, String)) -> Result<Sysctl> {
        let valid_name = name.split('.').count() >= 2
            && name.split('.').all(|c| {
                !c.is_empty() && c.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            });
        if !valid_name {
            return fail(format!("invalid sysctl name {name:?}"));
        }
        if value.is_empty() || value.bytes().any(|b| b.is_ascii_control() && b != b'\t') {
            return fail(format!("invalid value for sysctl {name}"));
        }
        Ok(Sysctl { path: format!("/proc/sys/{}", name.replace('.', "/")), value })
    }
}
