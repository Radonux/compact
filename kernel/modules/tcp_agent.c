// SPDX-License-Identifier: GPL-2.0
/*
 * tcp_agent: congestion controls actuated by a userspace policy.
 *
 * A policy (Orca- or Sage-style agent, or an exported expliCCit policy)
 * observes the flow itself (TCP_INFO) and writes struct expl_agent_action
 * to /dev/expl_agent. Three controllers apply it:
 *
 *   "agent"       Sage-style. cong_control holds snd_cwnd and the pacing rate
 *                 at the latest action on every delivery. Until the first
 *                 action the initial window is held.
 *   "agent_pure"  Sage's own controller ("pure" with DeepCC pacing, as Sage
 *                 ships it in its 4.19.112-0062 kernel; read from that
 *                 binary), on this kernel's stack: each new action sets
 *                 snd_cwnd once, and every delivery paces at
 *                 max(cwnd, packets_out) * MTU / srtt; its hooks ask for a
 *                 send buffer of three windows, as pure's do. Nothing below
 *                 the controller is changed.
 *   "agent_cubic" Orca-style. Each new action sets snd_cwnd once, at the
 *                 next ACK; between actions Cubic (Linux 7.0's, with
 *                 HyStart, vendored in cubic.h) evolves the window and the
 *                 core stack handles loss and pacing. TCP_CC_INFO returns
 *                 Orca's per-period statistics (struct agent_cubic_info)
 *                 and starts a new period.
 *
 * Each action is for one socket: the writer names it by its own file
 * descriptor, and the action is kept in that socket's congestion-control
 * state, so any number of agent sockets run side by side. Every accepted
 * action is reported to expl_rec as AGENT_ACTION with the socket's port.
 *
 * Locking: action_lock (irqsave) guards every socket's published action.
 * Readers run in softirq or in socket-backlog processing (BH enabled); the
 * writer runs in process context and holds a reference to the socket's
 * file for the whole write. Socket state is only touched from the CC hooks,
 * which run with the socket locked.
 */
#include <linux/fs.h>
#include <linux/inet_diag.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/net.h>
#include <linux/spinlock.h>
#include <linux/uaccess.h>
#include <net/tcp.h>

#include "cubic.h"
#include "expl_rec.h"

static DEFINE_SPINLOCK(action_lock);	/* guards seq, cwnd and pacing_rate */

/* In icsk_ca_priv, so each agent socket carries its own action. */
struct agent_priv {
	u32 seq;			/* actions accepted; 0: none yet */
	u32 applied_seq;		/* agent_pure, agent_cubic: last applied */
	u32 cwnd;
	u32 pad;
	u64 pacing_rate;
};

/* Orca's per-period statistics (its kernel's deepcc_api): RTT samples, rate
 * samples and lost packets since the last TCP_CC_INFO read.
 */
struct orca_stats {
	u64 rate_sum;			/* bytes per second, summed over samples */
	u64 read_us;			/* when the period began */
	u32 rate_cnt;
	u32 rtt_cnt;
	u32 rtt_min_us;			/* over the connection, as Orca's */
	u32 pre_lost;			/* tp->lost when the period began */
	u32 pre_delivered;		/* tp->delivered at the last ACK */
};

/* agent_cubic's area: Cubic first, where cubic.h expects it. */
struct agent_cubic_priv {
	struct bictcp cubic;
	struct agent_priv agent;
	struct orca_stats stats;
};

/* TCP_CC_INFO of agent_cubic, in place of union tcp_cc_info. */
struct agent_cubic_info {
	u32 rtt_samples;		/* Orca's cnt */
	u32 rate;			/* Orca's thr: mean rate sample, bytes/s */
	u32 lost_bytes;
	u32 elapsed_us;			/* the period's length */
	u32 min_rtt_us;
};

static struct tcp_congestion_ops tcp_agent_cubic;

static struct agent_priv *agent_of(struct sock *sk)
{
	if (READ_ONCE(inet_csk(sk)->icsk_ca_ops) == &tcp_agent_cubic)
		return &((struct agent_cubic_priv *)inet_csk_ca(sk))->agent;
	return inet_csk_ca(sk);
}

/* This socket's latest action into *out; its seq (0: none yet). */
static u32 agent_current(struct sock *sk, struct expl_agent_action *out)
{
	const struct agent_priv *p = agent_of(sk);
	unsigned long flags;
	u32 seq;

	spin_lock_irqsave(&action_lock, flags);
	out->cwnd = p->cwnd;
	out->pacing_rate = p->pacing_rate;
	seq = p->seq;
	spin_unlock_irqrestore(&action_lock, flags);
	return seq;
}

static void agent_set_pacing(struct sock *sk, u64 rate)
{
	u64 max = READ_ONCE(sk->sk_max_pacing_rate);

	if (!rate) {
		WRITE_ONCE(sk->sk_pacing_rate, max);
		return;
	}
	/* Ask the stack to pace internally, as BBR does (unless fq paces). */
	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);
	WRITE_ONCE(sk->sk_pacing_rate, min(rate, max));
}

static void agent_apply(struct sock *sk, const struct expl_agent_action *a)
{
	struct tcp_sock *tp = tcp_sk(sk);

	tcp_snd_cwnd_set(tp, min(a->cwnd, tp->snd_cwnd_clamp));
	agent_set_pacing(sk, a->pacing_rate);
}

static void agent_init(struct sock *sk)
{
	struct agent_priv *p = agent_of(sk);
	unsigned long flags;

	spin_lock_irqsave(&action_lock, flags);
	memset(p, 0, sizeof(*p));
	spin_unlock_irqrestore(&action_lock, flags);
}

/* "agent": hold the latest action. */
static void agent_cong_control(struct sock *sk, u32 ack, int flag,
			       const struct rate_sample *rs)
{
	struct expl_agent_action a;

	if (agent_current(sk, &a))
		agent_apply(sk, &a);
}

/* "agent_pure": Sage's pure_tcp_main as cong_control. Its TCP_CWND sets the
 * window at once and then deepcc_update_cwnd paces; here the window is set at
 * the next delivery, and every delivery paces, as there (enable 2, pacing
 * type srtt, coefficient 100): max(cwnd, packets_out) * tcp_mss_to_mtu() *
 * USEC_PER_SEC / (srtt_us >> 3), unpaced before the first RTT sample.
 */
static void agent_pure_cong_control(struct sock *sk, u32 ack, int flag,
				    const struct rate_sample *rs)
{
	struct agent_priv *p = agent_of(sk);
	struct tcp_sock *tp = tcp_sk(sk);
	struct expl_agent_action a;
	u32 seq = agent_current(sk, &a);
	u32 srtt = tp->srtt_us >> 3;
	u64 rate;

	if (seq != p->applied_seq) {
		p->applied_seq = seq;
		tcp_snd_cwnd_set(tp, min(a.cwnd, tp->snd_cwnd_clamp));
	}
	rate = (u64)tcp_mss_to_mtu(sk, tp->mss_cache) * USEC_PER_SEC *
	       max(tcp_snd_cwnd(tp), tp->packets_out);
	if (srtt)
		rate = div_u64(rate, srtt);
	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);
	WRITE_ONCE(sk->sk_pacing_rate, min_t(u64, rate, READ_ONCE(sk->sk_max_pacing_rate)));
}

/* pure's bbr_sndbuf_expand and bbr_min_tso_segs. */
static u32 agent_pure_sndbuf_expand(struct sock *sk)
{
	return 3;
}

static u32 agent_pure_min_tso_segs(struct sock *sk)
{
	return READ_ONCE(sk->sk_pacing_rate) > 149999 ? 2 : 1;
}

/* "agent_cubic": apply each new action's window once, at the next ACK of any
 * kind (cong_avoid alone would miss actions while not cwnd-limited); Cubic
 * runs in between. The pacing rate is the stack's, as under Orca.
 */
static void agent_cubic_init(struct sock *sk)
{
	struct agent_cubic_priv *ca = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);

	cubictcp_init(sk);
	agent_init(sk);
	memset(&ca->stats, 0, sizeof(ca->stats));
	ca->stats.read_us = tcp_clock_us();
	ca->stats.pre_lost = tp->lost;
	ca->stats.pre_delivered = tp->delivered;
}

static void agent_cubic_in_ack_event(struct sock *sk, u32 flags)
{
	struct agent_cubic_priv *ca = inet_csk_ca(sk);
	struct orca_stats *st = &ca->stats;
	struct tcp_sock *tp = tcp_sk(sk);
	struct expl_agent_action a;
	u32 seq = agent_current(sk, &a);

	if (seq != ca->agent.applied_seq) {
		ca->agent.applied_seq = seq;
		tcp_snd_cwnd_set(tp, min(a.cwnd, tp->snd_cwnd_clamp));
	}
	/* Orca averages every rate sample. This hook runs before the stack
	 * takes this ACK's sample, so each ACK that delivered data counts the
	 * latest saved sample: one ACK late.
	 */
	if (tp->delivered != st->pre_delivered) {
		st->pre_delivered = tp->delivered;
		if (tp->rate_interval_us) {
			st->rate_sum += div_u64((u64)tp->rate_delivered * tp->mss_cache *
						USEC_PER_SEC, tp->rate_interval_us);
			st->rate_cnt++;
		}
	}
}

static void agent_cubic_acked(struct sock *sk, const struct ack_sample *sample)
{
	struct orca_stats *st = &((struct agent_cubic_priv *)inet_csk_ca(sk))->stats;

	cubictcp_acked(sk, sample);
	if (sample->rtt_us > 0) {
		st->rtt_cnt++;
		if (!st->rtt_min_us || st->rtt_min_us > sample->rtt_us)
			st->rtt_min_us = sample->rtt_us;
	}
}

/* Only TCP_CC_INFO (ext ~0U) reads and resets the period; inet_diag gets
 * nothing, so it cannot cut a period short.
 */
static size_t agent_cubic_get_info(struct sock *sk, u32 ext, int *attr,
				   union tcp_cc_info *info)
{
	struct orca_stats *st = &((struct agent_cubic_priv *)inet_csk_ca(sk))->stats;
	struct agent_cubic_info *out = (struct agent_cubic_info *)info;
	struct tcp_sock *tp = tcp_sk(sk);
	u64 now = tcp_clock_us();

	if (ext != ~0U)
		return 0;
	out->rtt_samples = st->rtt_cnt;
	out->rate = st->rate_cnt ? min_t(u64, div_u64(st->rate_sum, st->rate_cnt), U32_MAX) : 0;
	out->lost_bytes = (tp->lost - st->pre_lost) * tp->mss_cache;
	out->elapsed_us = min_t(u64, now - st->read_us, U32_MAX);
	out->min_rtt_us = st->rtt_min_us;
	st->rate_sum = 0;
	st->rate_cnt = 0;
	st->rtt_cnt = 0;
	st->read_us = now;
	st->pre_lost = tp->lost;
	*attr = 0;
	return sizeof(*out);
}

static u32 agent_ssthresh(struct sock *sk)
{
	return max(tcp_snd_cwnd(tcp_sk(sk)) >> 1U, 2U);
}

static u32 agent_undo_cwnd(struct sock *sk)
{
	return tcp_snd_cwnd(tcp_sk(sk));
}

static struct tcp_congestion_ops tcp_agent __read_mostly = {
	.init = agent_init,
	.ssthresh = agent_ssthresh,
	.cong_control = agent_cong_control,
	.undo_cwnd = agent_undo_cwnd,
	.owner = THIS_MODULE,
	.name = "agent",
};

static struct tcp_congestion_ops tcp_agent_pure __read_mostly = {
	.init = agent_init,
	.ssthresh = agent_ssthresh,
	.cong_control = agent_pure_cong_control,
	.undo_cwnd = agent_undo_cwnd,
	.sndbuf_expand = agent_pure_sndbuf_expand,
	.min_tso_segs = agent_pure_min_tso_segs,
	.owner = THIS_MODULE,
	.name = "agent_pure",
};

static struct tcp_congestion_ops tcp_agent_cubic __read_mostly = {
	.init = agent_cubic_init,
	.ssthresh = cubictcp_recalc_ssthresh,
	.cong_avoid = cubictcp_cong_avoid,
	.set_state = cubictcp_state,
	.undo_cwnd = tcp_reno_undo_cwnd,
	.cwnd_event = cubictcp_cwnd_event,
	.pkts_acked = agent_cubic_acked,
	.in_ack_event = agent_cubic_in_ack_event,
	.get_info = agent_cubic_get_info,
	.owner = THIS_MODULE,
	.name = "agent_cubic",
};

static bool is_agent(const struct sock *sk)
{
	const struct tcp_congestion_ops *ops = READ_ONCE(inet_csk(sk)->icsk_ca_ops);

	return ops == &tcp_agent || ops == &tcp_agent_pure || ops == &tcp_agent_cubic;
}

/* Context: process. Untrusted input: one whole, valid action per write,
 * for a TCP socket of the writer's that runs an agent congestion control.
 */
static ssize_t agent_write(struct file *file, const char __user *buf,
			   size_t count, loff_t *ppos)
{
	struct expl_rec_agent_action r = {
		.hdr = { .type = EXPL_REC_AGENT_ACTION, .len = sizeof(r) },
	};
	struct expl_agent_action a;
	struct agent_priv *p;
	struct socket *sock;
	unsigned long flags;
	struct sock *sk;
	int err;

	if (count != sizeof(a))
		return -EINVAL;
	if (copy_from_user(&a, buf, sizeof(a)))
		return -EFAULT;
	if (a.abi != EXPL_AGENT_ABI || a.cwnd < 2 || a.cwnd > EXPL_AGENT_MAX_CWND ||
	    a.pacing_rate > EXPL_AGENT_MAX_PACING)
		return -EINVAL;
	sock = sockfd_lookup(a.fd, &err);
	if (!sock)
		return err;
	sk = sock->sk;
	if (!sk || !sk_is_tcp(sk) || !is_agent(sk)) {
		sockfd_put(sock);
		return -EINVAL;
	}

	p = agent_of(sk);
	spin_lock_irqsave(&action_lock, flags);
	p->cwnd = a.cwnd;
	p->pacing_rate = a.pacing_rate;
	r.action_seq = ++p->seq;
	spin_unlock_irqrestore(&action_lock, flags);

	r.hdr.t_ns = ktime_get_ns();
	r.cwnd = a.cwnd;
	r.pacing_rate = a.pacing_rate;
	r.sport = ntohs(inet_sk(sk)->inet_sport);
	sockfd_put(sock);
	expl_rec_emit(&r.hdr);
	return count;
}

static const struct file_operations agent_fops = {
	.owner = THIS_MODULE,
	.write = agent_write,
};

static struct miscdevice agent_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "expl_agent",
	.fops = &agent_fops,
	.mode = 0200,
};

static int __init tcp_agent_init(void)
{
	int err;

	BUILD_BUG_ON(sizeof(struct expl_rec_agent_action) != 40);
	BUILD_BUG_ON(sizeof(struct expl_agent_action) != 24);
	BUILD_BUG_ON(sizeof(struct agent_priv) > ICSK_CA_PRIV_SIZE);
	BUILD_BUG_ON(sizeof(struct agent_cubic_priv) > ICSK_CA_PRIV_SIZE);
	BUILD_BUG_ON(sizeof(struct agent_cubic_info) > sizeof(union tcp_cc_info));
	cubic_scales_init();
	err = tcp_register_congestion_control(&tcp_agent);
	if (err)
		return err;
	err = tcp_register_congestion_control(&tcp_agent_pure);
	if (err)
		goto err_agent;
	err = tcp_register_congestion_control(&tcp_agent_cubic);
	if (err)
		goto err_pure;
	err = misc_register(&agent_dev);
	if (err)
		goto err_cubic;
	return 0;

err_cubic:
	tcp_unregister_congestion_control(&tcp_agent_cubic);
err_pure:
	tcp_unregister_congestion_control(&tcp_agent_pure);
err_agent:
	tcp_unregister_congestion_control(&tcp_agent);
	return err;
}

static void __exit tcp_agent_exit(void)
{
	misc_deregister(&agent_dev);
	tcp_unregister_congestion_control(&tcp_agent_cubic);
	tcp_unregister_congestion_control(&tcp_agent_pure);
	tcp_unregister_congestion_control(&tcp_agent);
}

module_init(tcp_agent_init);
module_exit(tcp_agent_exit);
MODULE_DESCRIPTION("Congestion control actuated by a userspace policy");
MODULE_LICENSE("GPL");
