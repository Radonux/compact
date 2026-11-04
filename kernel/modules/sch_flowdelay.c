// SPDX-License-Identifier: GPL-2.0
/*
 * sch_flowdelay: a fixed extra delay per TCP flow, for flows with different
 * base RTTs on one bottleneck.
 *
 * It sits on the receiver's egress device, so it delays each flow's ACKs
 * (keyed by their destination port, the sender's port) before they enter
 * the path: an access link of the flow's own length on the receiver's side.
 * Packets of other ports, and anything not IPv4 TCP, pass at once.
 *
 * Each configured port has its own FIFO, since all its packets wait the same
 * time and so leave in order. Dequeue takes the ready head that is due
 * first, so packets leave in order of their release times. There is no
 * loss, no rate limit and no reordering within a flow.
 *
 * Locking: as any qdisc, enqueue, dequeue and reset run under the root lock;
 * init and destroy under RTNL.
 *
 * Cost: O(flows) per packet, with at most FLOWDELAY_MAX_FLOWS flows.
 */
#include <linux/ip.h>
#include <linux/module.h>
#include <linux/skbuff.h>
#include <net/ip.h>
#include <net/netlink.h>
#include <net/pkt_sched.h>
#include <net/sch_generic.h>

#include "sch_flowdelay.h"

#define FLOWDELAY_LIMIT BIT(20)

struct flowdelay_sched {
	u32 n;
	u16 port[FLOWDELAY_MAX_FLOWS];
	u64 delay_ns[FLOWDELAY_MAX_FLOWS];
	/* One FIFO per port, then one for packets that pass at once. */
	struct sk_buff_head q[FLOWDELAY_MAX_FLOWS + 1];
	struct qdisc_watchdog watchdog;
};

/* The release time in ns, in skb->cb. */
static u64 *flowdelay_t(const struct sk_buff *skb)
{
	qdisc_cb_private_validate(skb, sizeof(u64));
	return (u64 *)qdisc_skb_cb(skb)->data;
}

static const struct nla_policy flowdelay_policy[TCA_FLOWDELAY_MAX + 1] = {
	[TCA_FLOWDELAY_MAP] = NLA_POLICY_MIN_LEN(sizeof(struct tc_flowdelay_entry)),
};

/* The FIFO of @skb: its TCP destination port's, or q->n to pass at once. */
static u32 flowdelay_class(const struct flowdelay_sched *q, const struct sk_buff *skb)
{
	int off = skb_network_offset(skb);
	struct iphdr _iph;
	const struct iphdr *iph;
	__be16 _ports[2];
	const __be16 *ports;
	u16 dport;
	u32 i;

	if (skb->protocol != htons(ETH_P_IP))
		return q->n;
	iph = skb_header_pointer(skb, off, sizeof(_iph), &_iph);
	if (!iph || iph->protocol != IPPROTO_TCP || ip_is_fragment(iph))
		return q->n;
	ports = skb_header_pointer(skb, off + iph->ihl * 4, sizeof(_ports), _ports);
	if (!ports)
		return q->n;
	dport = ntohs(ports[1]);
	for (i = 0; i < q->n; i++)
		if (q->port[i] == dport)
			return i;
	return q->n;
}

static int flowdelay_enqueue(struct sk_buff *skb, struct Qdisc *sch,
			     struct sk_buff **to_free)
{
	struct flowdelay_sched *q = qdisc_priv(sch);
	u32 c = flowdelay_class(q, skb);

	if (unlikely(sch->q.qlen >= sch->limit))
		return qdisc_drop(skb, sch, to_free);
	*flowdelay_t(skb) = ktime_get_ns() + (c < q->n ? q->delay_ns[c] : 0);
	__skb_queue_tail(&q->q[c], skb);
	sch->q.qlen++;
	qdisc_qstats_backlog_inc(sch, skb);
	return NET_XMIT_SUCCESS;
}

static struct sk_buff *flowdelay_dequeue(struct Qdisc *sch)
{
	struct flowdelay_sched *q = qdisc_priv(sch);
	u64 now = ktime_get_ns(), next = U64_MAX;
	struct sk_buff *skb;
	u32 c, best = q->n + 1;

	for (c = 0; c <= q->n; c++) {
		skb = skb_peek(&q->q[c]);
		if (skb && *flowdelay_t(skb) < next) {
			next = *flowdelay_t(skb);
			best = c;
		}
	}
	if (best > q->n)
		return NULL;
	if (next > now) {
		qdisc_watchdog_schedule_ns(&q->watchdog, next);
		return NULL;
	}
	skb = __skb_dequeue(&q->q[best]);
	sch->q.qlen--;
	qdisc_qstats_backlog_dec(sch, skb);
	qdisc_bstats_update(sch, skb);
	return skb;
}

static void flowdelay_reset(struct Qdisc *sch)
{
	struct flowdelay_sched *q = qdisc_priv(sch);
	u32 c;

	for (c = 0; c <= q->n; c++)
		__skb_queue_purge(&q->q[c]);
	qdisc_watchdog_cancel(&q->watchdog);
}

static int flowdelay_init(struct Qdisc *sch, struct nlattr *opt,
			  struct netlink_ext_ack *extack)
{
	struct flowdelay_sched *q = qdisc_priv(sch);
	struct nlattr *tb[TCA_FLOWDELAY_MAX + 1];
	const struct tc_flowdelay_entry *e;
	u32 c, len;
	int err;

	/* destroy() runs even when init fails, so set up what it touches first. */
	qdisc_watchdog_init(&q->watchdog, sch);
	q->n = 0;
	for (c = 0; c <= FLOWDELAY_MAX_FLOWS; c++)
		__skb_queue_head_init(&q->q[c]);
	sch->limit = FLOWDELAY_LIMIT;

	if (!opt) {
		NL_SET_ERR_MSG_MOD(extack, "flowdelay needs options");
		return -EINVAL;
	}
	err = nla_parse_nested(tb, TCA_FLOWDELAY_MAX, opt, flowdelay_policy, extack);
	if (err)
		return err;
	if (NL_REQ_ATTR_CHECK(extack, opt, tb, TCA_FLOWDELAY_MAP))
		return -EINVAL;
	len = nla_len(tb[TCA_FLOWDELAY_MAP]);
	if (len % sizeof(*e) || len / sizeof(*e) > FLOWDELAY_MAX_FLOWS) {
		NL_SET_ERR_MSG_ATTR(extack, tb[TCA_FLOWDELAY_MAP],
				    "map must hold 1 to 65 whole entries");
		return -EINVAL;
	}
	e = nla_data(tb[TCA_FLOWDELAY_MAP]);
	for (c = 0; c < len / sizeof(*e); c++) {
		if (e[c].delay_ns > FLOWDELAY_MAX_DELAY_NS) {
			NL_SET_ERR_MSG_ATTR(extack, tb[TCA_FLOWDELAY_MAP],
					    "delay above 10 s");
			return -EINVAL;
		}
		q->port[c] = e[c].port;
		q->delay_ns[c] = e[c].delay_ns;
	}
	q->n = len / sizeof(*e);
	return 0;
}

static void flowdelay_destroy(struct Qdisc *sch)
{
	struct flowdelay_sched *q = qdisc_priv(sch);

	/* reset() already purged the queues; the core calls it first. */
	qdisc_watchdog_cancel(&q->watchdog);
}

static struct Qdisc_ops flowdelay_qdisc_ops __read_mostly = {
	.id = "flowdelay",
	.priv_size = sizeof(struct flowdelay_sched),
	.enqueue = flowdelay_enqueue,
	.dequeue = flowdelay_dequeue,
	.peek = qdisc_peek_dequeued,
	.init = flowdelay_init,
	.reset = flowdelay_reset,
	.destroy = flowdelay_destroy,
	.owner = THIS_MODULE,
};
MODULE_ALIAS_NET_SCH("flowdelay");

static int __init flowdelay_module_init(void)
{
	return register_qdisc(&flowdelay_qdisc_ops);
}

static void __exit flowdelay_module_exit(void)
{
	unregister_qdisc(&flowdelay_qdisc_ops);
}

module_init(flowdelay_module_init);
module_exit(flowdelay_module_exit);
MODULE_DESCRIPTION("Fixed extra delay per TCP flow, keyed by destination port");
MODULE_LICENSE("GPL");
