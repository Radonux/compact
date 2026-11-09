/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Netlink ABI of sch_mmlink (TCA_OPTIONS of a qdisc of kind "mmlink").
 * Mirrored by the guest runner (guest/src/netlink.rs). The qdisc has no
 * change operation: configuration is fixed for the qdisc's lifetime.
 */
#ifndef _SCH_MMLINK_H
#define _SCH_MMLINK_H

enum mmlink_order {
	MMLINK_ORDER_DELAY_LINK = 0,	/* data direction: mm-delay, then mm-link */
	MMLINK_ORDER_LINK_DELAY = 1,	/* ACK direction: mm-link, then mm-delay */
};

/*
 * BASE_NS	u64, required: CLOCK_MONOTONIC ns of trace ms 0.
 * DELAY_NS	u64, one-way delay of the delay line (default 0).
 * ORDER	u8, required: enum mmlink_order.
 * TRACE	binary: trace[i] = delivery opportunities at ms i; at least 2
 *		bytes; the last byte is nonzero and its index is the repeat
 *		period. A netlink attribute holds under 64 KiB, about 65 s.
 * TRACE_PATH	NUL-terminated string: a file holding TRACE's bytes, read once
 *		at init, at most MMLINK_TRACE_MAX bytes. Exactly one of TRACE
 *		and TRACE_PATH is required.
 * LIMIT_PKTS	u32, droptail limit in packets, 0 = none.
 * LIMIT_BYTES	u32, droptail limit in bytes, 0 = none. At least one of the
 *		two limits must be nonzero.
 * DELAY_LIMIT	u32, safety bound on packets in the delay line (default
 *		1 << 20); overflowing it invalidates the episode.
 * LINK_ID	u32, required: copied into every record.
 * LOSS		u32, i.i.d. drop probability times 2^32 (default 0): Mahimahi's
 *		mm-loss, applied before the delay line and the link.
 */
enum {
	TCA_MMLINK_UNSPEC,
	TCA_MMLINK_BASE_NS,
	TCA_MMLINK_DELAY_NS,
	TCA_MMLINK_ORDER,
	TCA_MMLINK_TRACE,
	TCA_MMLINK_LIMIT_PKTS,
	TCA_MMLINK_LIMIT_BYTES,
	TCA_MMLINK_DELAY_LIMIT,
	TCA_MMLINK_LINK_ID,
	TCA_MMLINK_LOSS,
	TCA_MMLINK_TRACE_PATH,
	__TCA_MMLINK_MAX
};

/* Largest trace TRACE_PATH may name: 64 MiB, 18.6 hours at one byte per ms. */
#define MMLINK_TRACE_MAX (64U * 1024 * 1024)

#define TCA_MMLINK_MAX (__TCA_MMLINK_MAX - 1)

#endif /* _SCH_MMLINK_H */
