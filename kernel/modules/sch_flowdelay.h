/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Netlink options of sch_flowdelay (mirrored in guest/src/netlink.rs).
 */
#ifndef SCH_FLOWDELAY_H
#define SCH_FLOWDELAY_H

#include <linux/types.h>

/* The observed flow and 64 cross flows (guest/src/plan.rs MAX_CROSS). */
#define FLOWDELAY_MAX_FLOWS 65
/* Ten seconds: far above any path's RTT, low enough to catch unit slips. */
#define FLOWDELAY_MAX_DELAY_NS (10ULL * 1000 * 1000 * 1000)

enum {
	TCA_FLOWDELAY_UNSPEC,
	TCA_FLOWDELAY_MAP,	/* struct tc_flowdelay_entry[], at least one */
	__TCA_FLOWDELAY_MAX,
};

#define TCA_FLOWDELAY_MAX (__TCA_FLOWDELAY_MAX - 1)

/* Packets of TCP destination port @port wait @delay_ns. */
struct tc_flowdelay_entry {
	__u64 delay_ns;
	__u16 port;
	__u16 pad[3];
};

#endif /* SCH_FLOWDELAY_H */
