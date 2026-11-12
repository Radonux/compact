# Candidate concepts from the literature

compact derives a controller's concepts from what a trained controller
uses. This page is the other way of choosing them: a review of the
quantities that the literature on congestion control, queue management
and path measurement defines. It serves compact in three ways.

- **A baseline.** A concept set picked from this review, at the same
  budget, is what the derived set must beat
  ([methodology.md](methodology.md#claims), C3).
- **Names.** A fitted formula that matches a quantity defined here gets
  that quantity's name and citation. One that matches nothing here is an
  unfamiliar concept, and is reported as such.
- **Vocabulary.** The basic quantities here are the base variables of the
  grounding library, and the review shows which of them the emulator does
  not label yet ([below](#what-this-means-for-compact)).

A concept must have an exact value in emulation at every step, because
net1 is supervised on it. The sender does not have to be able to measure
it; estimating what the sender cannot measure directly is net1's job. A
basic concept is a quantity the link itself determines and the emulator
logs directly, such as capacity, the queue or a packet's delay. A
compositional concept is a fixed formula over basic concepts in a short
window, such as the fraction of capacity in use.

For each concept the page says what it means, how the emulator records
it, whether the sender can measure it, and how the cited source defines
it. It reviews 78 papers and RFCs ([method](#how-this-review-was-done)).
The definitions come from the cited sources. The judgments about ground
truth and about what the sender can see, and the last section's
inferences, are ours.

## In short

- The basic concepts (capacity, link usage, delay, loss) and the simplest
  compositions of them (utilization, available capacity) have one set of
  definitions, shared by the IETF
  metrics RFCs and the bandwidth-estimation literature. Rates are defined
  over an explicit interval and delays per packet, and each is exact from
  the emulator's link records.
- expliCCit's busy fraction, as the emulator labels it, is the standard link
  utilization: bits sent over bits the link could have sent, in the window
  (RFC 5136).
- Most compositional concepts with an exact formula come from
  congestion control and queue management schemes that run at the router.
  They measure how much capacity is unused, how much of the queue persists
  over a round trip, how full the path is relative to an empty queue, each
  flow's fair share, and whether the queue is growing. A router computes
  them from its own link state, so the emulator can compute them exactly.
- Utilization stops at 1 once the queue never empties, so it cannot say how
  overloaded a link is. Arrival rate over capacity (the load factor) keeps
  growing with overload.
- Concepts about the cause of congestion (is the competing traffic elastic,
  do two flows share a bottleneck, did this flow fill the queue itself, was
  this loss congestion) need hundreds of milliseconds to seconds of data at
  the sender. The emulator knows their true value outright.
- The concept-based explainers for networking (Patel et al. 2024 and Agua,
  from one author group) name their concepts but give none a formula. A
  concept's value there is the similarity between text embeddings of an
  LLM-written description of the state and of the concept's name. Closed
  forms over measured quantities are what that work lacks, and this
  document supplies candidates for them.

## How to read the tables

Each section opens with what its concepts mean, in a table, and ends with
their definitions, in a list. The definitions are the implementation
reference.

In the tables:

- "Truth" says how the emulator records the value. The emulator's link qdisc
  logs every arrival, departure and drop (with its cause and its flow), the
  queue after each event, one delivery-opportunity record per millisecond,
  and the buffer limit. "Exact" means a function of those records in the
  window; "exact, per flow" also needs each packet's flow; "model" means the
  value exists only in a model the emulator does not run.
- "Sender" says whether the sender can measure the value: "yes", "estimate"
  (biased, delayed or conditional, as the cited estimators are) or "no".
- "Label" is the emulator's existing ground-truth label, if there is one
  ([architecture.md](architecture.md#ground-truth-labels)).

Notation used in the definitions:

| Symbol | Meaning |
|---|---|
| $`C`$ | link capacity (bit/s) |
| $`y`$ | arrival rate at the link, all flows (bit/s) |
| $`\mu`$ | departure rate from the link (bit/s) |
| $`q`$ | queue (bytes or packets), $`B`$ the buffer limit |
| $`R_m`$ | round-trip propagation delay (base RTT) |
| $`N`$ | number of active flows; $`x_i`$ the rate of flow $`i`$ |
| $`[T, T+I]`$ | the window over which a value is defined |

## The path at rest

These describe the path with no queue: how fast the bottleneck is, how far
away the receiver is, and how much it can hold. They change only when the
path changes.

| Concept | What it means | Truth | Sender | Label |
|---|---|---|---|---|
| Capacity | the most the bottleneck can deliver per second in the window | exact | estimate | capacity_bps |
| Base RTT | round-trip time with empty queues: propagation plus transmission | exact | estimate | base_rtt_ns, one_way_delay_ns |
| Buffer size | how much the bottleneck queue can hold before it drops | exact | estimate, after one overflow | buffer_pkts, buffer_bytes |
| Bandwidth-delay product (BDP) | data in flight when the link is full and the queue empty | exact | estimate | none |
| Non-congestive delay | delay variation that does not come from the bottleneck queue (link-layer jitter, ACK aggregation) | exact: zero on a constant delay line | no | none |

Definitions:

- Capacity, RFC 5136 [32]: the maximum IP-layer bits that can be sent and
  correctly received over link $`L`$ during $`[T, T+I]`$, divided by $`I`$,
  written $`C(L,T,I)`$. A path's capacity is that of its narrow link,
  $`C = \min_i C_i`$ [32] [23]. In the emulator: delivery opportunities in
  the window times 1500 bytes, over $`I`$.
- Capacity, as senders estimate it: the spacing of a back-to-back pair,
  $`C = L/\delta`$ [22] [11]; a windowed maximum of the delivery rate (BBR,
  from its slides [12]); the 95th percentile of 100 ms receive rates [45];
  a low-pass filter of the ACK rate [77]. Long packet trains measure less
  than capacity under cross traffic, $`R = C_1 / (1 + u_1 C_1/C_0)`$ on one
  hop [22] [23].
- Base RTT: $`R_m`$ in CCAC's model [5]. Senders estimate it as a minimum:
  of all RTT samples (Vegas [10]), of recent one-way delays (LEDBAT [38]),
  over 10 s (PBE-CC [56], and Copa, pending), or the mode of delays in
  windows below the bottleneck rate (iBox [45]). A minimum is right only
  once the queue has emptied inside its window.
- Buffer size: $`\beta`$ in CCAC; loss is possible only once the queue
  exceeds it [5]. iBox estimates it as $`B = b\,(d_{\mathrm{full}} -
  d_{\mathrm{empty}})`$, bottleneck rate times the delay difference between
  a full and an empty buffer [45].
- BDP: $`C R_m`$ (CCmatic, pending). The same product appears as the pipe
  size [49], the rule-of-thumb buffer $`\mathrm{RTT} \times C`$ [25], and
  the number of packets in the system at maximum power [31].
- Non-congestive delay: CCAC bounds it by $`D`$, the maximum per-packet
  jitter [5]; CCmatic writes $`\mathrm{RTT}(t) = R_m + q_{\mathrm{del}}(t) +
  \mathrm{jitter}(t)`$ with $`0 \le \mathrm{jitter}(t) \le D`$ (pending).

## How busy the link is

These describe the load on the bottleneck in one window. Excess demand and
idle time are our names for simple differences; the load factor comes from
ATM explicit-rate control.

| Concept | What it means | Truth | Sender | Label |
|---|---|---|---|---|
| Arrival rate | bits per second entering the bottleneck queue, all flows | exact | no | arrival_rate_bps |
| Usage (departure rate) | bits per second the link actually sent, all flows | exact | estimate, with in-network telemetry | none (capacity times busy fraction) |
| Receive rate | the observed flow's share of the usage | exact, per flow | yes | receive_rate_bps |
| Utilization (busy fraction) | the fraction of the link's capacity that was used | exact | estimate | busy_fraction |
| Available capacity | capacity left unused by all flows (spare capacity); never negative | exact | estimate | none |
| Excess demand | arrival rate minus capacity: positive means the queue grows | exact | estimate | none |
| Load factor | arrival rate over a target rate just below capacity; unlike utilization it exceeds 1 under overload | exact | no | none |
| Idle time | the fraction of the window in which the link had nothing to send | exact | no | none (1 minus busy fraction) |

Definitions:

- Usage, RFC 5136 [32]: $`\mathrm{Used}(L,T,I)`$, the IP-layer bits from
  any source correctly received over $`L`$ during $`[T, T+I]`$, divided by
  $`I`$. HPCC's departure rate from telemetry is
  $`\mathrm{txRate} = \Delta \mathrm{txBytes} / \Delta t`$ [17].
- Utilization, RFC 5136 [32]:
  $`U(L,T,I) = \mathrm{Used}(L,T,I) / C(L,T,I)`$. Equivalently, the time
  average of the 0/1 busy indicator over the averaging timescale $`\tau`$
  [23]:

  ```math
  \bar u(t-\tau, t) = \frac{1}{\tau} \int_{t-\tau}^{t} u(x)\, dx, \qquad u(x) \in \{0, 1\}
  ```

  On a trace-driven link (Mahimahi, pending), this is bytes sent over bytes
  the delivery opportunities offered, which is the emulator's busy fraction.
- Available capacity, RFC 5136 [32]:
  $`A(L,T,I) = C(L,T,I)\,\bigl(1 - U(L,T,I)\bigr)`$; the tight link is the
  one with the least [32] [23] [21]. In fluid form, the time average of
  $`C - \lambda`$ over the window, with $`\lambda`$ the cross-traffic rate
  [24] [69] [70].
- Excess demand: $`y - C`$. XCP's spare bandwidth is this quantity with the
  opposite sign (its prose sign convention is inconsistent) [42]; RCP uses
  $`C - y(t)`$ [14]; Nimbus uses $`\mu - S - \hat z`$ [19].
- Load factor, OSU's ATM scheme [63]: $`z = y / C_{\mathrm{target}}`$, with
  the target set to 85% to 95% of link bandwidth.
- Idle time: $`1 - U`$. BLUE lowers its marking probability on each link
  idle event and raises it on each overflow loss [73].

## The queue

These describe the bottleneck queue: how big it is, how long packets wait,
whether it is a standing queue or a passing burst, and where it is heading.
From the standing queue down, the rows are compositional concepts that
router schemes define.

| Concept | What it means | Truth | Sender | Label |
|---|---|---|---|---|
| Queue | data waiting in the bottleneck buffer now | exact | estimate | queue_pkts, queue_bytes |
| Free buffer | buffer space left before drops start | exact | estimate | free_buffer_pkts, buffer_fill |
| Sojourn time | how long a packet waited in the queue | exact | estimate (RTT minus base RTT) | queue_delay_ns, queue_delay_mean_ns |
| Standing queue | the part of the queue that did not drain within a round trip: its minimum over a recent window | exact | estimate | none |
| Average queue | a smoothed queue, as RED and DECbit use it | exact | no | none |
| Queue growth rate | how fast the queue grows or drains, in seconds of delay per second | exact | estimate (RTT gradient) | none |
| Little's-law delay | the queue divided by the departure rate: how long a packet arriving now will wait | exact | no | none |
| Normalized inflight | how full the path is: 1 when the link is busy and its queue empty, plus the queue as a fraction of the BDP | exact | estimate, with telemetry | none |
| Normalized power | (queue plus BDP) over BDP, times arrival rate over capacity: 1 when the link is just full; it rises with the queue, and with the arrival rate before the queue has built | exact | estimate | none |
| Classic power | throughput over delay; highest at the knee, where the link is full and the queue still empty | exact | estimate | none |
| Virtual queue | the queue a slightly slower link would have; warns before the real queue builds | exact | no | none |

Definitions:

- Queue, CCAC [5]: $`Q(t) = A(t) - L(t) - S(t)`$ over cumulative arrivals,
  losses and service. Free buffer is $`B - q`$ and buffer fill is $`q/B`$;
  CCAC's loss condition is the same comparison with $`\beta`$ [5].
- Sojourn time, CoDel [40]: "the time a packet departs the buffer minus the
  time the packet arrived at the buffer".
- Standing queue: XCP takes the minimum queue seen by an arriving packet
  during the last propagation delay [42]. CoDel acts when the sojourn time
  has stayed above TARGET = 5 ms for INTERVAL = 100 ms [40]. Copa's
  RTTstanding is the minimum RTT over the last $`\mathrm{srtt}/2`$
  (pending).
- Average queue: RED updates $`\bar q \leftarrow (1-w_q)\,\bar q + w_q\,q`$
  on each arrival, $`w_q = 0.002`$ in most of its simulations [26]. DECbit
  averages the queue over the previous busy-and-idle cycle and the current
  one: the area under $`q(t)`$ since the start of the previous busy period,
  over the elapsed time (DEC report version of [72]).
- Queue growth rate, TIMELY's model [15], with the queue in seconds:
  $`\dot q = (y - C) / C`$. The ATM EPRCA scheme reads overload from the
  queue change over K cells [63].
- Little's-law delay, PIE [27]: $`\hat d = q / \mu`$, from Little's law
  $`L = \lambda W`$, with $`\lambda`$ there the arrival rate [53].
- Normalized inflight, HPCC [17], for link $`j`$ with capacity $`C_j`$
  (HPCC writes it $`B_j`$) and base RTT $`T`$:

  ```math
  U_j = \frac{q_j}{C_j T} + \frac{\mathrm{txRate}_j}{C_j}
  ```

  Over the emulator's labels this is queue over (capacity times base RTT),
  plus the busy fraction.
- Normalized power, PowerTCP [60], with $`b`$ the bottleneck bandwidth,
  $`\tau`$ the base RTT and $`\lambda`$ the arrival rate:

  ```math
  \Gamma(t) = \bigl(q(t) + b\,\tau\bigr)\,\lambda(t), \qquad \frac{\Gamma}{b^2 \tau} = \Bigl(1 + \frac{q}{b\tau}\Bigr)\frac{\lambda}{b}
  ```

  The right-hand form is our algebra. PowerTCP computes $`\lambda`$ as the
  queue's growth plus the departure rate, which is why it reacts to growth.
- Classic power: $`P = \mathrm{throughput}^{\alpha} / \mathrm{delay}`$,
  highest at the knee [66]; $`P(\rho) = \rho / (\mu T(\rho))`$, which for
  M/M/1 peaks at $`\rho = 0.5`$ with one packet in the system, where
  $`\rho`$ is the utilization and $`\mu`$ the service rate [31].
- Virtual queue, AVQ [74]: on each arrival
  $`VQ \leftarrow \max\bigl(VQ - \tilde C\,(t - s), 0\bigr)`$, and the packet
  is marked if $`VQ + b > B`$, with $`b`$ the packet's size, $`B`$ the
  buffer and $`s`$ the previous arrival time. The virtual capacity follows
  $`\dot{\tilde C} = \alpha\,(\gamma C - \lambda)`$, with $`\gamma`$ the
  target utilization, $`\lambda`$ the arrival rate and $`\alpha`$ a gain.

## Delay and loss at the endpoints

These are what the endpoints see of the queue: delay, its trend and its
variation, losses and marks. Most learned controllers take them as inputs.
Their ground truth comes from per-packet records.

| Concept | What it means | Truth | Sender | Label |
|---|---|---|---|---|
| One-way delay | time from sending a packet to its arrival | exact | partly: needs synchronized clocks | one_way_delay_ns (the delay line) |
| Delay variation | the difference in delay between two packets; jitter is its smoothed size | exact | partly | none |
| Delay gradient | whether delay is rising or falling, relative to its level | exact | yes | none |
| Latency ratio | current RTT over base RTT | exact | yes | none |
| Loss rate | the fraction of packets dropped, split by cause | exact | yes, late and ambiguous | drop_fraction, random_loss_rate |
| Marked fraction | the fraction of packets that saw the queue above a threshold | exact | yes, with marking | none |

Definitions:

- One-way delay, RFC 7679 [33]: $`dT`$, where the first bit leaves at wire
  time $`T`$ and the last bit arrives at $`T + dT`$; undefined if the packet
  does not arrive within $`T_{\max}`$. Its minimum, median and percentiles
  are defined over a stream [33].
- Delay variation, RFC 3393 [35]: $`\mathrm{ipdv} = dT_2 - dT_1`$ for a
  selected pair. Jitter is an EWMA of $`\lvert \mathrm{ipdv} \rvert`$ with
  gain 1/16; peak-to-peak is maximum minus minimum delay per subinterval.
- Delay gradient: TIMELY smooths the RTT difference with an EWMA and
  divides by the minimum RTT [15]; Aurora uses the derivative of latency
  over a monitor interval [68]; Vivace fits a regression slope (pending).
  Jain's normalized delay gradient is positive above the knee [75]:

  ```math
  \mathrm{NDG} = \frac{dD/dW}{D/W} \approx \frac{D - D_{\mathrm{old}}}{D + D_{\mathrm{old}}} \cdot \frac{W + W_{\mathrm{old}}}{W - W_{\mathrm{old}}}
  ```

- Latency ratio: Aurora's monitor-interval mean latency over the least such
  mean so far [68]; Remy's latest RTT over the minimum RTT [18].
- Loss, RFC 7680 [34]: 0 if the packet arrives within $`T_{\max}`$, 1 if
  not; the loss ratio is the mean over a stream. The emulator records each
  drop's cause (full queue or random loss).
- Marked fraction, DCTCP [44]: $`F`$, the fraction of packets marked in the
  last window of data, smoothed as $`\alpha \leftarrow (1-g)\,\alpha + g F`$,
  which estimates the probability that the queue exceeds $`K`$. RFC 3168
  sets the congestion-experienced mark where the router would otherwise drop
  [59].
- Loss and throughput: $`\mathrm{BW} = (\mathrm{MSS}/\mathrm{RTT}) \cdot
  k/\sqrt{p}`$ for loss rate $`p`$, with $`k = \sqrt{3/2}`$ (Mathis et al.
  call the constant $`C`$) [41]; Padhye et al. add timeouts [13].

## The other flows

These describe who else is on the link and how the capacity is split. They
matter whenever the observed flow shares the bottleneck.

| Concept | What it means | Truth | Sender | Label |
|---|---|---|---|---|
| Cross-traffic rate | arrival rate of all other flows | exact, per flow | estimate | arrival_rate_bps less flow_arrival_rate_bps; cross_traffic_share (as a share) |
| Active flows | how many flows have packets at the bottleneck in the window | exact, per flow | no | active_flows |
| Fair share | capacity divided equally among the active flows | exact, per flow | no | none |
| Max-min share | the fair share after flows that need less have been served | exact, per flow | no | none |
| Fairness index | how evenly the link is split: 1 for equal shares, $`k/n`$ when $`k`$ of $`n`$ flows share equally and the rest get nothing | exact, per flow | no | none |

Definitions:

- Cross-traffic rate: $`r_i`$, the average rate of cross traffic entering
  link $`i`$ [22] [23]. Senders estimate it as $`\hat z = \mu S / R - S`$
  from their send and receive rates (Nimbus [19]) or from queue changes and
  drain time (iBox [45]).
- Active flows: $`N(t)`$, the number of ongoing flows [14]; also the $`N`$
  of DCTCP's queue analysis [44] and of the buffer-sizing rule
  $`B = (\mathrm{RTT} \times C)/\sqrt{n}`$ [25]. RCP's router estimates it
  as $`\hat N = C / R(t - T)`$ [14].
- Fair share: $`C / N(t)`$, the processor-sharing rate [14]; the OSU scheme
  uses the target rate over the number of active sources [63].
- Max-min share, the MIT explicit-rate scheme [63], iterated until the set
  of underloading flows stops changing:

  ```math
  \text{share} = \frac{C - \sum_{\text{underloading}} x_i}{N - N_{\text{underloading}}}
  ```

  Users that share a bottleneck get equal shares [30] [54].
- Fairness index [30] [78], for $`n`$ flows:
  $`F = \bigl(\sum_i x_i\bigr)^2 / \bigl(n \sum_i x_i^2\bigr)`$.

## The cause of congestion

These say why the queue builds rather than how much. A sender needs
hundreds of milliseconds to seconds to estimate them; in the emulator each
is a fact of the episode.

| Concept | What it means | Window at the sender | Truth |
|---|---|---|---|
| Elasticity | whether the competing traffic backs off when we send more (a TCP flow does, a fixed-rate stream does not) | about 5 s [19] | exact: each competitor's type is known |
| Shared bottleneck | whether two flows wait in the same queue | 350 ms intervals over 30 to 50 intervals [37]; about 1 s [50] | exact: known from the topology |
| Self-induced congestion | whether this flow filled a previously empty queue itself, rather than meeting a queue other traffic built | slow start [20] | exact: the queue before the flow ramped up |
| Loss cause | whether a loss came from a full queue or from a random error | the delay history around each loss [51] | exact: each drop record names its cause |
| Loss event rate | how often loss events occur, counting several losses within one round trip as one event, as TFRC does | eight loss intervals [39] | exact |
| ACK compression | ACKs bunched together by a queue on the return path, which breaks self-clocking | per burst [49] | exact: both directions are recorded |

Definitions:

- Elasticity, Nimbus [19]: pulse the sending rate at $`f_p = 5`$ Hz and
  compute, from the FFT of the cross-traffic estimate $`\hat z`$,

  ```math
  \eta = \frac{\lvert \mathrm{FFT}_{\hat z}(f_p) \rvert}{\max_{f \in (f_p, 2 f_p)} \lvert \mathrm{FFT}_{\hat z}(f) \rvert}
  ```

  The cross traffic is inelastic if $`\eta < 2`$.
- Shared bottleneck, RFC 8382 [37]: four statistics of one-way delay and
  loss per flow, compared across flows. They are skew_est (samples below
  minus above the mean delay, over the count), var_est (mean absolute
  deviation), freq_est (significant crossings of the mean, over the number
  of intervals) and pkt_loss. The base interval is 350 ms; RFC 8382 uses 30
  intervals for the mean delay, skew_est and var_est, and 50 for freq_est
  and pkt_loss.
  Rubenstein et al. compare the delay or loss correlation across two flows
  with that within one flow [50].
- Self-induced congestion, Sundaresan et al. [20]: from RTTs during slow
  start, $`\mathrm{NormDiff} = (\mathrm{RTT}_{\max} -
  \mathrm{RTT}_{\min}) / \mathrm{RTT}_{\max}`$ and the coefficient of
  variation of RTT. A flow is labelled self-induced when its slow-start
  throughput exceeds 0.8 of capacity.
- Loss cause, Cen et al. [51]: the Biaz rule (the gap around a loss against
  multiples of the minimum inter-arrival time), the Spike rule (relative
  one-way delay against thresholds between its minimum and maximum) and
  ZigZag (loss count and delay against its mean and deviation).
- Loss event rate, RFC 5348 [39]: a loss event is one or more losses within
  one RTT, and $`p = 1 / \bar I`$, with $`\bar I`$ the weighted mean of the
  last eight loss intervals.

## Models rather than measurements

These are defined through a model of the network, not a window of records,
so the emulator cannot record them. They are useful as targets only through
that model.

| Concept | What it means | Sources |
|---|---|---|
| Link price | a running congestion cost per link that rises while demand exceeds capacity and the queue exceeds its target | [28] [48] [29] |
| Proportional fairness | the allocation that maximizes the sum of log rates under the capacity limits | [29] |
| Knee and cliff | the knee is the load past which throughput hardly grows but delay does; the cliff is where throughput collapses | [66] [31] |
| Bottleneck structure | which link limits each flow, and how a change to one link propagates to the others | [67] [43] |
| Link-rate bounds | the range of link rates consistent with the ACKs seen so far | CCmatic (pending) |
| Link-rate forecast | a probability distribution over the next link rate, updated from arrivals | Sprout (pending) |

Definitions:

- Link price: REM updates
  $`p_l(t+1) = \bigl[p_l(t) + g\,(a_l (b_l(t) - b_l^*) + x_l(t) - c_l(t))\bigr]^+`$
  [28]; Low and Lapsley update
  $`p_l(t+1) = \bigl[p_l(t) + \gamma\,(x^l - c_l)\bigr]^+`$, and a path's
  price is the sum of its links' prices [48]. Given its constants a price is
  computable from the records, but it carries state from window to window.
- Proportional fairness, Kelly et al. [29]: a feasible $`x`$ such that
  $`\sum_r (x_r^* - x_r)/x_r \le 0`$ for every feasible $`x^*`$; the optimum
  of $`\sum_r w_r \log x_r`$ subject to $`Ax \le C`$.
- Link-rate bounds, CCmatic (pending): from an ACK rate $`r`$ over an
  interval $`T`$, $`rT \in [C(T - D), C(T + D)]`$ when the sending rate is at
  least $`C`$, inverted into bounds on $`C`$.

## Concepts used to explain learned controllers

Two papers from one group explain learned controllers through named
concepts [1] [4]. The 2024 paper [1] covers adaptive bitrate only: it asks
an LLM to summarize a survey and propose concepts, noting that simply
asking for concepts "does not produce a usable set". Agua [4] extends this
to congestion control with eight concepts:

- increasing packet loss and decreasing packet loss;
- stable network conditions and volatile network conditions;
- rapidly increasing latency and rapidly decreasing latency;
- low network utilization and high network utilization.

Neither paper defines a concept by a formula. A concept's value for a state
is the cosine similarity between text embeddings of an LLM's description of
the state and of the concept, binned as low (0 to 0.2), medium (0.2 to 0.6)
and high (0.6 to 1.0) [4]. Agua's reported fidelity, 0.932 to 1.000, is
agreement with the controller [4], not agreement with the path.

The other explanation and verification papers in the review express a
controller's behavior in its own inputs:

- properties over Aurora's latency gradient, latency ratio and sending
  ratio [47];
- properties over Orca's queueing delay and loss [61];
- the components of the reward [64];
- decision trees over input features [58];
- closed-form policies found by symbolic regression [62].

Mahak describes a controller's performance over ground-truth link
parameters: capacity, minimum RTT, buffer size and a bandwidth change ratio
[7].

Agua's eight names map onto closed forms from this document. The mapping
is ours:

- increasing or decreasing loss: the change in loss rate between windows
  [34];
- rapidly increasing or decreasing latency: the queue growth rate [15];
- low or high utilization: utilization [32];
- stable or volatile conditions: the variation of capacity or of available
  capacity over a window.

## What this means for compact

This section is our inference, not a finding of the review.

**The library's base variables.** Nearly every compositional concept
above is a closed form over a few basic ones: capacity, arrival rate,
departure rate, queue, sojourn time, base RTT, buffer, active flows and
the cause of each drop. The emulator labels these
([architecture.md](architecture.md#ground-truth-labels)), including the
three that expliCCit's seven concepts do not need: the arrival rate of
all flows (`arrival_rate_bps`), the observed flow's own arrival rate at
the bottleneck (`flow_arrival_rate_bps`) and the number of active flows
(`active_flows`).

Most concepts in the busy-link, queue and other-flow sections are then
one ratio or product of base variables, which is exactly what the
library's operators generate. For example:

- available capacity is capacity times (1 minus busy fraction);
- the load factor is arrival rate over capacity;
- the fair share is capacity over active flows;
- the flow's position relative to its fair share is receive rate over
  fair share.

Without those three, grounding could not return any concept that depends on demand
or on the number of flows, whatever the latents encode. This is the
vocabulary objection in concrete form, and the sensitivity check (C4) can
only test variables that are in the library to begin with.

**Windows.** Each definition should name its window, as RFC 5136 does
[32]: these values change with it [23]. The emulator's labels use a
trailing 100 ms. At 10 ms a 12 Mbit/s link offers about ten 1500-byte
delivery opportunities, so utilization moves in steps of 0.1, and at 1.2
Mbit/s it offers about one. The library's windowed operators should
therefore start no shorter than the labels' own window.

**A composite or its bases, not both.** A concept computed exactly from
other concepts adds nothing net2 could not compute itself, and it puts an
exact dependency among the set's coordinates. Any per-concept attribution
then divides credit between them in a way set by the attribution method,
not by the policy. The sparse basis of step 4 should be chosen with this
in mind, and so should the hand-picked baseline.

**Where an estimator can beat the published heuristics.** The
cause-of-congestion concepts are plain facts in the emulator, while a
sender needs seconds of probing or a slow start to infer them. If the
latents encode one of them, the heuristics here say how hard it is to
estimate from the sender.

**What the review did not find.** No included paper defines a
congestion-control concept by a formula over path state and checks it
against ground truth, and none compares candidate concept sets by how
well a policy on them performs. Those two gaps are compact's question.

## How this review was done

The question and the criteria were fixed before the first search. Papers
qualified if they defined, measured, estimated or used a quantity of path,
link, queue or competing-traffic state, or defined concepts to explain or
verify a learned networking controller, and if they stated the definition.

Excluded were:

- radio-only quantities;
- application-level metrics only;
- explainable ML outside networking;
- theses and magazine pieces;
- controllers whose quantities were all defined by an earlier included
  paper (a criterion added during screening, and logged).

The searches ran on 2026-11-11 through the lit-review skill's script, on
Crossref and arXiv only. OpenAlex's free daily quota for the network was
spent and Semantic Scholar refused requests without a key. 120 queries were
logged:

- the seeds (Orca [2], CCmatic, Patel et al. 2024 [1], Agua [4]) and classic
  sources, looked up by title;
- topical arXiv queries in cs.NI;
- one backward snowball cycle through the papers' reference lists.

| Step | Records |
|---|---|
| Identified | 197 |
| Excluded on title and abstract | 116: 45 off topic, 43 redundant controllers, 11 explainable ML outside networking, 11 theses or magazine pieces, 5 duplicates, 1 application-level |
| Excluded on full text | 3 |
| Included | 78: 61 read in full, 17 at abstract level |

Extraction ran in eight batches, one per theme. Each batch read its
papers' full text where a copy could be fetched and recorded every quantity
with its formula and its location in the paper. The records were then
reviewed, and RFC 5136's and RFC 8382's formulas were checked against the
RFC text. Every citation resolves to an included record, and every DOI
resolves.

Limits:

- The corpus leans toward known work, since most sources were looked up by
  title, and no forward snowball (papers citing the seeds) was possible.
- Five USENIX papers have no DOI, so they could not enter the corpus without
  OpenAlex. They were read in full but carry no number here: CCmatic, Copa,
  PCC Vivace, Sprout and Mahimahi.
- Orca's full text could not be fetched [2]; its inputs are known only from
  its released code. BBR's definitions [12] come from its IETF slides.
- Some formulas were recovered from PDFs whose text layer lost the symbols:
  RED [26], Chiu and Jain's fairness index [30] and Cen et al.'s thresholds
  [51].
- For some records, the text read was another version of the indexed
  paper, so the record is held at abstract level:
  - TCP congestion signatures [20]: the IMC 2017 text;
  - DECbit [72], BLUE [73], FAST [76] and Westwood [77]: technical reports
    or related articles;
  - Jain, Chiu and Hawe's report [78]: a scan, read for 6 of 38 pages.

Gaps in the corpus:

- Among the explanation papers, none defines a congestion control concept
  by a formula over path state or checks a concept against ground truth.
- No included paper compares candidate concept sets by how well a policy on
  them performs.
- The number of competing flows appears only as a model variable or a
  router estimate; no included paper estimates it end to end.
- No included paper evaluates these concepts at a 10 ms window.

## Pending (read, not yet in the corpus)

| Paper | Full text |
|---|---|
| Agarwal et al., Towards Provably Performant Congestion Control (CCmatic), NSDI 2024 | https://www.usenix.org/system/files/nsdi24-agarwal-anup.pdf |
| Arun and Balakrishnan, Copa: Practical Delay-Based Congestion Control for the Internet, NSDI 2018 | https://www.usenix.org/system/files/conference/nsdi18/nsdi18-arun.pdf |
| Dong et al., PCC Vivace: Online-Learning Congestion Control, NSDI 2018 | https://www.usenix.org/system/files/conference/nsdi18/nsdi18-dong.pdf |
| Winstein et al., Stochastic Forecasts Achieve High Throughput and Low Delay over Cellular Networks (Sprout), NSDI 2013 | https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final113.pdf |
| Netravali et al., Mahimahi: Accurate Record-and-Replay for HTTP, USENIX ATC 2015 | https://www.cs.princeton.edu/~ravian/publications/mahimahi_atc.pdf |

## References

[1] Patel et al., Toward Trustworthy Learning-Enabled Systems with Concept-Based Explanations. ACM Workshop on Hot Topics in Networks (HotNets), 2024. https://doi.org/10.1145/3696348.3696894

[2] Abbasloo et al., Classic Meets Modern: A Pragmatic Learning-Based Congestion Control for the Internet. ACM SIGCOMM, 2020. https://doi.org/10.1145/3387514.3405892 (abstract level)

[3] Khan et al., The case for model-driven interpretability of delay-based congestion control protocols. 2021. https://arxiv.org/abs/2102.04911 (abstract level)

[4] Patel et al., Agua: A Concept-Based Explainer for Learning-Enabled Systems. ACM SIGCOMM, 2025. https://doi.org/10.1145/3718958.3754341

[5] Arun et al., Toward formally verifying congestion control behavior. ACM SIGCOMM, 2021. https://doi.org/10.1145/3452296.3472912

[6] Yen et al., Computers Can Learn from the Heuristic Designs and Master Internet Congestion Control. ACM SIGCOMM, 2023. https://doi.org/10.1145/3603269.3604838

[7] Pazhooheshy et al., Harnessing ML For Network Protocol Assessment: A Congestion Control Use Case. ACM Workshop on Hot Topics in Networks, 2023. https://doi.org/10.1145/3626111.3628182 (abstract level)

[8] Jacobson, Congestion avoidance and control. Symposium proceedings on Communications architectures and protocols, 1988. https://doi.org/10.1145/52324.52356

[9] Paxson and Allman, Computing TCP's Retransmission Timer. 2000. https://doi.org/10.17487/rfc2988

[10] Brakmo and Peterson, TCP Vegas: end to end congestion avoidance on a global Internet. IEEE Journal on Selected Areas in Communications, 1995. https://doi.org/10.1109/49.464716

[11] Keshav, A control-theoretic approach to flow control. ACM SIGCOMM, 1991. https://doi.org/10.1145/115992.115995

[12] Cardwell et al., BBR: Congestion-based congestion control. Communications of the ACM, 2017. https://doi.org/10.1145/3009824 (abstract level)

[13] Padhye et al., Modeling TCP throughput: A simple model and its empirical validation. ACM SIGCOMM, 1998. https://doi.org/10.1145/285243.285291

[14] Dukkipati et al., Processor Sharing Flows in the Internet. Lecture Notes in Computer Science Quality of Service: IWQoS 2005, 2005. https://doi.org/10.1007/11499169_22

[15] Mittal et al., TIMELY: RTT-based Congestion Control for the Datacenter. ACM SIGCOMM, 2015. https://doi.org/10.1145/2785956.2787510

[16] Kumar et al., Swift: Delay is Simple and Effective for Congestion Control in the Datacenter. ACM SIGCOMM, 2020. https://doi.org/10.1145/3387514.3406591

[17] Li et al., HPCC: High precision congestion control. ACM SIGCOMM, 2019. https://doi.org/10.1145/3341302.3342085

[18] Winstein and Balakrishnan, TCP ex machina: Computer-generated congestion control. ACM SIGCOMM, 2013. https://doi.org/10.1145/2486001.2486020

[19] Goyal et al., Elasticity detection: A building block for internet congestion control. ACM SIGCOMM, 2022. https://doi.org/10.1145/3544216.3544221

[20] Sundaresan et al., TCP Congestion Signatures. Applied Networking Research Workshop, 2018. https://doi.org/10.1145/3232755.3232760 (abstract level)

[21] Jain and Dovrolis, End-to-end available bandwidth: Measurement methodology, dynamics, and relation with TCP throughput. IEEE/ACM Transactions on Networking, 2003. https://doi.org/10.1109/tnet.2003.815304

[22] Dovrolis et al., What do packet dispersion techniques measure? IEEE INFOCOM, 2001. https://doi.org/10.1109/infcom.2001.916282

[23] Prosad et al., Bandwidth estimation: metrics, measurement techniques, and tools. IEEE Network, 2003. https://doi.org/10.1109/mnet.2003.1248658

[24] Strauss et al., A measurement study of available bandwidth estimation tools. ACM SIGCOMM, 2003. https://doi.org/10.1145/948205.948211

[25] Appenzeller et al., Sizing router buffers. ACM SIGCOMM, 2004. https://doi.org/10.1145/1030194.1015499

[26] Floyd and Jacobson, Random early detection gateways for congestion avoidance. IEEE/ACM Transactions on Networking, 1993. https://doi.org/10.1109/90.251892

[27] Pan et al., Proportional Integral Controller Enhanced (PIE): A Lightweight Control Scheme to Address the Bufferbloat Problem. 2017. https://doi.org/10.17487/rfc8033

[28] Athuraliya et al., REM: active queue management. IEEE Network, 2001. https://doi.org/10.1109/65.923940

[29] Kelly et al., Rate control for communication networks: shadow prices, proportional fairness and stability. Journal of the Operational Research Society, 1998. https://doi.org/10.1038/sj.jors.2600523

[30] Chiu and Jain, Analysis of the increase and decrease algorithms for congestion avoidance in computer networks. Computer Networks and ISDN Systems, 1989. https://doi.org/10.1016/0169-7552(89)90019-6

[31] Kleinrock, Internet congestion control using the power metric: Keep the pipe just full, but no fuller. Ad Hoc Networks, 2018. https://doi.org/10.1016/j.adhoc.2018.05.015

[32] Chimento and Ishac, Defining Network Capacity. 2008. https://doi.org/10.17487/rfc5136

[33] Almes et al., A One-Way Delay Metric for IP Performance Metrics (IPPM). 2016. https://doi.org/10.17487/rfc7679

[34] Almes et al., A One-Way Loss Metric for IP Performance Metrics (IPPM). 2016. https://doi.org/10.17487/rfc7680

[35] Demichelis and Chimento, IP Packet Delay Variation Metric for IP Performance Metrics (IPPM). 2002. https://doi.org/10.17487/rfc3393

[36] Morton et al., Metrics and Methods for One-Way IP Capacity. 2021. https://doi.org/10.17487/rfc9097

[37] Ferlin et al., Shared Bottleneck Detection for Coupled Congestion Control for RTP Media. 2018. https://doi.org/10.17487/rfc8382

[38] Shalunov et al., Low Extra Delay Background Transport (LEDBAT). 2012. https://doi.org/10.17487/rfc6817

[39] Floyd et al., TCP Friendly Rate Control (TFRC): Protocol Specification. 2008. https://doi.org/10.17487/rfc5348

[40] Nichols and Jacobson, Controlled Delay Active Queue Management. 2018. https://doi.org/10.17487/rfc8289

[41] Mathis et al., The macroscopic behavior of the TCP congestion avoidance algorithm. ACM SIGCOMM, 1997. https://doi.org/10.1145/263932.264023

[42] Katabi et al., Congestion control for high bandwidth-delay product networks. ACM SIGCOMM, 2002. https://doi.org/10.1145/964725.633035

[43] Hu et al., Locating internet bottlenecks: Algorithms, measurements, and implications. ACM SIGCOMM, 2004. https://doi.org/10.1145/1030194.1015474

[44] Alizadeh et al., Data center TCP (DCTCP). ACM SIGCOMM, 2010. https://doi.org/10.1145/1851182.1851192

[45] Ashok et al., Data-Driven Network Path Simulation with iBox. ACM SIGMETRICS Performance Evaluation Review, 2022. https://doi.org/10.1145/3547353.3522646

[46] Dethise et al., Cracking Open the Black Box: What Observations Can Tell Us About Reinforcement Learning Agents. 2019 Workshop on Network Meets AI & ML - NetAI'19, 2019. https://doi.org/10.1145/3341216.3342210 (abstract level)

[47] Eliyahu et al., Verifying learning-augmented systems. ACM SIGCOMM, 2021. https://doi.org/10.1145/3452296.3472936 (abstract level)

[48] Low and Lapsely, Optimization flow control. I. Basic algorithm and convergence. IEEE/ACM Transactions on Networking, 1999. https://doi.org/10.1109/90.811451

[49] Zhang et al., Observations on the dynamics of a congestion control algorithm: The effects of two-way traffic. ACM SIGCOMM, 1991. https://doi.org/10.1145/115992.116006

[50] Rubenstein et al., Detecting shared congestion of flows via end-to-end measurement. IEEE/ACM Transactions on Networking, 2002. https://doi.org/10.1109/tnet.2002.1012369

[51] Cen et al., End-to-end differentiation of congestion and wireless losses. IEEE/ACM Transactions on Networking, 2003. https://doi.org/10.1109/tnet.2003.818187

[52] Coates et al., Internet tomography. IEEE Signal Processing Magazine, 2002. https://doi.org/10.1109/79.998081

[53] Little, A Proof for the Queuing Formula: L = lambda W. Operations Research, 1961. https://doi.org/10.1287/opre.9.3.383

[54] Jaffe, Bottleneck Flow Control. IEEE Transactions on Communications, 1981. https://doi.org/10.1109/tcom.1981.1095081 (abstract level)

[55] Zaki et al., Adaptive Congestion Control for Unpredictable Cellular Networks. ACM SIGCOMM, 2015. https://doi.org/10.1145/2829988.2787498

[56] Xie et al., PBE-CC: Congestion Control via Endpoint-Centric, Physical-Layer Bandwidth Measurements. ACM SIGCOMM, 2020. https://doi.org/10.1145/3387514.3405880

[57] Liao et al., Astraea: Towards Fair and Efficient Learning-based Congestion Control. Nineteenth European Conference on Computer Systems, 2024. https://doi.org/10.1145/3627703.3650069

[58] Meng et al., Interpreting Deep Learning-Based Networking Systems. ACM SIGCOMM, 2020. https://doi.org/10.1145/3387514.3405859 (abstract level)

[59] Ramakrishnan et al., The Addition of Explicit Congestion Notification (ECN) to IP. 2001. https://doi.org/10.17487/rfc3168

[60] Addanki et al., PowerTCP: Pushing the Performance Limits of Datacenter Networks. 2021. https://arxiv.org/abs/2112.14309

[61] Yang et al., Canopy: Property-Driven Learning for Congestion Control. 2024. https://arxiv.org/abs/2412.10915 (abstract level)

[62] Martins et al., Closed-form congestion control via deep symbolic regression. 2024. https://arxiv.org/abs/2405.01435 (abstract level)

[63] Jain, Congestion Control and Traffic Management in ATM Networks: Recent Advances and A Survey. Computer Networks and ISDN Systems, vol 28, no 13, February 1995, pp1723-1738, 1998. https://arxiv.org/abs/cs/9809085

[64] Patel et al., CrystalBox: Future-Based Explanations for Input-Driven Deep RL Systems. 2023. https://arxiv.org/abs/2302.13483 (abstract level)

[65] Scherrer et al., A Control-Theoretic Perspective on BBR/CUBIC Congestion-Control Competition. 2025. https://arxiv.org/abs/2510.22773

[66] Jain and Ramakrishnan, Congestion Avoidance in Computer Networks with a Connectionless Network Layer, Part I: Concepts, Goals and Methodology. 1998. https://arxiv.org/abs/cs/9809095

[67] Ros-Giralt et al., A Quantitative Theory of Bottleneck Structures for Data Networks. 2022. https://arxiv.org/abs/2210.03534

[68] Jay et al., Internet Congestion Control via Deep Reinforcement Learning. 2018. https://arxiv.org/abs/1810.03259

[69] Liebeherr et al., A System Theoretic Approach to Bandwidth Estimation. 2008. https://arxiv.org/abs/0801.0455

[70] Lübben et al., A Foundation for Stochastic Bandwidth Estimation of Networks with Random Service. 2010. https://arxiv.org/abs/1008.0050

[71] Dey et al., Estimating Network Link Characteristics using Packet-Pair Dispersion: A Discrete Time Queueing Theoretic View. 2009. https://arxiv.org/abs/0911.3528

[72] Ramakrishnan and Jain, A binary feedback scheme for congestion avoidance in computer networks. ACM Transactions on Computer Systems, 1990. https://doi.org/10.1145/78952.78955 (abstract level)

[73] Feng et al., The BLUE active queue management algorithms. IEEE/ACM Transactions on Networking, 2002. https://doi.org/10.1109/tnet.2002.801399 (abstract level)

[74] Kunniyur and Srikant, Analysis and design of an adaptive virtual queue (AVQ) algorithm for active queue management. ACM SIGCOMM, 2001. https://doi.org/10.1145/964723.383069

[75] Jain, A delay-based approach for congestion avoidance in interconnected heterogeneous computer networks. ACM SIGCOMM, 1989. https://doi.org/10.1145/74681.74686

[76] Jin et al., FAST TCP: motivation, architecture, algorithms, performance. IEEE INFOCOM 2004, 2006. https://doi.org/10.1109/infcom.2004.1354670 (abstract level)

[77] Mascolo et al., TCP westwood: Bandwidth estimation for enhanced transport over wireless links. annual international conference on Mobile computing and networking, 2001. https://doi.org/10.1145/381677.381704 (abstract level)

[78] Jain et al., A Quantitative Measure Of Fairness And Discrimination For Resource Allocation In Shared Computer Systems. 1998. https://arxiv.org/abs/cs/9809099 (abstract level)
