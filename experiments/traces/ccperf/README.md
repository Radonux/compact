# ccperf's rate traces

Copied unchanged from ccperf (gitlab.com/ccperf/ccperf at `18d1620`,
`resources/traces`, MIT), which takes them from Pantheon's observatory
(github.com/StanfordSNR/observatory). `experiments/ccperf.py` replays them
in its `replay_rate_trace` experiment. Each line is a Mahimahi delivery
opportunity in ms; each file runs about 60 s and begins with 1 to 5 s
without a delivery, as ccperf replays it.
