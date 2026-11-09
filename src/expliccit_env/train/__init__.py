"""Training net1 (path beliefs) and net2 (the controller).

    spec.py        what to train (TrainSpec, PPOSpec), parsed once
    data.py        steps.parquet shards -> arrays, split by path (or link)
    models.py      the networks by name: net1's encoders, the offline actor
    beliefs.py     the kinds of belief net1 states (value, interval,
                   distribution) and every way the rest reads them
    graph.py       the deployed policy: front end, window target, decision
    fit.py         net1 supervised; net2 offline on a frozen net1 (cloning,
                   TD3+BC, IQL or expliCCit's objective); expliCCit's models
    recurrent.py   a net2 with memory that decides once per block of steps
    actors.py      what that net2 computes: front end, target, head, graph
    mixture.py     the training links' traffic (stock, copies, info, rise)
    trajectories.py  rollouts' steps -> decisions, rewards and labels
    handoff.py     continuing or teaching from an earlier run
    learners.py    PPO, TD3+BC and DAgger on those decisions
    ppo.py         the closed-loop run that ties them together
    rollouts.py    the closed-loop episodes a learner runs in UML
    share.py       the share rule and the cooperative window on net1's means
    traffic.py     a head on net1's input that states the other traffic
    attribution.py exact Shapley values of a decision over its LCFs
    score.py       log utility of closed-loop runs, from ground truth
    export.py      one ONNX graph per policy, under the sage-v1 contract
    run.py         training runs and imports, as the CLI calls them

Needs the ``train`` extra (PyTorch), except data.py, spec.py and score.py.
Architectures and algorithms are chosen by name in the specs.
"""
