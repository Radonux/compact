"""Training net1 (path beliefs) and net2 (the controller).

    spec.py       what to train from step data (TrainSpec), parsed once
    data.py       steps.parquet shards -> arrays, split by path, never by row
    models.py     architectures by name, and the stages of a policy graph
    fit.py        net1 supervised; net2 offline on a frozen net1 (cloning,
                  TD3+BC, IQL or expliCCit's objective); expliCCit's models
    online.py     net2 in closed loop, off-policy from a replay buffer
    recurrent.py  a net2 with memory that decides once per block of steps
    ppo.py        that net2 in closed loop: PPO, TD3+BC or DAgger
    share.py      the share rule and the cooperative window on net1's means
    score.py      log utility of closed-loop runs, from ground truth
    export.py     one ONNX graph per policy, under the sage-v1 contract
    run.py        training runs and imports, as the CLI calls them

Needs the ``train`` extra (PyTorch), except data.py, spec.py and score.py.
Architectures and algorithms are chosen by name in the specs.
"""
