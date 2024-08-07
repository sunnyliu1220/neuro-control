# neuro-control
This project aims to apply control theory to perturbation experiments in neuroscience. Here is a rough roadmap:

- First, we start with simple dynamical systems (e.g. linear) and explore the effects of partial observation, observation time, network size, etc. on the system identification performance. We try to reproduce some of the results in [Qian et al., 2024](https://doi.org/10.1101/2024.05.24.595741). The focus will be to see whether the hallucinated dynamical structures (e.g. spurious attractors and limit cycles) are actually important errors.

- Second, we fit simple dynamical systems to Omar's experimental data in Mark's lab. We will explore the same things as in the first step and look at how much worse it does on real data. This will give us a sense of the complexity of real data compared to simulations.

- Third, we will try to apply optimal control to a misidentified system. We ask how badly the errors from sysid propagates to control. If in some scenarios the errors don't become pathological, we can potentially design perturbation experiments even from limited recordings. We aim for analytical solutions to these simple models, but numerical solutions for more complex models are also desirable if that complexity is warranted.