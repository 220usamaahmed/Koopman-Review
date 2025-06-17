# Koopman Operators and Deep Learning

## DESKO: STABILITY-ASSURED ROBUST CONTROL OF NONLINEAR SYSTEMS WITH A DEEP STOCHASTIC KOOPMAN OPERATOR

### Experiments

#### Environments
- CartPole
- HalfCheetah
- SoPrA (Soft continuum robotic arm)
- Gene Regulatory Networks (GRN) (Biological three-gene regulatory network)
- Note: They also tested versions with process noise and observation noise

#### Baselines
1. Deep Koopman Operator (DKO)
2. Ensemble of Multilayer Perceptrons (MLP) (10 MLPs trained to minimize cumulative prediction error)
3. Soft Actor-Critic (SAC)

### MPC Formulation
- linear MPC
- standard quadratic program

### Advantages and Main Novelty
The core innovation is making Koopman operators stochastic. Instead of deterministic observable functions, they learn a probabilistic distribution over observables using neural networks that output mean and variance parameters.

- resist disturbances up to 5× the maximum control input magnitude

- theoretical stability proofs for both exact and approximated Koopman operators

- the method scales to high-dimensional systems better than alternatives like Gaussian processes

- outperforms deterministic Koopman methods on systems with process and observation noise

- the entropy constraint they introduce prevents overfitting and maintains appropriate uncertainty levels

---