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

## DECISION S4: EFFICIENT SEQUENCE-BASED RL VIA STATE SPACE LAYERS

### Experiments

#### Environments

#### Baselines

### MPC Formulation

### Advantages and Main Novelty

---

### Experiments

#### Environments
- HalfCheetah, Hopper, Walker2D
- AntMaze-umaze and AntMaze-diverse tasks
- Expert, medium, and medium-replay datasets from D4RL

#### Baselines
- Decision Transformer and its variants: CDT, ODT
- Traditional RL methods: IQL, CQL, BEAR, BRAC
- Trajectory Transformer variants: T_Tu, T_Tq (sample-based methods using beam search)
- RNN-based models (for ablation studies)

### MPC Formulation
Does not use MPC. Instead, it uses a sequence-based approach similar to Decision Transformers.

### Advantages and Main Novelty
- pplication of S4 layers to reinforcement learning**, replacing transformers in sequence-based RL
- Hybrid training approach: off-policy pre-training followed by on-policy fine-tuning
- Novel stable actor-critic mechanism adapted for S4 layers
- fewer parameters than Decision Transformer
- Linear complexity in sequence length vs quadratic for transformers