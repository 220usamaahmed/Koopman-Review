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
- pplication of S4 layers to reinforcement learning, replacing transformers in sequence-based RL
- Hybrid training approach: off-policy pre-training followed by on-policy fine-tuning
- Novel stable actor-critic mechanism adapted for S4 layers
- fewer parameters than Decision Transformer
- Linear complexity in sequence length vs quadratic for transformers

---

## EFFICIENT DYNAMICS MODELING IN INTERACTIVE ENVIRONMENTS WITH KOOPMAN THEORY

### Experiments

#### Environments
- D4RL offline datasets: hopper, walker2d, and halfcheetah with three quality levels each (expert, medium-expert, full-replay)
- DeepMind Control Suite: Ball in Cup Catch, Cartpole Swingup, Cheetah Run, Finger Spin, Walker Walk
- Quadruped Run, Quadruped Walk, Cheetah Run, Acrobot Swingup

#### Baselines
- MLP-based dynamics model (standard non-linear latent dynamics)
- GRU-based dynamics model (recurrent approach)
- Transformer-based dynamics model (causal transformer for parallel training)
- Diagonal State Space Models (DSSM)
- TD-MPC
- SAC with SPR (Self-Predictive Representations for model-free RL)

### MPC Formulation
- TD-MPC framework where they replace the original MLP-based "Task-Oriented Latent Dynamics" (TOLD) model with their diagonal Koopman operator

### Advantages and Main Novelty
- linearizing non-linear controlled dynamics in a learned latent space

---

## MAMKO: MAMBA-BASED KOOPMAN OPERATOR FOR MODELING AND PREDICTIVE CONTROL

### Experiments

#### Environments
- CartPole system
- Gene Regulatory Networks (GRN) (synthetic three-gene regulatory network with oscillatory behavior)
- Reactor-Separator Chemical Process (RSCP) - two continuous stirred tank reactors with flash tank separator (time-invariant and time-varying versions)
- Wastewater treatment system - large-scale system with 145 states, 2 control inputs, 14 disturbances
- Time-delay RSCP system - RSCP with 0.025h time delay

#### Baselines
For modeling:
- **Deep Koopman Operator (DKO)** - Koopman method with deep neural networks but invariant operators
- **Multilayer Perceptrons (MLP)** - standard fully connected neural networks

For control:
- DKO-based MPC
- MLP-based MPC uses neural networks as predictive models with nonlinear optimization (CasADi + IPOPT)
- Soft Actor-Critic (SAC)
- PID controller
- NMPC

### MPC Formulation
numerical/convex optimization-based

### Advantages and Main Novelty
Integration of Mamba architecture with Koopman operator theory