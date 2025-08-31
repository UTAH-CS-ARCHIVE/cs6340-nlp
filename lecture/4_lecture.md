# [Lecture Notes] Hidden Markov Models (HMM)

## 1) Pre-requisites
- Probability (conditioning, Bayes, total probability), linear algebra, multivariate calculus.  
- Information theory: log-likelihood, KL divergence, Jensen’s inequality.  
- Dynamic programming and semiring viewpoint (sum–product vs max–sum).

<br>

## 2) Formalism of HMMs

### 2.1 Components and Constraints
- Hidden states $\mathcal{S}=\{1,\dots,N\}$, observations $\mathbf{o}=(o_1,\dots,o_T)$.  
- Parameters $\theta=(\boldsymbol{\pi},A,B)$, with
  - initial distribution $\pi_i=P(s_1=i)$, 

    $$

    \sum_{i=1}^{N}\pi_i=1,\quad \pi_i\ge 0.

    $$

  - transition probabilities $a_{ij}=P(s_{t+1}=j\mid s_t=i)$,

    $$

    \sum_{j=1}^{N} a_{ij}=1\ \forall i,\quad a_{ij}\ge 0.

    $$

  - emission distributions $b_i(o)=P(o_t=o\mid s_t=i)$:
    - **Discrete:** $b_i(k)\ge 0$ with 

      $$

      \sum_{k=1}^{V} b_i(k)=1.

      $$

    - **Continuous:** e.g., Gaussian $b_i(o)=\mathcal{N}(o\mid \mu_i,\Sigma_i)$, or a GMM per state.

### 2.2 Factorization
For a path $\mathbf{s}=(s_1,\dots,s_T)$ and observations $\mathbf{o}$,

$$

P(\mathbf{s},\mathbf{o}\mid\theta)
=
\pi_{s_1}\, b_{s_1}(o_1)\,\prod_{t=2}^{T} a_{s_{t-1},s_t}\, b_{s_t}(o_t).

$$

The likelihood is the marginal

$$

P(\mathbf{o}\mid \theta)=\sum_{\mathbf{s}} P(\mathbf{s},\mathbf{o}\mid\theta),

$$

which is intractable by brute force but solvable in $O(TN^2)$ by dynamic programming.

<br>

## 3) Forward–Backward Inference (sum–product)

### 3.1 Forward recursion (derivation)
Define the forward message

$$

\alpha_t(i)=P(o_{1:t},\,s_t=i\mid\theta).

$$

Initial condition:

$$

\alpha_1(i)=\pi_i\, b_i(o_1).

$$

Inductive step: using the law of total probability and conditional independences of HMMs,

$$

\alpha_t(j)
=
\sum_{i=1}^{N} P(o_{1:t},s_{t-1}=i,s_t=j\mid\theta)
=
\sum_{i=1}^{N} \alpha_{t-1}(i)\, a_{ij}\, b_j(o_t).

$$

Hence the likelihood is

$$

P(\mathbf{o}\mid\theta)=\sum_{i=1}^{N}\alpha_T(i).

$$

### 3.2 Backward recursion (derivation)
Define the backward message

$$

\beta_t(i)=P(o_{t+1:T}\mid s_t=i,\theta).

$$

Terminal condition:

$$

\beta_T(i)=1.

$$

Recursion (conditioning on $s_{t+1}$ and $o_{t+1}$):

$$

\beta_t(i)=\sum_{j=1}^{N} a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j).

$$

### 3.3 Likelihood and posteriors; consistency identities
A key identity (provable by induction) is

$$

P(\mathbf{o}\mid\theta)=\sum_{i=1}^{N}\alpha_t(i)\,\beta_t(i)\quad \text{for any } t.

$$

From Bayes,

$$

\gamma_t(i)
=
P(s_t=i\mid\mathbf{o},\theta)
=
\frac{\alpha_t(i)\,\beta_t(i)}{P(\mathbf{o}\mid\theta)}.

$$

For adjacent states,

$$

\xi_t(i,j)
=
P(s_t=i,s_{t+1}=j\mid\mathbf{o},\theta)
=
\frac{\alpha_t(i)\,a_{ij}\,b_j(o_{t+1})\,\beta_{t+1}(j)}{P(\mathbf{o}\mid\theta)}.

$$

Normalization checks:

$$

\sum_{i=1}^{N}\gamma_t(i)=1,\qquad
\sum_{i=1}^{N}\sum_{j=1}^{N}\xi_t(i,j)=1.

$$

### 3.4 Numerical stabilization: scaling and log-domain
To avoid underflow, introduce per-time scaling constants $c_t$.

- **Forward scaling:** compute unscaled $\alpha_t'$ then set

  $$

  c_t=\sum_{i}\alpha_t'(i),\qquad
  \alpha_t(i)=\frac{\alpha_t'(i)}{c_t}.

  $$

  Then

  $$

  \log P(\mathbf{o}\mid\theta)=-\sum_{t=1}^{T}\log c_t.

  $$

- **Backward scaling:** to maintain $\sum_i \alpha_t(i)\beta_t(i)=1$ at each $t$, set

  $$

  \beta_T(i)=\frac{1}{c_T},\qquad
  \beta_t(i)=\frac{\sum_j a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j)}{c_t}.

  $$

- **Log-domain alternative:** replace sums by $\operatorname{LSE}$ (log-sum-exp); products become sums of logs. Both approaches are equivalent up to round-off.

<br>

## 4) Viterbi Decoding (max–product)

### 4.1 Optimal substructure and recurrence (proof)
We wish to find

$$

\hat{\mathbf{s}}
=
\arg\max_{\mathbf{s}} P(\mathbf{s}\mid \mathbf{o},\theta)
=
\arg\max_{\mathbf{s}} P(\mathbf{s},\mathbf{o}\mid\theta).

$$

Define the optimal-score function (log-domain for stability)

$$

\delta_t(j)
=
\max_{s_{1:t-1}}
\log P(s_{1:t-1}, s_t=j, o_{1:t}\mid \theta).

$$

By the Markov property and factorization,

$$

\delta_t(j)
=
\Big[\max_{i}\ \delta_{t-1}(i) + \log a_{ij}\Big]
+ \log b_j(o_t),

$$

with initialization $\delta_1(i)=\log\pi_i + \log b_i(o_1)$.  
**Proof sketch (optimal substructure):** any optimal path ending in $j$ at time $t$ must extend an optimal path ending at some $i$ at $t-1$; otherwise we could improve it, contradicting optimality.

### 4.2 Path reconstruction and differences vs posterior decoding
Backpointers record the maximizing predecessor at each step; a single backward sweep reconstructs $\hat{\mathbf{s}}$.  
**Contrast:** Viterbi gives the **globally** best path; posterior decoding $\arg\max_i \gamma_t(i)$ is **locally** optimal per time step and may yield an impossible path under $A$.

<br>

## 5) Baum–Welch (EM) Learning

### 5.1 Complete-data log-likelihood and the Q-function
For one sequence $(\mathbf{s},\mathbf{o})$,

$$

\log p(\mathbf{s},\mathbf{o}\mid\theta)
=
\log \pi_{s_1}
+ \sum_{t=2}^{T} \log a_{s_{t-1},s_t}
+ \sum_{t=1}^{T} \log b_{s_t}(o_t).

$$

EM maximizes the observed likelihood via

$$

Q(\theta,\theta^{\text{old}})
=
\mathbb{E}_{p(\mathbf{s}\mid \mathbf{o},\theta^{\text{old}})}
\big[\log p(\mathbf{s},\mathbf{o}\mid\theta)\big],

$$

and iterates $\theta^{\text{new}}=\arg\max_{\theta}Q(\theta,\theta^{\text{old}})$.

For multiple sequences, expectations sum over sequences.

### 5.2 E-step: expectations via Forward–Backward
Compute $\gamma_t(i)$ and $\xi_t(i,j)$ for each sequence under $\theta^{\text{old}}$. These are the expected **state** and **transition** counts (posterior responsibilities) that feed the M-step.

### 5.3 M-step (discrete emissions): Lagrange-multiplier derivations

**Initial distribution.** Maximize

$$

\sum_{i}\Big(\sum_{n}\gamma^{(n)}_1(i)\Big)\log \pi_i
\quad \text{s.t.}\quad \sum_i \pi_i=1.

$$

Lagrangian $\mathcal{L}=\sum_i C_i\log\pi_i+\lambda(1-\sum_i\pi_i)$ with $C_i=\sum_n\gamma^{(n)}_1(i)$.  
Setting $\partial \mathcal{L}/\partial \pi_i=0$ yields

$$

\pi_i
=
\frac{C_i}{\sum_{i'} C_{i'}}.
$$

**Transitions.** Row-wise maximization for each $i$:

$$

\max_{(a_{ij})_j}\ \sum_{j} \Big(\sum_{n,t}\xi_t^{(n)}(i,j)\Big)\log a_{ij}
\quad \text{s.t.}\quad \sum_j a_{ij}=1.

$$

With $N_{ij}=\sum_{n,t}\xi_t^{(n)}(i,j)$ and $N_i=\sum_j N_{ij}=\sum_{n,t}\gamma_t^{(n)}(i)$,

$$

a_{ij}=\frac{N_{ij}}{N_i}.
$$

**Emissions (discrete).** For each state $i$,

$$

b_i(k)
=
\frac{\sum_{n,t:\,o_t^{(n)}=k} \gamma_t^{(n)}(i)}
     {\sum_{n,t} \gamma_t^{(n)}(i)}.

$$

All updates arise from maximizing linear forms in $\log$-parameters subject to simplex constraints.

### 5.4 M-step (Gaussian / GMM emissions): weighted MLE derivations

**Single Gaussian per state.** With $b_i(o)=\mathcal{N}(o\mid \mu_i,\Sigma_i)$, maximize

$$

\sum_{n,t} \gamma_t^{(n)}(i)\,\log \mathcal{N}(o_t^{(n)}\mid \mu_i,\Sigma_i).

$$

Setting derivatives to zero gives responsibility-weighted MLE:

$$

\mu_i
=
\frac{\sum_{n,t}\gamma_t^{(n)}(i)\,o_t^{(n)}}
     {\sum_{n,t}\gamma_t^{(n)}(i)},
\qquad
\Sigma_i
=
\frac{\sum_{n,t}\gamma_t^{(n)}(i)\,(o_t^{(n)}-\mu_i)(o_t^{(n)}-\mu_i)^\top}
     {\sum_{n,t}\gamma_t^{(n)}(i)}.

$$

**GMM per state.** If $b_i(o)=\sum_{m=1}^{M} w_{im}\,\mathcal{N}(o\mid \mu_{im},\Sigma_{im})$, introduce component responsibilities $\gamma_{t,i,m}$ (posterior of component $m$ within state $i$). Then

$$

w_{im}=\frac{\sum_{n,t}\gamma_{t,i,m}^{(n)}}{\sum_{n,t}\sum_{m'}\gamma_{t,i,m'}^{(n)}},\quad
\mu_{im}
=
\frac{\sum_{n,t}\gamma_{t,i,m}^{(n)}\,o_t^{(n)}}
     {\sum_{n,t}\gamma_{t,i,m}^{(n)}},

$$

$$

\Sigma_{im}
=
\frac{\sum_{n,t}\gamma_{t,i,m}^{(n)}\,(o_t^{(n)}-\mu_{im})(o_t^{(n)}-\mu_{im})^\top}
     {\sum_{n,t}\gamma_{t,i,m}^{(n)}}.

$$

### 5.5 MAP updates with Dirichlet priors (smoothing)
Place Dirichlet priors on each categorical row: $A_{i\cdot}\sim\mathrm{Dir}(\alpha_{i\cdot})$, $B_{i\cdot}\sim\mathrm{Dir}(\beta_{i\cdot})$. The posterior mode (MAP) yields

$$

a_{ij}^{\text{MAP}}
=
\frac{N_{ij}+\alpha_{ij}-1}{\sum_{j'} (N_{ij'}+\alpha_{ij'}-1)},
\qquad
b_i(k)^{\text{MAP}}
=
\frac{N_{ik}+\beta_{ik}-1}{\sum_{k'} (N_{ik'}+\beta_{ik'}-1)}.

$$

This generalizes add-$\kappa$ smoothing and prevents zero probabilities.

### 5.6 Convergence (Jensen) and identifiability
EM guarantees non-decreasing observed log-likelihood:

$$

\log p(\mathbf{o}\mid \theta^{\text{new}})
\;\ge\;
\log p(\mathbf{o}\mid \theta^{\text{old}}),

$$

a consequence of Jensen’s inequality applied to the EM lower bound. Convergence is to a **local** optimum.  
HMM parameters are identifiable only up to **label permutation** of hidden states (label switching); weak priors or partial supervision can anchor semantics.

<br>

## 6) Model Properties and Diagnostics

### 6.1 Complexity & memory; streaming variants
- **Forward–Backward / Viterbi:** time $O(TN^2)$, memory $O(TN)$ (can be reduced to $O(N)$ for streaming likelihood/posteriors or by checkpointing for Viterbi).  
- **GMM emissions:** per-time evaluation cost scales with mixture count and feature dimension.

### 6.2 Sanity checks & invariants
- Row sums of $A$ and each emission distribution equal $1$ after every update.  
- Posteriors normalize:

  $$

  \sum_i \gamma_t(i)=1,\qquad \sum_{i,j}\xi_t(i,j)=1.

  $$

- Scaled Forward–Backward satisfies

  $$

  \log P(\mathbf{o}\mid\theta)=-\sum_{t}\log c_t.

  $$

- EM **monotonicity:** observed log-likelihood is non-decreasing across iterations (to numerical tolerance).

### 6.3 Evaluation criteria (theoretical perspective)
- **POS tagging:** sequence accuracy, per-tag F1; posterior vs Viterbi decoding may trade precision/recall.  
- **ASR:** WER derives from minimum edit distance; decoding hyperparameters (beam, LM scale) effectively regularize the search objective.  
- **Model selection:** AIC/BIC penalize parameter count; for HMMs, both $N$ (states) and emission complexity (e.g., mixture count) contribute.M).  
- MAP (Dirichlet) updates correctly incorporate hyperparameters and avoid zeros.  
- All distributions (rows of $A$, emissions, $\boldsymbol{\pi}$) are nonnegative and sum to $1$.