# [Assignment] Hidden Markov Models


## Part I. Foundations

### (1) Define the three parameter sets of an HMM ($\pi$, $A$, $B$). Roles and constraints
- **Initial distribution** $\boldsymbol{\pi}$: $\pi_i = P(s_1=i)$, the prior over the first hidden state.  
  **Constraints:** $\sum_{i=1}^{N}\pi_i=1$, $\pi_i\ge 0$.
- **Transition matrix** $A$: $a_{ij}=P(s_{t+1}=j\mid s_t=i)$, the Markov dynamics over hidden states.  
  **Constraints:** $\sum_{j=1}^{N} a_{ij}=1$ $\forall i$, $a_{ij}\ge 0$.
- **Emission distributions** $B$: $b_i(o)=P(o_t=o\mid s_t=i)$, the observation likelihood given state $i$.  
  **Discrete case:** $b_i(k)\ge 0$, $\sum_{k=1}^{V} b_i(k)=1$.  
  **Continuous case:** e.g., Gaussian $b_i(o)=\mathcal{N}(o\mid \mu_i,\Sigma_i)$ (valid pdf).

### (2) Derive $P(\mathbf{s},\mathbf{o}\mid\theta)$ and explain intractability of $P(\mathbf{o}\mid\theta)$
Let $\mathbf{s}=(s_1,\dots,s_T)$, $\mathbf{o}=(o_1,\dots,o_T)$. By the chain rule and HMM independences,

$$
P(\mathbf{s},\mathbf{o}\mid\theta)
=\pi_{s_1}\, b_{s_1}(o_1)\,\prod_{t=2}^{T} a_{s_{t-1},s_t}\, b_{s_t}(o_t).
$$

The likelihood marginalizes the hidden path:

$$
P(\mathbf{o}\mid\theta)=\sum_{\mathbf{s}} P(\mathbf{s},\mathbf{o}\mid\theta),
$$

which is a sum over $N^T$ paths — **exponential** in $T$ — hence brute-force is intractable; we use dynamic programming (Forward–Backward) in $O(TN^2)$.

<br>

## Part II. Forward–Backward Inference

### (1) Forward recursion from the law of total probability
Define

$$
\alpha_t(i)=P(o_{1:t},\,s_t=i\mid\theta).
$$

**Base:** $\alpha_1(i)=\pi_i\, b_i(o_1)$.  
**Induction:** For $t\ge 2$,

$$\begin{aligned}
\alpha_t(j)
&= P(o_{1:t}, s_t=j) 
= \sum_{i=1}^N P(o_{1:t}, s_{t-1}=i, s_t=j)\\
&= \sum_{i=1}^N P(o_{1:t-1}, s_{t-1}=i)\, P(s_t=j\mid s_{t-1}=i)\, P(o_t\mid s_t=j)\\
&= \sum_{i=1}^N \alpha_{t-1}(i)\, a_{ij}\, b_j(o_t).
\end{aligned}$$


### (2) Backward recursion and relation to posteriors
Define

$$
\beta_t(i)=P(o_{t+1:T}\mid s_t=i,\theta).
$$

**Terminal:** $\beta_T(i)=1$.  
**Recurrence:** For $t\le T-1$,

$$
\beta_t(i)=\sum_{j=1}^{N} a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j).
$$

**Posterior:** By Bayes and conditional independence,

$$
P(s_t=i\mid\mathbf{o},\theta)=\frac{P(o_{1:t},s_t=i)\,P(o_{t+1:T}\mid s_t=i)}{P(\mathbf{o})}
= \frac{\alpha_t(i)\,\beta_t(i)}{P(\mathbf{o}\mid\theta)}.
$$

### (3) Prove $\sum_i \alpha_t(i)\beta_t(i)=P(\mathbf{o}\mid\theta)$ (any $t$)
$$
\begin{aligned}
\sum_i \alpha_t(i)\beta_t(i)
&= \sum_i P(o_{1:t}, s_t=i)\, P(o_{t+1:T}\mid s_t=i)\\
&= \sum_i P(o_{1:t}, o_{t+1:T}, s_t=i) \quad (\text{chain rule})\\
&= P(o_{1:T}) = P(\mathbf{o}\mid\theta).
\end{aligned}
$$

Thus the identity holds for all $t$.

### (4) Why scaling/log-domain are necessary; difference between them
**Problem:** Products of many probabilities underflow in finite precision.  
- **Scaling:** At each $t$, compute unscaled $\alpha_t'$ and set

  $
  c_t=\sum_i \alpha_t'(i),\ \alpha_t(i)=\alpha_t'(i)/c_t.
  $

  Then $\log P(\mathbf{o}\mid\theta)=-\sum_{t=1}^{T}\log c_t$. Backward messages use matching normalization (e.g., $\beta_T(i)=1/c_T$, $\beta_t=\frac{A\,(b(\cdot)\odot \beta_{t+1})}{c_t}$) to preserve $\sum_i \alpha_t(i)\beta_t(i)=1$ in scaled space.
- **Log-domain:** Work with $\log \alpha,\log \beta$; replace sums by $\operatorname{LSE}$ (log-sum-exp). This avoids choosing $c_t$ but requires careful numerical $\operatorname{LSE}$.  
**Difference:** Scaling keeps variables in probability space with explicit per-time normalizers; log-domain converts products to sums, often more stable but a bit costlier due to $\operatorname{LSE}$.

<br>

## Part III. Viterbi Decoding

### (1) Recurrence for $\delta_t(j)$ via optimal substructure
Let

$$
\delta_t(j)=\max_{s_{1:t-1}} \log P(s_{1:t-1}, s_t=j, o_{1:t}\mid\theta).
$$

Using the HMM factorization,

$$
\delta_t(j)=\Big[\max_{i} \ \delta_{t-1}(i) + \log a_{ij}\Big] + \log b_j(o_t),
$$

with $\delta_1(i)=\log \pi_i + \log b_i(o_1)$.

### (2) Backpointers for path reconstruction
Record

$$
\psi_t(j)=\arg\max_{i}\ \delta_{t-1}(i)+\log a_{ij}.
$$

After the forward pass, pick $\hat{s}_T=\arg\max_j \delta_T(j)$ and trace backward: $\hat{s}_{t-1}=\psi_t(\hat{s}_t)$ to recover $\hat{\mathbf{s}}$.

### (3) Viterbi vs posterior decoding; when they disagree
- **Viterbi:** one **globally** most probable path $\arg\max_{\mathbf{s}} P(\mathbf{s}\mid\mathbf{o})$.
- **Posterior decoding:** per-time **marginals** $\hat{s}_t=\arg\max_i P(s_t=i\mid\mathbf{o})$ (may violate transitions).

**Example:** Suppose $A$ forbids $1\to 1$ but posteriors at $t=1,2$ both favor state 1 marginally. Posterior decoding yields $(1,1)$, an **invalid** path; Viterbi returns a valid sequence like $(1,2)$ that respects $A$.

<br>

## Part IV. Parameter Estimation with EM (Baum–Welch)

### (1) E-step responsibilities from Forward–Backward; expected counts

$$
\gamma_t(i)=P(s_t=i\mid\mathbf{o},\theta)=\frac{\alpha_t(i)\beta_t(i)}{P(\mathbf{o}\mid\theta)},\quad
\xi_t(i,j)=P(s_t=i,s_{t+1}=j\mid\mathbf{o},\theta)
=\frac{\alpha_t(i)\,a_{ij}\,b_j(o_{t+1})\,\beta_{t+1}(j)}{P(\mathbf{o}\mid\theta)}.
$$

Interpretation: $\sum_t \gamma_t(i)$ is the expected **count** of visits to state $i$; $\sum_{t} \xi_t(i,j)$ is the expected **count** of transitions $i\!\to\! j$.

(For multiple sequences, sum $\gamma,\xi$ over sequences.)

### (2) M-step updates via Lagrange multipliers (discrete emissions)
Let $N_{ij}=\sum_{t=1}^{T-1}\xi_t(i,j)$, $N_i=\sum_{t=1}^{T}\gamma_t(i)$, and $N_{ik}=\sum_{t: o_t=k}\gamma_t(i)$.

- **Initial distribution $\pi$**  
  Maximize $\sum_i \gamma_1(i)\log \pi_i$ s.t. $\sum_i \pi_i=1$.
  With $\mathcal{L}=\sum_i \gamma_1(i)\log\pi_i + \lambda(1-\sum_i \pi_i)$,

  $$
  \frac{\partial\mathcal{L}}{\partial \pi_i}=\frac{\gamma_1(i)}{\pi_i}-\lambda=0
  \Rightarrow \pi_i=\frac{\gamma_1(i)}{\sum_{i'}\gamma_1(i')}.
  $$

  (For multiple sequences, replace $\gamma_1$ by $\sum_n \gamma_1^{(n)}$.)

- **Transitions $a_{ij}$**  
  Maximize $\sum_j N_{ij}\log a_{ij}$ s.t. $\sum_j a_{ij}=1$ (row-wise).  
  Lagrangian $\mathcal{L}_i=\sum_j N_{ij}\log a_{ij}+\lambda_i(1-\sum_j a_{ij})$ gives

  $$
  a_{ij}=\frac{N_{ij}}{\sum_{j'}N_{ij'}}=\frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}.
  $$

- **Emissions $b_i(k)$ (discrete)**  
  Maximize $\sum_k N_{ik}\log b_i(k)$ s.t. $\sum_k b_i(k)=1$.  
  Lagrangian yields

  $$
  b_i(k)=\frac{N_{ik}}{\sum_{k'} N_{ik'}}=\frac{\sum_{t: o_t=k} \gamma_t(i)}{\sum_t \gamma_t(i)}.
  $$

### (3) Gaussian emissions: responsibility-weighted MLE
If $b_i(o)=\mathcal{N}(o\mid \mu_i,\Sigma_i)$,

$$
\mu_i=\frac{\sum_t \gamma_t(i)\, o_t}{\sum_t \gamma_t(i)},\qquad
\Sigma_i=\frac{\sum_t \gamma_t(i)\,(o_t-\mu_i)(o_t-\mu_i)^\top}{\sum_t \gamma_t(i)}.
$$

(With multiple sequences, sums run over all time steps in all sequences.)

### (4) Dirichlet priors and MAP smoothing
Place Dirichlet priors per categorical row: $A_{i\cdot}\sim \mathrm{Dir}(\alpha_{i\cdot})$, $B_{i\cdot}\sim \mathrm{Dir}(\beta_{i\cdot})$. Posterior **mode** (MAP) updates are

$$
a_{ij}^{\mathrm{MAP}}=\frac{N_{ij}+\alpha_{ij}-1}{\sum_{j'} (N_{ij'}+\alpha_{ij'}-1)},\qquad
b_i(k)^{\mathrm{MAP}}=\frac{N_{ik}+\beta_{ik}-1}{\sum_{k'} (N_{ik'}+\beta_{ik'}-1)}.
$$

**Benefit:** avoids zeros and **degeneracies** (e.g., unseen emissions or transitions) that MLE would set to $0$, improving generalization.

<br>

## Part V. Convergence and Identifiability

### (1) EM monotonicity via Jensen’s inequality
Let $Q(\theta,\theta^{\text{old}})=\mathbb{E}_{p(\mathbf{s}\mid\mathbf{o},\theta^{\text{old}})}[\log p(\mathbf{s},\mathbf{o}\mid\theta)]$. Then

$$
\log p(\mathbf{o}\mid\theta) 
= \log \sum_{\mathbf{s}} p(\mathbf{s}\mid\mathbf{o},\theta^{\text{old}})\frac{p(\mathbf{s},\mathbf{o}\mid\theta)}{p(\mathbf{s}\mid\mathbf{o},\theta^{\text{old}})}
\ge \mathbb{E}_{p(\mathbf{s}\mid\mathbf{o},\theta^{\text{old}})}\big[\log p(\mathbf{s},\mathbf{o}\mid\theta)\big] + \mathrm{const}
= Q(\theta,\theta^{\text{old}})+\mathrm{const}.
$$
Maximizing $Q$ therefore **increases** (or leaves unchanged) $\log p(\mathbf{o}\mid\theta)$ each iteration.

### (2) Why only local optima; two mitigation strategies
The HMM likelihood is **non-convex** (latent variables), so EM can get stuck in **local maxima**.  
**Mitigations:**  
(i) Multiple random or **informed initializations** (e.g., k-means for Gaussians; supervised tagger init).  
(ii) **Priors/regularization** (Dirichlet, covariance floors), **annealing** (start with smoothed parameters, reduce smoothing), or **split–merge** heuristics for states.

### (3) Label switching problem and remedies
HMM states are **exchangeable**: permuting labels leaves the likelihood unchanged, so parameters are identifiable only up to permutation.  
**Remedies:** weakly informative **asymmetric priors**, **anchor** observations (partial supervision), or post-hoc **canonicalization** (e.g., order states by mean F0 in ASR or by emission means).

<br>

## Part VI. Applications and Model Diagnostics

### (1) POS tagging vs ASR (speech recognition)
- **POS tagging (discrete emissions or classifiers):**  
  **Evaluation:** sequence accuracy, per-tag F1.  
  **Decoding:** Viterbi over modest $N$; emissions are token-level features or categorical distributions.
- **ASR (continuous emissions):**  
  **Evaluation:** Word Error Rate (WER).  
  **Decoding:** Large search spaces with beam search / WFST composition; emissions are GMMs (or HMM-DNN hybrids historically) over acoustic frames; tunable LM scales/word insertion penalties complicate inference.

### (2) Three sanity checks for a 10-state HMM with Gaussians
1. **Normalization & positivity:** $\sum_j a_{ij}=1$ for all rows; $\pi$ sums to 1; emission pdfs valid (mixture weights $\ge 0$, sum to 1 if GMMs).  
2. **Posterior consistency:** $\sum_i \gamma_t(i)=1$ and $\sum_{i,j}\xi_t(i,j)=1$ (per $t$); no **dead states** ($\sum_t\gamma_t(i)$ not $\approx 0$ unless intended).  
3. **Numerics & monotonicity:** Covariances $\Sigma_i \succ 0$ (eigenvalues above a floor), log-likelihood **non-decreasing** across EM iterations; scaling constants $c_t$ finite (no under/overflow spikes).

### (3) HMMs vs modern neural sequence models; where HMMs still useful
- **Strengths of HMMs:** interpretable states, small-data efficiency, guaranteed monotonic EM updates, fast inference on CPU, principled uncertainty via $\gamma,\xi$.  
- **Weaknesses:** first-order memory, limited expressivity for rich observations without many parameters, local optima, independence assumptions.  
- **Neural models:** capture **long-range dependencies** and rich context; state-of-the-art accuracy but need large data/compute and can be less transparent.  
- **Still practical today:** **Forced alignment** in ASR/TTS pipelines (phoneme–frame alignment), low-resource or **embedded** keyword spotting, and as **initializers** for hybrid HMM-DNN systems.