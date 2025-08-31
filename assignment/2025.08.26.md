# [Assignment] Markov Chains in Natural Language Processing

## Part I. Fundamentals of Markov Chains

### 1) Markov property, simplification, and linguistic examples
**Definition.** A (time-homogeneous) Markov chain over states $\mathcal{S}$ satisfies

$$
\Pr(X_t = x_t \mid X_{t-1}=x_{t-1},\dots,X_1=x_1)
= \Pr(X_t = x_t \mid X_{t-1}=x_{t-1})
= P_{x_{t-1},x_t}.
$$

This **first-order Markov property** says the next state depends only on the current state.

**How it simplifies modeling.**  
It reduces an intractable conditional over the whole history to a **local** conditional. Estimation then uses **bigram** counts rather than full histories; storage and computation scale with $|\mathcal{S}|^2$ instead of exploding with sequence length.

**Where it is reasonable.**  
Local collocations: after “of”, the next word is often “the” (English), i.e., $\Pr(\text{the}\mid\text{of})$ is high, and conditioning on earlier words changes this only slightly.

**Where it breaks down.**  
Long-distance agreement and discourse phenomena: in “The **keys** to the old cabinet **are**…”, the verb “are” depends on “keys”, not merely the immediately previous word; similarly, coreference and topic shifts require longer context than a single-step dependency.

---

### 2) Row-stochasticity of powers of a transition matrix
**Claim.** If $P$ is **row-stochastic** (each row sums to 1, entries nonnegative), then $P^t$ is row-stochastic for all integers $t\ge 1$.

**Proof (by induction).**  
Base case: $t=1$ holds by assumption.  
Inductive step: assume $P^t$ is row-stochastic. Then

$$
P^{t+1} = P^t P.
$$

The $(i,\cdot)$-row sum of $P^{t+1}$ is

$$
\sum_j (P^{t+1})_{ij} \;=\; \sum_j \sum_k (P^t)_{ik} P_{kj}
\;=\; \sum_k (P^t)_{ik} \underbrace{\sum_j P_{kj}}_{=1}
\;=\; \sum_k (P^t)_{ik} \;=\; 1,
$$

and entries remain nonnegative as sums of products of nonnegative numbers. Thus $P^{t+1}$ is row-stochastic. By induction, $P^t$ is row-stochastic for all $t\ge 1$. $\square$

**Intuition for language modeling.**  
The $i$-th row of $P^t$ is a valid **$t$-step next-word distribution** given the current word is $i$. Composing steps preserves “proper distributions,” so multi-step predictions remain probabilistic.

<br>

### 3) Chapman–Kolmogorov for multi-step word transitions
**Equation.** For any $s,u\in\mathcal{S}$ and $t\ge 1$,

$$
(P^{t+1})_{su}
= \sum_{v\in\mathcal{S}} (P^t)_{sv} \, P_{vu}.
$$

**Interpretation (bigrams → multi-steps).**  
If bigrams give us one-step probabilities $P_{ij}$, then **two-step** transitions are

$$
(P^2)_{ij} = \sum_k P_{ik} P_{kj},
$$

i.e., the probability to go from word $i$ to word $j$ in two steps equals the sum over all mediating words $k$ of “$i\!\to\!k$ then $k\!\to\!j$”. Likewise for longer horizons with $P^t$.

<br>

## Part II. Stationary Distribution and Ergodicity

### 1) Stationary distribution of the weather model
Let

$$
P = \begin{bmatrix}
0.7 & 0.3\\
0.4 & 0.6
\end{bmatrix},\quad
\pi = [\pi_R,\ \pi_S],\quad \pi \mathbf{1}=1,\ \pi\ge 0.
$$

Solve $\pi = \pi P$:


$$
\begin{cases}
\pi_R = 0.7\pi_R + 0.4\pi_S \\
\pi_S = 0.3\pi_R + 0.6\pi_S \\
\pi_R + \pi_S = 1
\end{cases}
\quad \Rightarrow\quad
0.3\pi_R = 0.4\pi_S \Rightarrow \dfrac{\pi_R}{\pi_S} = \dfrac{4}{3}.
$$

Normalize: $\pi_R = \frac{4}{7},\ \pi_S = \frac{3}{7}$.

**Interpretation.**  
In the long run, it is rainy $\frac{4}{7}\approx 57.1\%$ of days and sunny $\frac{3}{7}\approx 42.9\%$.

---

### 2) Irreducibility, aperiodicity, uniqueness, and convergence
- **Irreducible:** every state can reach every other with positive probability in some number of steps.  
- **Aperiodic:** each state’s return times have gcd equal to 1 (no strict cycling).

**Guarantees.** For a finite chain, **irreducible + aperiodic** (i.e., ergodic) implies a **unique** stationary distribution $\pi$, and for any start, $P^t(x,\cdot)\to \pi(\cdot)$ as $t\to\infty$.

**Counterexamples.**  
- Not irreducible: two disjoint classes (e.g., words $\{\text{New},\ \text{York}\}$ and $\{\text{Los},\ \text{Angeles}\}$ with no cross-links). Multiple stationary distributions exist; convergence depends on the starting class.  
- Periodic: a two-cycle $A\leftrightarrow B$ yields period 2; the distribution oscillates and does not converge to a single limit (though the **Cesàro average** does).

---

### 3) Word-level stationary distribution and unigram counts
For a word-level chain with stationary $\pi$, $\pi_w$ equals the **long-run proportion** of tokens that are $w$. In a large corpus,

$$
\widehat{\pi}_w \approx \frac{\text{occurrences of } w}{\text{total tokens}},
$$

i.e., empirical **unigram** frequencies are consistent estimators of $\pi$ when text is well-modeled as a stationary ergodic process.

<br>

## Part III. Mixing, Spectral Gap, and Convergence

### 1) Spectral gap and mixing time
Let eigenvalues of $P$ (row-stochastic) satisfy $1=\lambda_1 > |\lambda_2|\ge \cdots$. The **spectral gap** is

$$
\gamma \;=\; 1 - |\lambda_2|.
$$

Heuristically (and tightly under reversibility),

$$
\text{mixing time } \tau_{\mathrm{mix}}(\varepsilon) \;\lesssim\; \frac{1}{\gamma}\,\log\!\left(\frac{1}{\varepsilon\,\pi_{\min}}\right),
$$

so a **larger gap** $\gamma$ implies **faster** convergence to stationarity.

---

### 2) Effect of frequent function words on mixing
Function words like “the”, “of”, “to” act as **hubs** with broad connectivity: many words transition to and from them with moderate probability. This improves **conductance**, typically **increasing the spectral gap** and speeding mixing. In contrast, rare domain-specific words form **weakly connected pockets** (nearly absorbing neighborhoods), shrinking the gap and slowing convergence.

---

### 3) Two chains with the same vocabulary
- **Ergodic with large gap:** Rapidly “forgets” its start, offers smooth, diversified next-word predictions. Better for robust language modeling where coverage and fluency matter.  
- **Nearly reducible (tiny gap):** Gets “stuck” in topical islands; next-word predictions are brittle and overconfident within a narrow subdomain, poor at handling domain shifts.

---

## Part IV. Empirical Estimation from Corpora

### 1) Estimating $\widehat P$ and $\widehat \pi$
Let $C(i,j)$ be bigram counts (# times word $i$ followed by $j$), and $C(i)=\sum_j C(i,j)$ be outgoing counts. Then

$$
\widehat{P}_{ij} = \frac{C(i,j)}{C(i)},\qquad
\widehat{\pi}_i = \frac{\text{tokens equal to } i}{\text{total tokens}}.
$$

Under stationarity, $\widehat{\pi}$ is also the **left eigenvector** of $\widehat P$ with eigenvalue 1 (approximately, for large corpora).

---

### 2) Why outgoing and incoming counts agree (approximately)
For interior positions, each token contributes **one outgoing** and **one incoming** bigram occurrence. Differences arise only at sequence boundaries (sentence starts/ends), so discrepancies are **$O(\#\text{sequences})$**, negligible relative to total tokens in large corpora.

---

### 3) PMI in terms of $\pi$ and $P$, and a collocation example
Adjacent bigram probability in stationarity: $\Pr(i,j) = \pi_i P_{ij}$. Unigram $\Pr(i)=\pi_i$, $\Pr(j)=\pi_j$. Then

$$
\operatorname{PMI}(i,j)
= \log \frac{\Pr(i,j)}{\Pr(i)\Pr(j)}
= \log \frac{\pi_i P_{ij}}{\pi_i \pi_j}
= \log \frac{P_{ij}}{\pi_j}.
$$

**Interpretation.** PMI compares the observed next-word probability $P_{ij}$ to the baseline frequency $\pi_j$. High PMI means $j$ follows $i$ **disproportionately often** relative to how common $j$ is overall.

**Example.** For English, $\operatorname{PMI}(\text{New},\text{York})$ is high because $P_{\text{New},\text{York}}\gg \pi_{\text{York}}$; conversely, $\operatorname{PMI}(\text{the},\text{York})$ is modest despite $\pi_{\text{York}}$ due to low $P_{\text{the},\text{York}}$.

<br>

## Part V. Reversibility and Asymmetry

### 1) Detailed balance and symmetric bigram flows
**Detailed balance (reversibility).** A stationary $\pi$ satisfies

$$
\pi_i P_{ij} \;=\; \pi_j P_{ji}\quad\forall i,j.
$$
Define the **flow** $F_{ij} = \pi_i P_{ij}$. If detailed balance holds, then
$$
F_{ij} = F_{ji},
$$

so in stationarity, the expected number of $i\!\to\!j$ transitions equals $j\!\to\!i$ transitions—**symmetric bigram flows**.

---

### 2) Why natural language is typically non-reversible
Language exhibits **directional syntax and collocations**:
- “of the” is frequent, but “the of” is rare: $P_{\text{of},\text{the}}\gg P_{\text{the},\text{of}}$.
- “New York” $\gg$ “York New”.
- Verb–object order in English (“eat apples”) vs. object–verb is rare outside marked constructions.  
Hence $\pi_i P_{ij} \ne \pi_j P_{ji}$ in general; detailed balance fails.

---

### 3) When reversibility might be a useful approximation
At **abstraction layers** where directionality is weaker—e.g., **POS tags** in balanced corpora (NOUN–ADP vs. ADP–NOUN flows aggregated) or **topic/state transitions** in coarse discourse models—assuming near-reversibility can simplify spectral analysis and yield tractable inference, with limited loss for certain analytics tasks.

<br>

## Part VI. Interpretations and Extensions

### 1) States-as-words vs. states-as-POS-tags
- **Words**
  - **Pros:** Captures lexical selection, idioms, named entities, collocations.
  - **Cons:** Huge state space, sparsity, topical non-stationarity, domain shift.
- **POS tags**
  - **Pros:** Compact state space, better generalization across domains, highlights syntactic patterns (e.g., ADJ→NOUN).
  - **Cons:** Loses lexical semantics; cannot model entity-specific phenomena or idioms.

**Takeaway:** POS-level chains are good for **structural** regularities; word-level chains are better for **lexical** phenomena but need smoothing/regularization.

---

### 2) Markov chains over higher-level abstractions
Consider **dialogue acts** as states: $\{\text{GREETING}, \text{QUESTION}, \text{ANSWER}, \text{ACK}, \text{CLOSING}\}$.
A plausible transition sketch:

$$
\text{GREETING} \to \text{QUESTION} \to \text{ANSWER} \to \text{ACK} \to \text{QUESTION} \to \text{ANSWER} \to \text{CLOSING}.
$$

Estimating a transition matrix over these acts supports **turn-taking prediction**, **agent policy shaping**, and **conversation analysis** with far fewer states than words.

---

### 3) Strengths and limitations vs. HMMs and neural models
- **Strengths of (visible) Markov chains**
  - Simple, transparent, few parameters; closed-form linear-algebra tools.
  - Sample-efficient on small datasets; easy to estimate and analyze.
- **Limitations**
  - **Short memory**: cannot capture long-distance dependencies.
  - **Stationarity**: real text is non-stationary across topics/registers.
  - **Sparsity & asymmetry**: bigram graphs are skewed; many zero/low-prob edges.
- **Compared to HMMs**
  - HMMs separate **latent structure** (states) from **observations**, capturing regime-switching (e.g., POS tags). Inference (forward–backward) leverages sequences beyond one step.
- **Compared to neural sequence models**
  - Transformers model **long context**, semantics via embeddings/attention; far better predictive performance and generalization but less interpretable and more data-hungry.