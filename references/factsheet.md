# Cite-ready fact sheet

Exact statements verified by direct reading of the sources. Do not paraphrase from memory in the paper; work from this file.

## GNN expressiveness

**Morris et al., "Weisfeiler and Leman Go Neural," AAAI 2019 (arXiv:1810.02244).**
- Theorem 1: for all $t \geq 0$, all initial colourings consistent with node labels, all weights, the 1-GNN partition refines no more than the 1-WL partition. Node-level; graph-level follows via any permutation-invariant readout.
- Theorem 2: for each $t$, there exist weights that realise equality with 1-WL.

**Xu, Hu, Leskovec, Jegelka, "How Powerful are Graph Neural Networks?" ICLR 2019 (arXiv:1810.00826).**
- Lemma 2: if a GNN separates $G_1 \not\cong G_2$, then 1-WL also separates them.
- Theorem 3: sufficient conditions to match 1-WL — injective aggregation on multisets, injective readout, sufficient layers.
- Lemma 5, Corollary 6: sum aggregation on a countable feature space is injective on bounded-size multisets. GIN operationalises via sum + MLP.
- **Bound is on distinguishing power over countable feature spaces; not a uniform approximation statement over continuous features.**

**Chen, Villar, Chen, NeurIPS 2019 (arXiv:1905.12560).** Equivalence between GIso-discrimination and universal approximation of permutation-invariant continuous functions.
- Theorem 1 ($\mathcal{X}$ finite): universally approximating $\Rightarrow$ GIso-discriminating.
- Theorem 2 ($\mathcal{X}$ finite): GIso-discriminating $\Rightarrow$ $\mathcal{C}^{+2}$ universally approximating (augment class by composition with a 2-layer ReLU MLP).
- Theorem 3 ($\mathcal{X} \subset \mathbb{R}$ compact): universally approximating $\Rightarrow$ GIso-discriminating.
- Theorem 4 ($\mathcal{X}$ compact): continuous GIso-discriminating class $\Rightarrow$ $\mathcal{C}^{+2}$ universally approximating.
- All statements per fixed $n$.

**Azizian & Lelarge, "Expressive power of invariant and equivariant graph neural networks," ICLR 2021 (arXiv:2006.15646).**
- Proposition 3: message-passing GNN expressiveness $= 2$-WL$_I$ expressiveness (their convention: their "$2$-WL" $=$ standard node-level 1-WL). Higher-order variants match higher $k$-WL.
- Theorem 4 (approximation, fixed $n$, compact feature domain): $\mathrm{closure}(\mathrm{MGNN}_I) = \{f \in C_I(K, F) : \rho(2\text{-WL}_I) \subseteq \rho(f)\}$. Equivalently: continuous 1-WL-invariant functions on compact bounded-size graph spaces are exactly the uniformly-approximable-by-MPNN class.
- **This is the citation for the $\supseteq$ direction of our dichotomy theorem.**

## Polarization measures

**Hohmann, Devriendt, Coscia, "Quantifying ideological polarization on a network using generalized Euclidean distance," Science Advances 9, eabq2044 (2023).** Not Nature Communications.
- Definition: for graph $G$ and opinion $o \in [-1, +1]^{|V|}$, decompose $o = o^+ - o^-$ (positive part, negative part). Then
$$\delta_{G, o} = \sqrt{(o^+ - o^-)^\top L^\dagger (o^+ - o^-)}$$
where $L = D - A$ is the combinatorial Laplacian and $L^\dagger$ its Moore–Penrose pseudoinverse.
- Expansion: $\delta_{G, o}^2 = \mathbb{E}[\omega_{X^+, X^-}] - \tfrac{1}{2} \mathbb{E}[\omega_{X^+, X^+} + \omega_{X^-, X^-}]$, effective-resistance metric.
- Disconnected graphs: pseudoinverse applied per component.
- Feature domain: $o \in [-1, +1]^{|V|}$; binary $\pm 1$ is a special case.
- **Purely spectral in $G$: depends on Laplacian eigenpairs, not on triangle counts directly.**
- Empirical: Twitter (three 2015–2016 topical debates; 2020 US election) and US House roll-call across 36 Congresses. Not Reddit.

**Garimella et al., "Quantifying Controversy on Social Media," WSDM 2018.** Random Walk Controversy: given bipartition $\{A, B\}$, run random walks terminating at high-degree side-nodes; $\mathrm{RWC} = P_{AA} P_{BB} - P_{AB} P_{BA}$. Requires bipartition.

**Waller & Anderson, "Quantifying social organization and political polarization in online platforms," Nature 2021.** word2vec-style skip-gram on user co-commenting; polarization read off a one-dimensional axis defined by anchor communities. Reddit, 10k subreddits.

## Audit target shortlist (all on Twitter; Reddit port required)

**EchoGAE / Echo Chamber Score (Alatawi, Sheth, Liu, ASONAM 2023; arXiv:2307.04668).**
- 2-layer GCN encoder in graph auto-encoder; mean aggregation; per-community centroids used to compute cohesion minus separation.
- Score is embedding-space homophily by construction.
- Datasets: four Twitter retweet graphs (gun, abortion, Super Bowl, SXSW). Small (2–7k users).
- Code: https://github.com/faalatawi/echo-chamber-score.
- **Primary audit target: the score is a homophily statistic dressed as an echo-chamber measure.**

**InfoVGAE (Li et al., SIGIR 2022; arXiv:2110.00210).**
- Variational GAE, 2 GCN layers, rectified-Gaussian latent, TC regulariser; disentangled latent axes as readout.
- Explicit polarization framing.
- Code: https://github.com/jinningli/InfoVGAE.
- **Secondary target.**

**TIMME (Xiao et al., KDD 2020).**
- R-GCN over heterogeneous follow / retweet / mention / reply / like graph; multi-task heads for link prediction and ideology.
- Most-cited GNN ideology estimator (~180+ citations).
- Authors report "follow is the most important relation" — a homophily statement.
- Code: https://github.com/PatriciaXiao/TIMME.
- **Tertiary target if we need prominence weight.**

## Notes for the paper

- GED works only on connected graphs (or per-component). The 2 K_3 vs C_6 pair is not a clean GED separation because 2 K_3 is disconnected and opinion masses live on different components. Cite the Laplacian *spectrum* difference $\{0, 0, 3, 3, 3, 3\}$ vs $\{0, 1, 1, 3, 3, 4\}$ as the witness for non-1-WL invariance of any spectral measure; do not put a specific GED number on that pair.
- Azizian–Lelarge, not Chen–Villar–Chen, is the sharpest citation for the $\supseteq$ direction of the dichotomy — it gives the exact closure equal to continuous 1-WL-invariant functions on compact bounded-size graph spaces.
- All audit-target GNNs are Twitter, not Reddit. Either retrain on Waller–Anderson (added scope) or audit on their native Twitter data (weakens the Reddit angle). Both are viable; choose after we scope the empirical.
