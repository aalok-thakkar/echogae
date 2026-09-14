# Related work map

Each entry: what we cite it for, the exact claim we lean on, and the strength of the dependency (load-bearing / supporting / contextual).

## Expressiveness of message-passing GNNs

- Morris et al., "Weisfeiler and Leman Go Neural" (AAAI 2019). MPNN expressiveness upper-bounded by 1-WL. **Load-bearing.**
- Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019). GIN as the maximally expressive 1-WL-bounded MPNN. **Load-bearing.**
- Barcelo et al., "The Logical Expressiveness of Graph Neural Networks" (ICLR 2020). Logical characterisation matching 1-WL. **Supporting.**

## Beyond 1-WL

- Morris et al., "Weisfeiler and Leman Go Sparse: k-GNNs." Higher-order WL. **Supporting (escape route).**
- Bouritsas et al., "Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting" (TPAMI 2022). Substructure counts. **Supporting (escape route).**

## Structural polarization measures

- Hohmann, Devriendt, Coscia, "Quantifying Ideological Polarization on a Network" (Nature Comm. 2023). Generalized Euclidean distance. **Load-bearing.**
- Waller & Anderson, "Quantifying Social Organization and Political Polarization in Online Platforms" (Nature 2021). Community embeddings. **Load-bearing (empirical target and dataset).**
- Garimella, Morales, Gionis, Mathioudakis, "Quantifying Controversy on Social Media" (WSDM 2018). Random-walk controversy. **Supporting.**

## Ideal-point models (out-of-scope, cited to bound the claim)

- Barbera, "Birds of the Same Feather Tweet Together" (Political Analysis 2015). Bayesian ideal-point estimation on Twitter follower graphs. **Contextual: not affected by the impossibility.**

## GNNs applied to polarization / stance / community

- To fill in: recent ICWSM / WebConf / EMNLP papers using MPNNs as alignment or polarization estimators. These are the practices the paper's claim targets.

## Notes

- Verify each "load-bearing" citation reads as stated. Cite-and-move-on is fine for supporting; load-bearing entries need a checked quote and page number.
- If a load-bearing claim turns out weaker than needed, flag on `tasks/todo.md` and revisit theorem statement before writing the proof.
