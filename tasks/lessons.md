# Lessons

Corrections from the user, with the rule that prevents the repeat. Read at the start of each session.

## Do not describe a perturbation by its intent; measure it on the graph the model consumes

Wrong: writing "the rewiring removes higher-order structure" for TIMME because the swap routine was adapted from the closure-destroying EchoGAE rewire. TIMME binarises weighted relation records, so the same swap increased triangle counts 2-6x on the graph it sees.

Rule: before a sentence in the paper says what a rewiring preserves or destroys, compute the statistic (unique edges, triangles, clustering, neighbourhood composition) on the exact input the audited model builds, before and after, and put the numbers in a table. Same for "within X of the published value": pull the published table and quote it.
