#!/usr/bin/env python3
"""Generate numerical manuscript passages from the complete controlled results."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/validation/results';OUT=ROOT/'paper/generated'
r=json.loads((BASE/'summary.json').read_text())
if not r['complete']:raise SystemExit('Refusing manuscript numbers from an incomplete design.')
ts=['gun','abortion','super_bowl','sxsw'];names=['Gun control','Abortion','Super Bowl','SXSW'];pub=[.714,.626,.485,.465];stored=[.627835,.621917,.561706,.464753]
def put(name,text):(OUT/(name+'.tex')).write_text(text.strip()+'\n')
def f(x):return f'{x:.3f}'
def signed(x):return f'{x:+.3f}'
def interval(s):return f"[{signed(s['min'])}, {signed(s['max'])}]"
rows=[]
for t,n,p,nb in zip(ts,names,pub,stored):
 q=r['topics'][t];u=q['fixed_partition']['user'];c=q['fixed_partition']['community']
 rows.append(f"{n} & {p:.3f} & {nb:.3f} & ${f(u['mean'])}\\pm{f(u['sd'])}$ & ${f(c['mean'])}\\pm{f(c['sd'])}$ & ${signed(q['averaging_difference']['mean'])}$ \\\\")
text=r'''\section{Results: Definition, Reference, and Variation}\label{sec:results}

\subsection{The Averaging Unit Changes the Comparison}

\begin{table*}[t]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Topic & Published & Stored notebook & New: user average & New: community average & Paired difference\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\caption{Three reference records, distinguished explicitly. New executions use the notebook's 20/10-dimensional encoder and seed-42 partition, with regenerated text features. Entries report mean $\pm$ standard deviation across ten fitting seeds. The final column is community minus user averaging on identical fitted representations, averaged across those ten pairs. Historical numbers are documentary references, not additional audit runs.}
\label{tab:validated}
\end{table*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/validated_averaging.pdf}
\caption{Averaging changes the measurement without changing the model. (a) Each line joins two scores calculated on the same fit and partition; there are ten fits per topic. Crosses mark the published table. (b) The mean of gun control and abortion minus the mean of Super Bowl and SXSW, with each topic weighted equally. Historical and new records are kept separate. For new executions, horizontal lines span the ten fitting-seed contrasts; they are not confidence intervals or uncertainty across topics.}
\label{fig:averaging}
\end{figure*}

Table~\ref{tab:validated} reports both averages for every topic. Because each pair uses the same distances and communities, its difference has a direct explanation: the graph score assigns different weights to the community means. The effect is not an artefact of fitting a second model.

'''
s=r['topics']['sxsw'];d=s['averaging_difference']
text+=f"On SXSW, equal community weighting raises the score by ${f(d['mean'])}$ on average, with paired differences from ${f(d['min'])}$ to ${f(d['max'])}$. Its user-average ECS is ${f(s['fixed_partition']['user']['mean'])}$; its community-average ECS is ${f(s['fixed_partition']['community']['mean'])}$. Every one of the ten fits shows this direction. The user average is close to the published $0.465$, whereas the community average is substantially higher.\n\n"
text+=r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/validated_sxsw_weights.pdf}
\caption{Why the SXSW averages differ. (a) Actual community sizes determine the weights under user averaging; community averaging assigns 20\% to every group. (b) Community scores from the same ten model fits, with diamonds marking their means. Small communities can contribute strongly to the equal-community score while representing few users.}
\label{fig:weights}
\end{figure*}

Figure~\ref{fig:weights} makes the mechanism visible. The fixed SXSW partition has communities of 1,568, 712, 85, 54, and 17 users. Under user averaging, the largest contributes 64.4\% of the score and the smallest 0.7\%. Under community averaging, each contributes 20\%. The numerical discrepancy is therefore also a substantive choice: how much should a small separated group count in a description of the entire conversation?

'''
g=r['group_contrasts']
text+=f"The published polarising-minus-non-polarising group contrast is ${signed(g['published'])}$. The stored notebook gives ${signed(g['notebook_stored'])}$. Under the corrected new executions, the contrast is ${signed(g['user'])}$ with user averaging and ${signed(g['community'])}$ with community averaging (Figure~\\ref{{fig:averaging}}b). These are comparisons among these four named topics, not estimates of a population-level political versus non-political effect.\n\n"
grid=r['contrast_grid']
text+=f"At the fixed partition, the ten user-average contrasts range from ${signed(grid['user']['fixed_partition']['min'])}$ to ${signed(grid['user']['fixed_partition']['max'])}$. The corresponding community-average contrasts range from ${signed(grid['community']['fixed_partition']['min'])}$ to ${signed(grid['community']['fixed_partition']['max'])}$. Thus the corrected user-average comparison does not reverse the published direction, although its magnitude is smaller.\n\n"
text+=f"A descriptive extension evaluates the contrast at every common fitting/partition seed pair. Of 100 indexed configurations, {grid['undefined']} are undefined because at least one topic has only one community. User averaging gives a positive contrast in all {grid['user']['all_defined']['n']} defined configurations; community averaging reverses its sign in {grid['opposite_signs']}. This is a count over an explicitly enumerated configuration grid, not a probability of reversal across topics or a significance test. The extension and its timing are recorded in the protocol.\n\n"
text+=r'''\subsection{Fitting and Partition Choice Are Different Sources of Variation}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/validated_variation.pdf}
\caption{Sources of variation in the crossed seed experiment. Bars separate row (fitting), column (partition), and remaining interaction contributions to the observed mean squared deviation. Both averaging rules use the same fitted representations and detected partitions. Undefined partition columns are counted in the text and excluded here. These fractions describe a finite seed grid, not uncertainty across social populations.}
\label{fig:variation}
\end{figure*}

The crossed design shows why a single ``random seed'' is not an adequate account of robustness. Changing a partition can change both which users are compared and which community is nearest, even when every fitted representation remains fixed. Changing a fitting seed instead changes the edge split and learned geometry while preserving the chosen partition.

'''
undef=[(n,r['topics'][t]['undefined_cells']) for t,n in zip(ts,names) if r['topics'][t]['undefined_cells']]
if undef:
 text+='One-community partitions produced undefined scores for '+', '.join(f"{n} ({v} of 100 cells, or {v//10} of ten partitions)" for n,v in undef)+'. These are reported as failures to define the requested comparison, not as low echo-chamber scores. The remaining partition columns form the balanced numerical table in Figure~\\ref{fig:variation}.\n\n'
else:text+='All 400 cells have at least two communities and yield defined scores.\n\n'
for t,n in zip(ts,names):
 q=r['topics'][t]['finite_design_variation']['user']['fractions']
 text+=f"For {n}, partition choice accounts for {100*q['partition']:.1f}\\% and fitting for {100*q['fitting']:.1f}\\% of the observed user-score variation. "
text+='These percentages are conditional on the selected seed ranges and should not be read as universal properties of the model.\n\n'
text+=r'''\subsection{Configuration Controls and Remaining Historical Differences}

The wider-encoder control changes only the requested dimensions while keeping the largest-component input and partition fixed. Across five matched fitting seeds, its mean change in user-average ECS is '''+', '.join(f"${signed(r['topics'][t]['architecture']['user']['mean'])}$ for {n}" for t,n in zip(ts,names))+r'''. These controls quantify the consequence of using function defaults instead of the notebook's explicit settings; they do not establish which historical configuration produced the publication's table.

Retaining feature-covered users outside the largest component is a further change of measurement population. Assigning those users to singleton communities also changes the distance comparisons and, under equal-community averaging, the weight of very small groups. The artifact reports both widths and all three averaging conventions for this diagnostic. Gun control yields undefined user and community averages: six users at width 20 and four at width 100 have zero within- and nearest-between-community distances. Excluding singleton communities produces a different, defined quantity; it does not repair the original definition. We do not combine this population change with an averaging change and call their joint effect a failure to reproduce the notebook.

A partition-only check under NetworkX 3.1, the version named by the original release, reproduces all 40 current-version partitions exactly and gives identical scores on the fixed representations. This version substitution therefore does not explain the partition differences from the historical table: for example, the paper reports six SXSW communities, while both tested versions yield five at seed 42.

The fresh isolated-environment reference fit and unmodified-model check agree exactly with the primary SXSW seed-42 representations. The remaining differences between Table~\ref{tab:validated}'s historical and new records therefore cannot be explained by the three implementation mismatches corrected here alone. Feature history, historical dependencies, and undocumented execution choices remain possible explanations; the audit does not assign the residual difference to any one of them without evidence.
'''
put('validated_results',text)
# Structural tables: independent units are graph draws, each mean over three fits.
rows=[]
for t,n in zip(ts,names):
 for mode,label in [('closure','Closure'),('mixing','Mixing')]:
  q=r['topics'][t]['structural'][mode];orig=q['original'];rs=q['graph_records'];tri=[x['structure']['triangles'] for x in rs];mix=[100*x['structure']['cross_community_fraction'] for x in rs]
  rows.append(f"{n} & {label} & {min(tri):,}--{max(tri):,} & {min(mix):.1f}--{max(mix):.1f} & ${signed(q['delta_user']['mean'])}$ & ${interval(q['delta_user'])}$ \\\\")
text=r'''\subsection{Observed Structural Changes and Score Responses}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/validated_structural.pdf}
\caption{Two structural interventions, checked separately. Each condition has five graph realisations per topic. Panels show triangle counts relative to the original graph, change in the cross-community edge fraction, and the user-average score response. The dotted line marks the original triangle count. Each score point averages three within-seed differences for one graph; it is not an independent topic or a confidence interval. Closure-targeted swaps preserve every user's community-neighbour counts; degree-only mixing permits those counts to change.}
\label{fig:structural}
\end{figure*}

\begin{table*}[t]
\centering\small
\begin{tabular}{llrrrr}
\toprule
Topic & Condition & Triangles remaining & Cross-group edges (\%) & Mean ECS change & Range of graph means\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\caption{Verified structural outcomes. Triangle counts and mixing percentages span the five supplied graphs per condition; ECS changes are user-average rewired-minus-original differences, first averaged across three matched fitting seeds within each graph. The final range spans those five graph means. Original structure and all community-average deltas are included in the artifact.}
\label{tab:structural}
\end{table*}

All 40 rewired graphs preserve every user's degree, and all 20 closure-targeted graphs preserve each user's complete community-neighbour counts. The closure condition leaves the cross-community fraction unchanged by construction. The degree-only condition is evaluated by its measured mixing change rather than an assumed random-graph limit.

'''
text+=r"""Figure~\ref{fig:structural} shows a consistent mixing response: every graph-level mean decreases in all four topics. The mean changes are """+', '.join(f"${signed(r['topics'][t]['structural']['mixing']['delta_user']['mean'])}$ for {n}" for t,n in zip(ts,names))+r""". This supports sensitivity to changed group connections under this intervention.

Closure-targeted responses differ. Their mean changes are """+', '.join(f"${signed(r['topics'][t]['structural']['closure']['delta_user']['mean'])}$ for {n}" for t,n in zip(ts,names))+r""". Gun control's five graph means span zero; abortion changes only slightly, whereas Super Bowl and SXSW increase. These outcomes do not support general structural blindness, and they do not require treating a triangle as ground-truth echo-chamber strength.

Degree-only mixing can also increase triangle count, as Figure~\ref{fig:structural}a shows. Thus the two interventions cannot be reduced to more versus fewer triangles. Detailed score ranges appear in Appendix~\ref{app:rewire}; connectivity changes and paired fitting results remain available in the artifact.
"""
# Keep the exact detailed table in the appendix; the main text retains all four empirical figures.
a=text.index(r'\begin{table*}');b=text.index(r'\end{table*}',a)+len(r'\end{table*}')
put('validated_structural_table',text[a:b]);text=text[:a]+text[b:]
put('validated_structural_results',text)
put('validated_abstract',r'''\begin{abstract}
Comparing echo-chamber scores across online conversations requires a consistent definition of what the score averages. We audit EchoGAE's four-topic comparison and find that its published equation weights communities equally while its released notebook weights users equally. We verify the released source and raw inputs, regenerate text features, and separate fitting from community detection in a crossed seed experiment. On SXSW, the same ten fitted representations yield a mean score of $0.467$ under user averaging and $0.655$ under community averaging; the former is close to the published $0.465$. We then test structural response with five graph realisations and three fits per realisation under each of two interventions: triangle removal preserving each user's community connections, and degree-preserving rewiring that changes those connections. The corrected user-average topic contrast remains positive; equal community weighting reduces its mean from $0.131$ to $0.036$ at the reference partition. Every degree-only graph draw lowers the user-average score after averaging matched fits, while closure-targeted responses vary by topic. The audit distinguishes a measurement-definition discrepancy from a reproduction failure. Its findings concern these implementations and topics, not a universal claim that echo-chamber scores or graph neural networks are uninformative.
\end{abstract}''')
put('validated_conclusion',r'''\section{Conclusion}

A cross-topic score comparison can change because the unit being averaged changes, even when the users, graph, communities, and fitted representations do not. EchoGAE's published formula and released notebook expose exactly this ambiguity. Correcting the implementation comparison recovers the published SXSW score closely under user averaging and explains a large alternative value through community weighting.

The structural tests also give positive evidence: every degree-only graph draw lowers the user-average score, while closure-targeted responses differ by topic. These interventions test different properties. Their responses should be reported with repeated graph draws, explicit seed roles, and measured structural changes. The resulting critique is specific and testable: comparisons need an explicit averaging unit, traceable execution, and evidence linking structural response to the intended social concept.
''')
print('Generated corrected numerical manuscript passages.')
