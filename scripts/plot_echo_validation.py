#!/usr/bin/env python3
"""Figures for the corrected, implementation-checked experiment.

Minimal matplotlib style: two accent colours, hidden top/right spines,
no chart borders, no background fills, sparse ticks, no gridlines
except a faint x-grid on charts where reading exact values matters.
"""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/icwsm-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/validation/results'
OUT=ROOT/'paper/figures';OUT.mkdir(exist_ok=True)
r=json.loads((BASE/'summary.json').read_text())
if not r['complete']:
    raise SystemExit('Validation is incomplete; final manuscript figures were not generated.')

TOPICS=['gun','abortion','super_bowl','sxsw']
NAMES=['Gun control','Abortion','Super Bowl','SXSW']

# Palette: black ink for reference points, one accent per contrast dimension.
INK='#111111'
BLUE='#1F6FB2'
ORANGE='#D06A2E'
MUTED='#8A8A8A'
FAINT='#DDDDDD'

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':9,
    'axes.titlesize':9,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':9,
    'legend.fontsize':8,
    'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':0.6,
    'xtick.major.width':0.6,'ytick.major.width':0.6,
    'xtick.major.size':3,'ytick.major.size':3,
    'xtick.direction':'out','ytick.direction':'out',
})

def save(fig,name):
    for ext in ['pdf','png']:
        fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight',pad_inches=0.05,dpi=200)
    plt.close(fig)

def strip(ax,xgrid=True):
    ax.set_axisbelow(True)
    if xgrid:
        ax.grid(axis='x',color=FAINT,lw=0.5,zorder=0)
    ax.tick_params(colors=INK)

data={t:json.loads((BASE/f'{t}.json').read_text()) for t in TOPICS}

# --- Figure 3: paired averaging on identical fits, plus group contrast.
fig,axs=plt.subplots(1,2,figsize=(7.0,3.0),gridspec_kw={'width_ratios':[1.55,1]})
ax=axs[0]
for i,t in enumerate(TOPICS):
    a=[x for x in data[t]['factorial'] if x['partition_seed']==42]
    for j,x in enumerate(a):
        y=i+(j-4.5)*0.035
        ax.plot([x['user'],x['community']],[y,y],lw=0.5,color=MUTED,zorder=1)
        ax.scatter(x['user'],y,s=15,color=BLUE,zorder=3,linewidths=0)
        ax.scatter(x['community'],y,s=17,marker='s',facecolor='none',edgecolor=ORANGE,linewidths=0.9,zorder=3)
    ax.scatter([0.714,0.626,0.485,0.465][i],i+0.29,s=38,color=INK,marker='x',linewidths=1.2,zorder=4)
ax.set(yticks=range(4),yticklabels=NAMES,xlabel='Echo Chamber Score',xlim=(0.30,0.82))
ax.invert_yaxis()
strip(ax)
ax.set_title('(a) Two averages, the same fits',loc='left',pad=6)
ax.legend(handles=[
    Line2D([],[],marker='o',color=BLUE,ls='',label='User average'),
    Line2D([],[],marker='s',color=ORANGE,markerfacecolor='none',ls='',label='Community average'),
    Line2D([],[],marker='x',color=INK,ls='',label='Published'),
],loc='upper center',bbox_to_anchor=(0.5,-0.23),ncol=3,frameon=False,handletextpad=0.4,columnspacing=1.2)

ax=axs[1]
keys=['published','notebook_stored','user','community']
labels=['Published table','Stored notebook','New: user','New: community']
vals=[r['group_contrasts'][k] for k in keys]
colours=[INK,MUTED,BLUE,ORANGE]
ax.axvline(0,color=MUTED,lw=0.6)
for i,(v,c) in enumerate(zip(vals,colours)):
    ax.scatter(v,i,color=c,s=32,zorder=3,linewidths=0)
    if keys[i] in ['user','community']:
        interval=r['contrast_grid'][keys[i]]['fixed_partition']
        ax.plot([interval['min'],interval['max']],[i,i],color=c,lw=0.9)
    dy=-0.32 if i==3 else 0.32
    ax.text(v,i+dy,f'{v:+.3f}',va='center',ha='center',fontsize=8,color=INK)
ax.set(yticks=range(4),yticklabels=labels,
        ylim=(3.7,-0.6),
        xlim=(min(-0.05,min(vals)-0.04),max(vals)+0.08),
        xlabel='Difference of group means')
strip(ax)
ax.set_title('(b) Group contrast',loc='left',pad=6)
fig.subplots_adjust(wspace=0.5,bottom=0.27,top=0.9)
save(fig,'validated_averaging')

# --- Figure 4: why the SXSW averages differ. Community-size weights + per-community scores.
a=[x for x in data['sxsw']['factorial'] if x['partition_seed']==42]
sizes=np.array(a[0]['sizes']);order=np.argsort(-sizes)
means=np.mean([x['community_scores'] for x in a],axis=0)

fig,axs=plt.subplots(1,2,figsize=(7.0,2.6))
ax=axs[0]
x=np.arange(len(sizes))
ax.bar(x-0.19,100*sizes[order]/sizes.sum(),width=0.38,color=BLUE,label='User average',linewidth=0)
ax.bar(x+0.19,100/len(sizes),width=0.38,color=ORANGE,label='Community average',linewidth=0)
ax.set(xticks=x,xticklabels=[f'{n:,}' for n in sizes[order]],
       xlabel='Community size (users)',ylabel='Weight in graph score (%)')
strip(ax,xgrid=False)
ax.grid(axis='y',color=FAINT,lw=0.5,zorder=0)
ax.legend(frameon=False,fontsize=8,loc='upper right',handletextpad=0.4)
ax.set_title('(a) How much each community counts',loc='left',pad=6)

ax=axs[1]
for j,i in enumerate(order):
    vals=[row['community_scores'][i] for row in a]
    ax.scatter(np.repeat(j,10)+np.linspace(-0.08,0.08,10),vals,s=12,color=MUTED,linewidths=0,zorder=3)
    ax.scatter(j,means[i],marker='D',facecolor='white',edgecolor=BLUE,s=42,zorder=4,linewidths=0.9)
ax.set(xticks=x,xticklabels=[f'{n:,}' for n in sizes[order]],
       xlabel='Community size (users)',ylabel='Community ECS',ylim=(0,1))
strip(ax,xgrid=False)
ax.grid(axis='y',color=FAINT,lw=0.5,zorder=0)
ax.set_title('(b) Community scores across ten fits',loc='left',pad=6)
fig.subplots_adjust(wspace=0.35,bottom=0.22,top=0.86)
save(fig,'validated_sxsw_weights')

# --- Figure 5: variance decomposition of the crossed seed grid.
fig,axs=plt.subplots(1,2,figsize=(7.0,2.5),sharey=True)
for ax,key,title in zip(axs,['user','community'],['User average','Community average']):
    left=np.zeros(4)
    for label,color in [('fitting',BLUE),('partition',ORANGE),('interaction',MUTED)]:
        vals=[100*r['topics'][t]['finite_design_variation'][key]['fractions'][label] for t in TOPICS]
        ax.barh(range(4),vals,left=left,color=color,label=label.capitalize(),height=0.55,linewidth=0)
        left+=vals
    ax.set(yticks=range(4),yticklabels=NAMES,xlim=(0,100),xlabel='Share of observed squared variation (%)')
    strip(ax,xgrid=False)
    ax.set_title(title,loc='left',pad=6)
axs[0].invert_yaxis()
axs[0].legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(1.1,-0.27),handletextpad=0.4,columnspacing=1.2)
fig.subplots_adjust(bottom=0.29,wspace=0.22,top=0.85)
save(fig,'validated_variation')

# --- Figure 6: structural interventions on triangles, mixing, and score.
fig,axs=plt.subplots(1,3,figsize=(7.0,3.1),gridspec_kw={'width_ratios':[1,1,1.35]})
for i,t in enumerate(TOPICS):
    for mode,off,color,marker in [('closure',-0.14,BLUE,'o'),('mixing',0.14,ORANGE,'s')]:
        v=r['topics'][t]['structural'][mode]
        orig=v['original']
        records=v['graph_records']
        xs=[100*a['structure']['triangles']/orig['triangles'] for a in records]
        ms=[100*(a['structure']['cross_community_fraction']-orig['cross_community_fraction']) for a in records]
        ds=[a['mean_delta_user'] for a in records]
        for ax,vals in zip(axs,[xs,ms,ds]):
            ax.scatter(vals,i+off+np.linspace(-0.03,0.03,5),s=16,
                       color=color if marker=='o' else 'none',
                       edgecolor=color,marker=marker,linewidths=0.9,zorder=3)
for j,ax in enumerate(axs):
    ax.set(yticks=range(4),yticklabels=NAMES if j==0 else [])
    ax.invert_yaxis()
    strip(ax)

axs[0].set(xlabel='Triangles (% of original)')
axs[0].set_title('(a) Triangle count',loc='left',pad=6)
axs[0].axvline(100,color=MUTED,lw=0.6,ls=':')

axs[1].set(xlabel='Cross-group edges\n(change in percentage points)')
axs[1].set_title('(b) Mixing',loc='left',pad=6)

axs[2].set(xlabel='Change in user-average ECS')
axs[2].axvline(0,color=MUTED,lw=0.6)
axs[2].set_title('(c) Score response',loc='left',pad=6)

axs[1].legend(handles=[
    Line2D([],[],marker='o',color=BLUE,ls='',label='Closure-targeted'),
    Line2D([],[],marker='s',color=ORANGE,markerfacecolor='none',ls='',label='Degree-only mixing'),
],ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.42),handletextpad=0.4,columnspacing=1.2)
fig.subplots_adjust(wspace=0.22,bottom=0.33,top=0.87)
save(fig,'validated_structural')

print('Generated four corrected empirical figures.')
