#!/usr/bin/env python3
"""Design schematic. Minimal matplotlib style: outlined rectangles,
no fills, single-tone arrows, plain text."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/icwsm-matplotlib')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch

OUT=Path(__file__).resolve().parents[1]/'paper/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})

INK='#111111'
MUTED='#7A7A7A'
ACCENT='#1F6FB2'

fig,ax=plt.subplots(figsize=(7.0,2.25))
fig.subplots_adjust(left=0.01,right=0.99,bottom=0.02,top=0.98)
ax.set(xlim=(0,10),ylim=(0,3.25))
ax.axis('off')

def box(x,y,w,h,title,detail,accent=False):
    edge=ACCENT if accent else INK
    ax.add_patch(Rectangle((x,y),w,h,facecolor='none',edgecolor=edge,lw=0.9))
    ax.text(x+w/2,y+h*0.72,title,ha='center',va='center',fontsize=9,color=INK)
    ax.text(x+w/2,y+h*0.30,detail,ha='center',va='center',fontsize=8,color=MUTED)

def arrow(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,color=MUTED,lw=0.7))

box(0.05,1.10,2.00,1.05,'Verified inputs','Same users, graph,\nand regenerated text')
box(2.65,2.05,2.40,1.05,'10 partitions','Vary community detection,\nkeep representations fixed')
box(2.65,0.10,2.40,1.05,'10 model fits','Vary split and fitting,\nkeep partitions fixed')
box(5.65,1.10,1.80,1.05,'Cross both','100 evaluations\nper topic')
box(8.05,2.05,1.85,1.05,'User average','Equal weight\nfor every user')
box(8.05,0.10,1.85,1.05,'Community\naverage','Equal weight\nfor every community',accent=True)

arrow((2.10,1.80),(2.60,2.55))
arrow((2.10,1.40),(2.60,0.65))
arrow((5.10,2.55),(5.60,1.85))
arrow((5.10,0.65),(5.60,1.35))
arrow((7.50,1.85),(8.00,2.55))
arrow((7.50,1.35),(8.00,0.65))

for ext in ['pdf','png']:
    fig.savefig(OUT/f'validated_design.{ext}',bbox_inches='tight',pad_inches=0.03,dpi=200)
plt.close(fig)
