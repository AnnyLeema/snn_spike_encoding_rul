import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family']=['Liberation Sans','DejaVu Sans']
C=dict(grid=('#E3E7EB','#37474F'),sub=('#DCE8F5','#2E5E8E'),tr=('#FBE8D0','#B86A12'),
       sur=('#DDF1E4','#2E7D4F'),def_=('#FDE2E2','#C62828'),gray=('#F2F2F2','#757575'),
       pur=('#EDE7F6','#5E35B1'),teal=('#E0F2F1','#00796B'),yel=('#FFF8E1','#B08900'))
def fig(w,h):
    f,ax=plt.subplots(figsize=(w,h)); ax.set_xlim(0,w); ax.set_ylim(0,h); ax.axis('off'); return f,ax
def box(ax,x,y,w,h,text,kind='gray',fs=10,bold=False,dashed=False,align='center',r=0.08,lw=1.3):
    fc,ec=C[kind]
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,ls='--' if dashed else '-',zorder=2))
    ax.text(x if align=='center' else x-w/2+0.1,y,text,ha=align,va='center',fontsize=fs,fontweight='bold' if bold else 'normal',zorder=3,linespacing=1.25)
    return (x,y,w,h)
def arr(ax,p,q,color='#555555',lw=1.3,text=None,tc=None,fs=9,rad=0,ls='-',off=(0,0),head='-|>',ms=12,shrinkA=0,shrinkB=0,both=False):
    a=FancyArrowPatch(p,q,arrowstyle=('<|-|>' if both else head),mutation_scale=ms,color=color,lw=lw,ls=ls,
        connectionstyle=f"arc3,rad={rad}",shrinkA=shrinkA,shrinkB=shrinkB,zorder=1)
    ax.add_patch(a)
    if text:
        mx=(p[0]+q[0])/2+off[0]; my=(p[1]+q[1])/2+off[1]
        ax.text(mx,my,text,ha='center',va='center',fontsize=fs,color=tc or color,zorder=4,
                bbox=dict(fc='white',ec='none',pad=1.2))
def line(ax,p,q,color='#9E9E9E',lw=0.9,ls='-'):
    ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=lw,ls=ls,zorder=1)
def save(f,name):
    f.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.08,facecolor='white')
    f.savefig(name+'.pdf',bbox_inches='tight',pad_inches=0.08)
    from PIL import Image; im=Image.open(name+'.png'); im.thumbnail((1600,1600)); import os; pass
def poly(ax,pts,color='#555555',lw=1.3,ls='-',text=None,tpos=None,fs=9,ms=12):
    for p,q in zip(pts[:-2],pts[1:-1]): ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=lw,ls=ls,zorder=1,solid_capstyle='round')
    arr(ax,pts[-2],pts[-1],color=color,lw=lw,ls=ls,ms=ms)
    if text: ax.text(tpos[0],tpos[1],text,fontsize=fs,color=color,ha='center',va='center',bbox=dict(fc='white',ec='none',pad=1),zorder=4)
