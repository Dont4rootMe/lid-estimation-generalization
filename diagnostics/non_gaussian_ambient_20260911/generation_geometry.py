"""Posthoc fine-geometry diagnostics of preserved full-ambient samples.

Declared after the first PFGM++ image generation showed nearly correct total
variance despite poor LID. No model, sampler, scale or sample is changed.
Known renderer coordinates are measured alongside full ambient normal error.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.fair_campaign import file_sha,write_json
from models import training


def geometry(coordinates,task):
    if task=='exp':
        position=coordinates[:,0]+4
        observed=np.linalg.norm(coordinates[:,1:],axis=1)
        valid=(position>=0)&(position<=8)
        error=np.full_like(position,np.nan)
        error[valid]=abs(observed[valid]/(3*np.exp(-position[valid]))-1)
        boundaries=[0,2,4,6,8]
        definition='absolute relative radius error |radius/(3 exp(-u))-1|; zero on the Exp surface'
    else:
        radius=np.linalg.norm(coordinates,axis=1)
        position=np.divide(1.,radius,out=np.full_like(radius,np.inf),where=radius>0)
        valid=(position>=1)&(position<=100)
        phase=np.arctan2(coordinates[:,0],coordinates[:,1])
        difference=phase[valid]-position[valid]**2
        error=np.full_like(position,np.nan)
        error[valid]=abs(np.arctan2(np.sin(difference),np.cos(difference)))/np.pi
        boundaries=[1,25,50,75,100]
        definition='absolute wrapped phase error |wrap(atan2(x,y)-t_radius^2)|/pi, t_radius=1/radius; zero on Spiral; uniform phase has mean .5'
    def summary(mask):
        return dict(n=int(mask.sum()),mean=float(error[mask].mean()) if mask.any() else None,
            median=float(np.median(error[mask])) if mask.any() else None,
            q90=float(np.quantile(error[mask],.9)) if mask.any() else None)
    bins=[]
    for i,(lo,hi) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        mask=valid&(position>=lo)&((position<=hi) if i==len(boundaries)-2 else (position<hi))
        bins.append(dict(lower=lo,upper=hi,**summary(mask)))
    return dict(metric=definition,total_n=len(position),outside_domain_n=int((~valid).sum()),
        inside_domain=summary(valid),bins=bins),position,error,valid


def make(base,out,latex_table,cells=None):
    torch.set_num_threads(2)
    rows=[];table=[];source=Path(__file__);out.mkdir(parents=True,exist_ok=True)
    cells=cells or [('exp','coefficients'),('spiral','coefficients'),('exp','dataset'),('spiral','dataset')]
    fig,axes=plt.subplots(len(cells),2,figsize=(8,2.5*len(cells)),constrained_layout=True,squeeze=False)
    for i,(task,representation) in enumerate(cells):
        for j,method in enumerate(('t_flowmatching','pfgmpp')):
            run=base/'full_budget'/f'{method}__{task}__{representation}'
            receipt=json.loads((run/'generation_quality/summary.json').read_text())
            sample=run/'generation_quality/samples.npz'
            assert file_sha(sample)==receipt['samples_sha256']
            with np.load(sample) as z:
                generated=z['coordinates'];real=z['test_coordinates'];raw=z['test_raw']
                renderer=z['renderer'];offset=z['offset']
            # A separate reference to the precision floor caused only by one
            # float32 normalized-input round trip; never a modified test set.
            model=training.load_checkpoint(run/'model.pt',device='cpu')
            assert model.checkpoint_sha256==receipt['checkpoint_sha256']
            mean=model.normalization_mean.numpy();rms=np.float32(model.normalization_scale)
            rounded=((raw.astype(np.float32)-mean)/rms)*rms+mean
            rounded_coordinates=(rounded.astype(np.float64)-offset)@np.linalg.pinv(renderer)
            del model
            gen,position,error,valid=geometry(generated,task)
            control,_,_,_=geometry(real,task)
            precision_control,_,_,_=geometry(rounded_coordinates,task)
            row=dict(variant=method,task=task,representation=representation,
                generated=gen,real_test=control,float32_round_trip=precision_control,
                samples_sha256=receipt['samples_sha256'],
                generated_full_ambient_normal_ratio=receipt['metrics']['generated_normal_distance_over_data_rms_norm'],
                timing='posthoc after first PFGM++ Exp image aggregate generation result; no tuning',
                finite_endpoint='native512-step sampler followed by b at lambda1/256, not a zero-noise generator')
            rows.append(row)
            title=('Exp' if task=='exp' else 'Spiral')+' / '+('коэфф.' if representation=='coefficients' else 'рендер')
            name='t-Flow' if method=='t_flowmatching' else 'PFGM++'
            def fmt(value):return '--' if value is None else f'{value:.3g}'
            table.append([title,name,gen['outside_domain_n'],fmt(gen['inside_domain']['median']),
                fmt(gen['inside_domain']['q90']),fmt(precision_control['inside_domain']['q90'])])
            ax=axes[i,j]
            color='#2563eb' if method=='t_flowmatching' else '#d97706'
            ax.scatter(position[valid],error[valid],s=6,alpha=.4,color=color,label='Generated (inside domain)')
            if task=='exp':
                ax.set_yscale('symlog',linthresh=.02)
                ax.set(xlabel='Exp coordinate u',ylabel='Absolute relative radius error')
            else:
                ax.set(xlabel='Spiral t inferred from radius',ylabel='Absolute phase error / pi',ylim=(0,1))
                ax.axhline(.5,color='gray',ls=':',label='Uniform-phase mean')
            ax.set_title(f'{title}; {name}\n{gen["outside_domain_n"]}/512 outside generator domain',fontsize=9)
    fig.suptitle('Fine geometry of existing full-ambient samples (posthoc)\nExp: relative radius error; Spiral: wrapped phase error\nSamples and models unchanged; outside-domain counts are shown',fontsize=10)
    fig.savefig(out/'generation_fine_geometry.pdf');fig.savefig(out/'generation_fine_geometry.png');plt.close(fig)
    (out/'generation_geometry_source.py').write_bytes(source.read_bytes())
    write_json(out/'generation_fine_geometry.json',dict(status='complete',source_sha256=file_sha(source),rows=rows))
    (out/'fine_geometry_table.tex').write_text(latex_table(
        ['Данные','Модель','Вне области / 512','Медиана','90\\% квантиль','FP32 test 90\\%'],table,'llrrrr'))
    return rows
