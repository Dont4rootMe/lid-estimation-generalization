"""Auditable tables and standalone plots from the completed fixed campaign."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.fair_campaign import verify_measurements,write_json,file_sha
from experiments.fair_protocol import source_identity,digest,protocol

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_gaussian_20260911'
HERE=Path(__file__).resolve().parent
MODELS=['t_flowmatching','pfgmpp','posterior_rectified_flow']
NAMES={'t_flowmatching':'t-Flow (nu=5)','pfgmpp':'PFGM++ (D=128)',
    'posterior_rectified_flow':'Gaussian posterior RF'}
COLORS={'t_flowmatching':'#2563eb','pfgmpp':'#d97706','posterior_rectified_flow':'#059669'}
CELLS=['e6/e6_exp_pca/coefficients','e1/e1_spiral_pca/coefficients',
    'e6/e6_exp_pca/dataset','e1/e1_spiral_pca/dataset']


def selector_bootstrap(curves,target,scales):
    rng=np.random.default_rng(4771);error=abs(curves-target[:,None]);counts=np.zeros(len(scales),dtype=int)
    for _ in range(20):
        ids=rng.integers(len(curves),size=(100,len(curves)))
        mean=error[ids].mean(1)
        index=np.argmax(mean<=mean.min(1,keepdims=True)+1e-12,axis=1)
        counts+=np.bincount(index,minlength=len(scales))
    return [dict(lambda_value=float(s),count=int(n),fraction=float(n/2000))
        for s,n in zip(scales,counts) if n]


def titles(cell):
    _,dataset,rep=cell.split('/')
    return ('Exp' if dataset=='e6_exp_pca' else 'Spiral')+' / '+('coefficients, N=30' if rep=='coefficients' else 'PCA render, N=784')



def latex_table(headers,rows,alignment):
    return '\n'.join([r'\begin{center}\small',r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{'+alignment+'}',r'\toprule',
        ' & '.join(headers)+r' \\',r'\midrule',
        *[' & '.join(map(str,row))+r' \\' for row in rows],
        r'\bottomrule',r'\end{tabular}}',r'\end{center}',''])


def report_tables(out,tables):
    model_names={'t_flowmatching':'t-Flow','pfgmpp':'PFGM++',
        'posterior_rectified_flow':'Gaussian RF ($R$)'}
    def dataset(row):
        return ('Exp' if '/e6_' in row['cell_key'] else 'Spiral')+' / '+(
            'коэфф.' if row['cell_key'].endswith('/coefficients') else 'рендер')
    scores=[];oracles=[];denoising=[]
    for row in tables:
        prefix=[dataset(row),model_names[row['model']]]
        scores.append(prefix+[f"{row['true_lid']:.0f}",f"{row['response_selected_lambda']:.5g}",
            f"{row['response_test_mae']:.3f} [{row['response_mae_ci95_low']:.3f}, {row['response_mae_ci95_high']:.3f}]",
            f"{row['response_kneedle_mae']:.3f}"])
        oracles.append(prefix+[f"{row['min_lambda_learned_response_mean']:.3f}",
            f"{row['min_lambda_oracle_response_mean']:.3f}",
            f"{row['selected_lambda_model_vs_oracle_response_mae']:.3f}"])
        denoising.append(prefix+[f"{row['response_selected_lambda']:.5g}",
            f"{row['selected_lambda_denoising_risk']:.3f}",
            f"{row['selected_lambda_initial_denoising_risk']:.3f}"])
    (out/'score_table.tex').write_text(latex_table(
        ['Данные','Модель','$d$',r'$\lambda$','Test MAE [95\\% CI]','Kneedle MAE'],scores,'llrrlr'))
    (out/'oracle_table.tex').write_text(latex_table(
        ['Данные','Модель',r'$\R_{\min}$ сеть',r'$\R_{\min}$ эталон',r'$|R_{\rm model}-R_{\rm exact}|$'],oracles,'llrrr'))
    (out/'denoising_table.tex').write_text(latex_table(
        ['Данные','Модель',r'$\lambda$','Обученный posterior','Начальный preconditioner'],denoising,'llrrr'))
    new=[r for r in tables if r['model']!='posterior_rectified_flow']
    passed=sum(r['practical_half_unit_mae_check'] for r in new)
    boundary=sum(r['selected_scale_at_boundary'] for r in new)
    (out/'outcome.tex').write_text(
        f'Практический порог supervised test MAE$<0.5$ прошли {passed} из8 новых моделей/ячеек. '
        f'Граничный supervised масштаб выбран в {boundary} из8 случаев. '
        'Эти числа описывают конечную шкалу и данный бюджет. '
        'Точность при малом шуме оценивается следующей проверкой; '
        'она не следует из прохождения этого порога.\n')
def main():
    out=HERE/'results';out.mkdir(exist_ok=True)
    records={};tables=[];selection_rows=[];source=source_identity();rules=digest(protocol())
    for path in sorted((BASE/'full_budget').glob('*/complete.json')):
        done=verify_measurements(path);quality_path=path.parent/'quality/summary.json'
        quality=json.loads(quality_path.read_text())
        assert quality['status']=='complete' and quality['checkpoint_sha256']==done['checkpoint_sha256']
        assert done['source_sha256']==source and done['protocol_sha256']==rules
        assert done['kind']=='benchmark' and done['steps_completed']==128000
        assert done['test_n']==1000 and done['fit_n']==99000 and done['holdout_n']==1000
        with np.load(path.parent/'quality/response_curves.npz') as z:
            curve={k:z[k] for k in z.files}
        frozen=json.loads((path.parent/'quality/response_selection.json').read_text())
        selection=quality['response_selection'];idx=selection['selected_index']
        errors=abs(curve['test'][:,idx]-curve['test_target'])
        assert np.isclose(errors.mean(),quality['response_metrics']['mae'],rtol=0,atol=1e-12)
        bootstrap=selector_bootstrap(curve['holdout'],curve['holdout_target'],curve['scales'])
        selection_rows.append(dict(variant=done['variant'],cell=done['cell_key'],
            selection_bootstrap=bootstrap,rule='2000 source-train query bootstraps, fixed weights; original selection unchanged'))
        with np.load(path.parent/'quality/continuous_comparison.npz') as z:continuous={k:z[k] for k in z.files}
        record=dict(done=done,quality=quality,curve=curve,continuous=continuous,path=path.parent)
        key=(done['cell_key'],done['variant']);assert key not in records;records[key]=record
        at_min=quality['continuous_law'][0];at_selected=quality['continuous_law'][idx]
        risk=quality['denoising'][idx]
        tables.append(dict(dataset=titles(done['cell_key']),cell_key=done['cell_key'],model=done['variant'],
            true_lid=float(curve['test_target'][0]),test_n=1000,
            primary_readout=done['primary_readout'],primary_selected_lambda=done['selected_lambda'],
            primary_test_mae=done['metrics'][done['primary_readout']]['mae'],
            response_selected_lambda=selection['selected_lambda'],response_test_mae=float(errors.mean()),
            response_mae_ci95_low=quality['response_metrics']['mae_query_bootstrap_ci95'][0],
            response_mae_ci95_high=quality['response_metrics']['mae_query_bootstrap_ci95'][1],
            response_test_mean=quality['response_metrics']['mean'],
            response_kneedle_mae=quality['response_automatic']['mae'],
            response_kneedle_fallback_n=quality['response_automatic']['fallback_n'],
            response_kneedle_boundary_n=quality['response_automatic']['boundary_n'],
            selected_scale_at_boundary=selection['scale_unresolved'],
            selected_lambda_holdout_bootstrap_fraction=sum(b['fraction'] for b in bootstrap if b['lambda_value']==selection['selected_lambda']),
            min_lambda_learned_response_mean=at_min['learned_response_mean'],
            min_lambda_oracle_response_mean=at_min['oracle_response_mean'],
            min_lambda_model_vs_oracle_response_mae=at_min['learned_vs_oracle_response_mae'],
            selected_lambda_model_vs_oracle_response_mae=at_selected['learned_vs_oracle_response_mae'],
            selected_lambda_posterior_error_over_lambda_rms=at_selected['posterior_error_over_lambda_rms'],
            selected_lambda_denoising_risk=risk['trained_mean_risk'],
            selected_lambda_initial_denoising_risk=risk['initial_mean_risk'],
            practical_half_unit_mae_check=quality['practical_half_unit_mae_check'],
            parameters=done['resolved']['actual_parameters'],best_step=done['best_step'],
            native_validation_loss=done['native_validation_loss'],train_seconds=done['train_seconds'],
            checkpoint_sha256=done['checkpoint_sha256']))
    assert set(records)=={(cell,model) for cell in CELLS for model in MODELS}
    # Verify that data, common training and capacity match within every cell.
    for cell in CELLS:
        group=[records[cell,m]['done'] for m in MODELS];first=group[0]
        for done in group[1:]:
            for key in ('train_files_sha256','test_files_sha256','fit_indices_sha256',
                'holdout_indices_sha256','normalization_sha256','fit_n','holdout_n','test_n'):
                assert done[key]==first[key],(cell,key)
            for key in ('common_training','geometry','rank','reference_parameters','actual_parameters'):
                assert done['resolved'][key]==first['resolved'][key],(cell,key)
    tables.sort(key=lambda r:(CELLS.index(r['cell_key']),MODELS.index(r['model'])))
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    write_json(out/'selection_bootstrap.json',selection_rows)
    report_tables(out,tables)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})
    for logarithmic in (True,False):
        fig,axes=plt.subplots(2,2,figsize=(11.5,8),constrained_layout=True)
        for ax,cell in zip(axes.flat,CELLS):
            for method in MODELS:
                record=records[cell,method];z=record['continuous'];color=COLORS[method]
                mean=z['learned_response'].mean(1);oracle=z['oracle_response'].mean(1)
                ax.plot(z['scales'],mean,color=color,label=NAMES[method]+' learned')
                ax.plot(z['scales'],oracle,color=color,ls='--',alpha=.85,label=NAMES[method]+' exact law')
                idx=record['quality']['response_selection']['selected_index']
                ax.scatter([z['scales'][idx]],[mean[idx]],s=30,color=color,zorder=5)
            true=records[cell,MODELS[0]]['curve']['test_target'][0]
            ax.axhline(true,color='#4b5563',ls=':',lw=1.5,label='True LID')
            if logarithmic:ax.set_xscale('log',base=2)
            ax.set(xlabel='lambda (per-coordinate RMS noise)',ylabel='Mean posterior response R',title=titles(cell))
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle('32 fixed holdout queries: trained response versus continuous posterior\nDots: scale selected on all1000 source-train holdout queries; dashed curves are evaluation-only oracles',fontsize=11)
        stem='mean_response_logx' if logarithmic else 'mean_response_linearx'
        fig.savefig(out/(stem+'.pdf'));fig.savefig(out/(stem+'.png'));plt.close(fig)
    # A single fixed query: all models use the same first holdout ID in a cell.
    fig,axes=plt.subplots(4,3,figsize=(13,13),constrained_layout=True)
    for i,cell in enumerate(CELLS):
        for j,method in enumerate(MODELS):
            ax=axes[i,j];z=records[cell,method]['continuous']
            ax.plot(z['scales'],z['learned_response'][:,0],label='Learned',color=COLORS[method])
            ax.plot(z['scales'],z['oracle_response'][:,0],label='Exact posterior',ls='--',color='#334155')
            ax.axhline(records[cell,method]['curve']['test_target'][0],color='gray',ls=':')
            ax.set(title=titles(cell)+'\n'+NAMES[method]+f"; query ID{z['query_ids'][0]}",
                xlabel='lambda (RMS)',ylabel='Pointwise response R')
            if i==0 and j==0:ax.legend(frameon=False)
    fig.suptitle('One fixed holdout point per cell; linear lambda axis',fontsize=12)
    fig.savefig(out/'point_response_linearx.pdf');fig.savefig(out/'point_response_linearx.png');plt.close(fig)
    # Native losses are shown in separate panels; their units differ by method.
    fig,axes=plt.subplots(4,3,figsize=(12,10),constrained_layout=True)
    for i,cell in enumerate(CELLS):
        for j,method in enumerate(MODELS):
            record=records[cell,method];history=[json.loads(l) for l in (record['path']/'history.jsonl').read_text().splitlines()]
            ax=axes[i,j];steps=np.array([r['step'] for r in history]);loss=np.array([r['validation_loss'] for r in history])
            ax.plot(steps/1000,loss,color=COLORS[method],lw=.8)
            ax.axvline(record['done']['best_step']/1000,color='gray',ls='--')
            ax.set(title=titles(cell)+'\n'+NAMES[method],xlabel='Optimizer updates (thousands)',ylabel='Native holdout loss')
    fig.suptitle('Same update budget and checkpoint policy; native losses have different units',fontsize=12)
    fig.savefig(out/'native_training_curves.pdf');plt.close(fig)
    validation=dict(status='passed',cells=12,distinct_tasks=2,representations=2,
        source_sha256=source,protocol_sha256=rules,all_common_group_checks=True,
        primary_receipts_replayed=True,all_practical_half_unit_checks=all(r['practical_half_unit_mae_check'] for r in tables),
        new_models_half_unit_checks=all(r['practical_half_unit_mae_check'] for r in tables if r['model']!='posterior_rectified_flow'),
        finite_difference_max_gap=max(c['finite_difference_gap'] for r in records.values() for c in r['quality']['trained_derivative_checks']),
        quality_artifact_sha256={str(r['path'].relative_to(ROOT))+'/quality/summary.json':file_sha(r['path']/'quality/summary.json') for r in records.values()},
        artifacts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='verification.json'})
    write_json(out/'verification.json',validation)
    for row in tables:print(json.dumps({k:row[k] for k in ('dataset','model','true_lid','response_selected_lambda','response_test_mae','selected_scale_at_boundary','min_lambda_model_vs_oracle_response_mae')}),flush=True)


if __name__=='__main__':main()
