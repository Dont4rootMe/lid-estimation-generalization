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
BASE=ROOT/'artifacts/non_gaussian_ambient_20260911'
HERE=BASE/'report'
MODELS=['t_flowmatching','pfgmpp']
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
    return ('Exp' if dataset=='e6_exp_pca' else 'Spiral')+' / '+('coefficients, N=30' if rep=='coefficients' else 'PCA render, N=784')+' (full ambient model)'



def latex_table(headers,rows,alignment):
    return '\n'.join([r'\begin{center}\small',r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{'+alignment+'}',r'\toprule',
        ' & '.join(headers)+r' \\',r'\midrule',
        *[' & '.join(map(str,row))+r' \\' for row in rows],
        r'\bottomrule',r'\end{tabular}}',r'\end{center}',''])


def report_tables(out,tables):
    model_names={'t_flowmatching':'t-Flow','pfgmpp':'PFGM++'}
    def dataset(row):
        return ('Exp' if '/e6_' in row['cell_key'] else 'Spiral')+' / '+(
            'коэфф.' if row['cell_key'].endswith('/coefficients') else 'рендер')
    scores=[];oracles=[];denoising=[];probes=[];generation=[]
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
            f"{row['selected_lambda_initial_denoising_risk']:.3f}",
            f"[{row['denoising_paired_difference_ci95_low']:.3f}, {row['denoising_paired_difference_ci95_high']:.3f}]"])
        probes.append(prefix+['exact' if row['ambient_dim']==30 else '64 probes',
            f"{row['selected_probe_repeat_mae_min']:.3f}--{row['selected_probe_repeat_mae_max']:.3f}",
            f"{row['selected_benchmark_vs_exact_trace_mae']:.4f}",
            f"{row['float32_trace_max_gap']:.2g}"])
        generation.append(prefix+[f"{row['generated_total_variance_ratio']:.3f}",
            f"{row['generated_normal_distance_ratio']:.4f}",
            f"{row['generated_to_fit_distance_ratio']:.4f} / {row['test_to_fit_distance_ratio']:.4f}",
            f"{row['generation_refinement_rms']:.2g}"])
    (out/'score_table.tex').write_text(latex_table(
        ['Данные','Модель','$d$',r'$\lambda$','Test MAE [95\\% CI]','Kneedle MAE'],scores,'llrrlr'))
    (out/'oracle_table.tex').write_text(latex_table(
        ['Данные','Модель',r'$\R_{\min}$ сеть',r'$\R_{\min}$ эталон',r'$|R_{\rm model}-R_{\rm exact}|$'],oracles,'llrrr'))
    (out/'denoising_table.tex').write_text(latex_table(
        ['Данные','Модель',r'$\lambda$','Обученный posterior','Начальный preconditioner',r'95\% CI разности'],denoising,'llrrrl'))
    (out/'probe_table.tex').write_text(latex_table(
        ['Данные','Модель','Benchmark trace','Test MAE range',r'$|\widehat R-R_{\rm exact\ trace}|$',r'FP32 gap'],probes,'llllrr'))
    (out/'generation_table.tex').write_text(latex_table(
        ['Данные','Модель','Variance ratio','Normal ratio','Fit distance gen / test','ODE refinement'],generation,'llrrlr'))
    new=[r for r in tables if r['model']!='posterior_rectified_flow']
    passed=sum(r['practical_half_unit_mae_check'] for r in new)
    boundary=sum(r['selected_scale_at_boundary'] for r in new)
    (out/'outcome.tex').write_text(
        f'Практический порог supervised test MAE$<0.5$ прошли {passed} из 8 новых моделей/ячеек. '
        f'Граничный supervised масштаб выбран в {boundary} из 8 случаев. '
        'Эти числа описывают конечную шкалу и данный бюджет. '
        'Точность при малом шуме оценивается следующей проверкой; '
        'она не следует из прохождения этого порога.\n')
def main():
    out=HERE/'results';out.mkdir(parents=True,exist_ok=True)
    records={};tables=[];selection_rows=[];source=source_identity();rules=digest(protocol())
    for path in sorted((BASE/'full_budget').glob('*/complete.json')):
        done=verify_measurements(path);quality_path=path.parent/'quality/summary.json'
        quality=json.loads(quality_path.read_text())
        generation=json.loads((path.parent/'generation_quality/summary.json').read_text())
        assert quality['status']=='complete' and quality['checkpoint_sha256']==done['checkpoint_sha256']
        assert generation['status']=='complete' and generation['checkpoint_sha256']==done['checkpoint_sha256']
        assert done['source_sha256']==source and done['protocol_sha256']==rules
        assert done['kind']=='benchmark' and done['steps_completed']==128000
        assert done['test_n']==1000 and done['fit_n']==99000 and done['holdout_n']==1000
        with np.load(path.parent/'quality/response_curves.npz') as z:
            curve={k:z[k] for k in z.files}
        frozen=json.loads((path.parent/'quality/response_selection.json').read_text())
        selection=quality['response_selection'];idx=selection['selected_index']
        errors=abs(curve['test'][:,idx]-curve['test_target'])
        assert np.all(curve['test_target']==curve['test_target'][0]),'this report requires constant-LID Exp/Spiral cells'
        assert np.isclose(errors.mean(),quality['response_metrics']['mae'],rtol=0,atol=1e-12)
        bootstrap=selector_bootstrap(curve['holdout'],curve['holdout_target'],curve['scales'])
        selection_rows.append(dict(variant=done['variant'],cell=done['cell_key'],
            selection_bootstrap=bootstrap,rule='2000 source-train query bootstraps, fixed weights; original selection unchanged'))
        with np.load(path.parent/'quality/continuous_comparison.npz') as z:continuous={k:z[k] for k in z.files}
        record=dict(done=done,quality=quality,generation=generation,curve=curve,continuous=continuous,path=path.parent)
        key=(done['cell_key'],done['variant']);assert key not in records;records[key]=record
        at_min=quality['continuous_law'][0];at_selected=quality['continuous_law'][idx]
        risk=quality['denoising'][idx]
        cell_meta=done['cell'];ambient=done['resolved']['geometry']['ambient_dim'];rank=done['resolved']['rank']
        g=json.loads((BASE/'population'/f"{cell_meta['dataset']}__{cell_meta['representation']}"/'geometry.json').read_text())
        assert done['config']['field_projection_rank'] is None
        initial_curve=ambient/(1+curve['scales']**2)
        initial_errors=abs(initial_curve-float(curve['test_target'][0]))
        initial_index=int(np.argmin(initial_errors))
        raw_rms=selection['selected_lambda']*g['normalization_rms']
        tables.append(dict(dataset=titles(done['cell_key']),cell_key=done['cell_key'],model=done['variant'],
            true_lid=float(curve['test_target'][0]),test_n=1000,
            primary_readout=done['primary_readout'],primary_selected_lambda=done['selected_lambda'],
            primary_test_mae=done['metrics'][done['primary_readout']]['mae'],
            response_selected_lambda=selection['selected_lambda'],response_test_mae=float(errors.mean()),
            selected_raw_ambient_rms=raw_rms,ambient_dim=ambient,
            zero_head_selected_lambda=float(curve['scales'][initial_index]),
            zero_head_selected_mae=float(initial_errors[initial_index]),
            zero_head_mae_at_trained_lambda=float(initial_errors[idx]),
            selected_zero_head_vs_oracle_response_mae=float(abs(initial_curve[idx]-continuous['oracle_response'][idx]).mean()),
            selected_benchmark_vs_exact_trace_mae=at_selected['benchmark_vs_exact_trace_mae'],
            float32_trace_precision_passed=quality['float32_trace_precision_passed'],
            float32_trace_max_gap=max(c['float32_vs_float64_trace_gap'] for c in quality['trained_derivative_checks']),
            finite_difference_max_gap=max(c['finite_difference_gap'] for c in quality['trained_derivative_checks']),
            generated_total_variance_ratio=generation['metrics']['total_variance_ratio'],
            generated_normal_distance_ratio=generation['metrics']['generated_normal_distance_over_data_rms_norm'],
            generated_to_fit_distance_ratio=generation['metrics']['generated_to_fit_distance_over_data_rms_norm'],
            test_to_fit_distance_ratio=generation['metrics']['test_to_fit_distance_over_data_rms_norm'],
            generated_energy_distance_ratio=generation['metrics']['energy_distance_over_data_rms_norm'],
            generation_refinement_rms=generation['refinement_rms_per_coordinate'],
            generation_refinement_passed=generation['refinement_gate_passed'],
            selected_probe_repeat_mae_min=min([float(errors.mean())]+[x['mae'] for x in quality['probe_repeats']]),
            selected_probe_repeat_mae_max=max([float(errors.mean())]+[x['mae'] for x in quality['probe_repeats']]),
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
            denoising_paired_difference_ci95_low=risk['paired_risk_difference_ci95'][0],
            denoising_paired_difference_ci95_high=risk['paired_risk_difference_ci95'][1],
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
    zero_rows=[]
    for cell in CELLS:
        row=next(r for r in tables if r['cell_key']==cell)
        zero_rows.append([('Exp' if '/e6_' in cell else 'Spiral')+' / '+('коэфф.' if cell.endswith('/coefficients') else 'пиксели'),
            row['ambient_dim'],int(row['true_lid']),f"{row['zero_head_selected_lambda']:.5g}",f"{row['zero_head_selected_mae']:.3f}"])
    (out/'zero_head_table.tex').write_text(latex_table(['Данные','$N$','$d$',r'$\lambda_0$','MAE без nonlinear correction'],zero_rows,'lrrrr'))
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})
    for logarithmic in (True,False):
        fig,axes=plt.subplots(2,2,figsize=(11.5,8),constrained_layout=True)
        for ax,cell in zip(axes.flat,CELLS):
            for method in MODELS:
                record=records[cell,method];z=record['continuous'];color=COLORS[method]
                mean=z['learned_response'].mean(1);oracle=z['oracle_response'].mean(1)
                ax.plot(z['scales'],mean,color=color,label=NAMES[method]+' learned (full exact trace)')
                ax.plot(z['scales'],oracle,color=color,ls='--',alpha=.85,label=NAMES[method]+' exact law')
                idx=record['quality']['response_selection']['selected_index']
                ax.scatter([z['scales'][idx]],[mean[idx]],s=30,color=color,zorder=5)
            true=records[cell,MODELS[0]]['curve']['test_target'][0]
            ax.axhline(true,color='#4b5563',ls=':',lw=1.5,label='True LID')
            if logarithmic:ax.set_xscale('log',base=2)
            else:ax.set_xlim(0,64)
            ax.set(xlabel='lambda (per-coordinate RMS noise)',ylabel='Mean posterior response R',title=titles(cell))
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle('32 fixed holdout queries: trained response versus continuous posterior\nDots: scale selected on all 1000 source-train holdout queries; dashed curves are evaluation-only oracles',fontsize=11)
        stem='mean_response_logx' if logarithmic else 'mean_response_linearx'
        fig.savefig(out/(stem+'.pdf'));fig.savefig(out/(stem+'.png'));plt.close(fig)
    # A single fixed query: all models use the same first holdout ID in a cell.
    fig,axes=plt.subplots(4,2,figsize=(11,13),constrained_layout=True)
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
    fig,axes=plt.subplots(4,2,figsize=(11,10),constrained_layout=True)
    for i,cell in enumerate(CELLS):
        for j,method in enumerate(MODELS):
            record=records[cell,method];history=[json.loads(l) for l in (record['path']/'history.jsonl').read_text().splitlines()]
            ax=axes[i,j];steps=np.array([r['step'] for r in history]);loss=np.array([r['validation_loss'] for r in history])
            ax.plot(steps/1000,loss,color=COLORS[method],lw=.8)
            ax.axvline(record['done']['best_step']/1000,color='gray',ls='--')
            ax.set(title=titles(cell)+'\n'+NAMES[method],xlabel='Optimizer updates (thousands)',ylabel='Native holdout loss')
    fig.suptitle('Same update budget and checkpoint policy; native losses have different units',fontsize=12)
    fig.savefig(out/'native_training_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(11,12),constrained_layout=True)
    for i,cell in enumerate(CELLS):
        for j,method in enumerate(MODELS):
            record=records[cell,method];ax=axes[i,j]
            with np.load(record['path']/'generation_quality/samples.npz') as z:
                generated=z['coordinates'];real=z['test_coordinates']
            if '/e6_' in cell:
                gx,gy=generated[:,0]+4,np.linalg.norm(generated[:,1:],axis=1)
                tx,ty=real[:,0]+4,np.linalg.norm(real[:,1:],axis=1)
                ax.set(xlabel='Exp axial coordinate u',ylabel='Exp radius')
            else:
                gx,gy=generated.T;tx,ty=real.T
                ax.set(xlabel='Spiral coordinate 1',ylabel='Spiral coordinate 2')
            ax.scatter(tx,ty,s=6,alpha=.3,color='#64748b',label='Real test (1000)')
            ax.scatter(gx,gy,s=7,alpha=.6,color=COLORS[method],label='Native samples (512)')
            normal=record['generation']['metrics']['generated_normal_distance_over_data_rms_norm']
            ax.set_title(titles(cell)+'\n'+NAMES[method]+f'; full normal RMS / data RMS = {normal:.4f}',fontsize=8)
            if i==0 and j==0:ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Generator coordinates are displayed only; sampling retains all ambient coordinates',fontsize=11)
    fig.savefig(out/'generation_coordinates.pdf');fig.savefig(out/'generation_coordinates.png');plt.close(fig)
    fig,axes=plt.subplots(4,8,figsize=(11.5,6.8),constrained_layout=True)
    for i,(cell,method) in enumerate((cell,m) for cell in CELLS[2:] for m in MODELS):
        record=records[cell,method]
        with np.load(record['path']/'generation_quality/samples.npz') as z:
            raw=z['raw'];real=z['test_raw']
        lower,upper=float(real.min()),float(real.max())
        for j,ax in enumerate(axes[i]):
            pixels=real[j] if j<4 else raw[j-4]
            ax.imshow(pixels.reshape(28,28),cmap='gray',vmin=lower,vmax=upper)
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(('Real ' if j<4 else 'Generated ')+str(j%4+1),fontsize=8)
            if j==0:ax.set_ylabel(('Exp' if '/e6_' in cell else 'Spiral')+'\n'+NAMES[method],fontsize=8)
    fig.suptitle('Full28x28 images: fixed real-data intensity range per dataset; no per-image contrast normalization',fontsize=10)
    fig.savefig(out/'generation_images.pdf');fig.savefig(out/'generation_images.png');plt.close(fig)
    validation=dict(status=('passed' if all(r['float32_trace_precision_passed'] and r['generation_refinement_passed'] for r in tables) else 'numerical_precision_failed'),cells=8,distinct_tasks=2,representations=2,
        source_sha256=source,protocol_sha256=rules,all_common_group_checks=True,
        primary_receipts_replayed=True,all_practical_half_unit_checks=all(r['practical_half_unit_mae_check'] for r in tables),
        new_models_half_unit_checks=all(r['practical_half_unit_mae_check'] for r in tables if r['model']!='posterior_rectified_flow'),
        float32_trace_max_gap=max(c['float32_vs_float64_trace_gap'] for r in records.values() for c in r['quality']['trained_derivative_checks']),
        finite_difference_max_gap=max(c['finite_difference_gap'] for r in records.values() for c in r['quality']['trained_derivative_checks']),
        generation_refinement_all_passed=all(r['generation_refinement_passed'] for r in tables),
        generation_quality_sha256={str(r['path'].relative_to(ROOT))+'/generation_quality/summary.json':file_sha(r['path']/'generation_quality/summary.json') for r in records.values()},
        quality_artifact_sha256={str(r['path'].relative_to(ROOT))+'/quality/summary.json':file_sha(r['path']/'quality/summary.json') for r in records.values()},
        artifacts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='verification.json'})
    write_json(out/'verification.json',validation)
    for row in tables:print(json.dumps({k:row[k] for k in ('dataset','model','true_lid','response_selected_lambda','response_test_mae','selected_scale_at_boundary','min_lambda_model_vs_oracle_response_mae')}),flush=True)


if __name__=='__main__':main()
