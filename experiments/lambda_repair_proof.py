"""Build auditable paired tables and standalone scientific figures.

Every score is recomputed from retained pointwise predictions. The table names
the actual trained arm and budget; it never replaces missing runs by proposals.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAIRS=[
 ('VE / Exp','ve_exp_original32_run','ve_exp_spectral_terminal32','Spectral, paired noise, terminal EMA',32000),
 ('VE / Sphere4','ve_sphere_original32','ve_sphere_spectral32','Spectral, paired noise, terminal EMA',32000),
 ('FM velocity VP / Exp','directvp_exp_original32','directvp_exp_spectral_terminal32','Spectral, paired noise, terminal EMA',32000),
 ('VP / Exp','vp_exp_original32','vp_exp_antithetic32','Spectral, paired noise, native cosine',32000),
 ('FM posterior RF / Spiral','postrect_spiral_original32','postrect_spiral_spectral_terminal32_run','Spectral, paired noise, terminal EMA',32000),
 ('VE / Exp','ve_exp_original128','ve_exp_antithetic_terminal128','Spectral, paired noise, terminal EMA',128000),
 ('Bridge / Spaghetti','bridge_spaghetti_original128','bridge_spaghetti_antithetic_terminal128','Spectral, paired noise, terminal EMA',128000),
 ('RF / Spaghetti (coeff.)','rf_spaghetti_original128','rf_spaghetti_antithetic_terminal128','Spectral, paired noise, terminal EMA',128000),
 ('VE / Uniform (coeff.)','ve_uniform_original32','ve_uniform_spectral32','Spectral, paired noise',32000),
 ('FM posterior VP / Exp','postvp_exp_original32','postvp_exp_spectral_terminal32','Spectral, paired noise, terminal EMA',32000),
 ('Bridge / Spaghetti','bridge_spaghetti_original160','bridge_spaghetti_sampled_terminal160','Spectral, paired noise, terminal EMA',160000),
 ('FM posterior RF / Spiral','postrect_spiral_original64','postrect_spiral_sampled_terminal64','Spectral, paired noise, terminal EMA',64000),
]
CURVE_PAIRS=[5,6,4,1]
COLORS={'original':'#a64a36','fixed16':'#60859e','fixed0':'#006d77'}


def read_evaluation(run,probes):
    path=run/f'evaluation_b512_p{probes}'
    if not (run/'complete.json').exists() or not (path/'test_metrics.json').exists():return None
    receipt=json.loads((path/'test_metrics.json').read_text())
    values=np.load(path/'test_selected.npz')
    error=np.abs(values['full'][:,0]-values['target'])
    np.testing.assert_allclose(error.mean(),receipt['test_mae'],rtol=1e-12,atol=1e-12)
    assert len(error)==1000 and np.isfinite(error).all()
    assert values['scales'][0]==receipt['selected_lambda']
    curve=np.load(path/'holdout.npz')
    np.testing.assert_allclose(curve['response']+curve['correction'],curve['full'],rtol=0,atol=0)
    return receipt,error,curve,path


def paired_interval(old,new):
    rng=np.random.default_rng(6371);delta=old-new
    bootstrap=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1)
    lo,hi=np.quantile(bootstrap,[.025,.975]);return float(delta.mean()),float(lo),float(hi)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);rows=[];missing=[];by_pair={}
    for i,(name,old_name,new_name,arm,steps) in enumerate(PAIRS):
        old=read_evaluation(a.runs/old_name,16)
        fixed={n:read_evaluation(a.runs/new_name,n) for n in [16,0]}
        if old is None or any(v is None for v in fixed.values()):missing.append(name+' / '+str(steps));continue
        for run_name in [old_name,new_name]:
            completed=json.loads((a.runs/run_name/'complete.json').read_text())
            assert completed['metrics']['steps_completed']==steps,(run_name,completed['metrics'])
        by_pair[i]=(old,fixed)
        for probes,new in fixed.items():
            improvement,lo,hi=paired_interval(old[1],new[1]);o,n=old[0],new[0]
            rows.append(dict(case=name,updates_per_arm=steps,repair=arm,true_lid=n['target_lid'],
                holdout_n=1000,test_n=1000,old_probes=16,new_probes=probes,
                old_lambda=o['selected_lambda'],new_lambda=n['selected_lambda'],
                old_test_mae=o['test_mae'],new_test_mae=n['test_mae'],
                paired_mae_improvement=improvement,query_ci95_low=lo,query_ci95_high=hi,
                new_holdout_mae=n['holdout_mae'],new_bootstrap_probability_ge16=n['bootstrap_probability_ge16'],
                old_best_step=o['native_best_step'],new_best_step=n['native_best_step'],
                original_run=old_name,repaired_run=new_name,
                original_checkpoint_sha256=o['checkpoint_sha256'],repaired_checkpoint_sha256=n['checkpoint_sha256']))
    if missing and not a.allow_partial:raise RuntimeError('Required completed comparisons missing: '+', '.join(missing))
    if rows:
        with (a.output/'paired_results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    verification=dict(status='partial' if missing else 'complete',paired_cases=len(by_pair),missing=missing,
        source='Recomputed from retained1000-query test_selected.npz after source-train holdout selection',
        bootstrap='2000 paired query resamples at fixed trained weights and selected scales; not retraining uncertainty',
        original_trace='source batch512 Rademacher16',corrected_trace='same16 probes or exact learned affine-span trace',
        full_formula='tr J_q(x,lambda) + ||q(x,lambda)-x||^2/lambda^2',
        selection='unchanged supervised source-train bounded-v2 selector; no lambda cap',
        historical_results_replaced=False,rows=rows)
    (a.output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(a.output/'proof.pdf') as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.suptitle('LID: реализованный фикс и новые парные прогоны',fontsize=18,y=.96)
        caption=('Проверка: устраняет ли общий рецепт обучения катастрофические ошибки исходных моделей?\n'
                 '99k fit; веса выбраны по native loss; λ — по 1k source-train holdout; 1k test открывается после выбора.\n'
                 'q — posterior mean; Full = trace его Jacobian + ‖q(x,λ)−x‖²/λ². MAE = среднее |Full−LID|, меньше лучше.\n'
                 'До: исходная сеть и 16 проб trace. После: общая ковариационная residual-сеть и точный trace.\n'
                 'Один seed, одинаковое число обновлений в паре. Исходные модели заново обучены по исходному коду.')
        fig.text(.05,.91,caption,fontsize=10,linespacing=1.5,va='top')
        show=[r for r in rows if r['new_probes']==0]
        cells=[[r['case'],f"{r['updates_per_arm']//1000}k",f"{r['true_lid']:g}",
                f"{r['old_lambda']:.4g} → {r['new_lambda']:.4g}",
                f"{r['old_test_mae']:.4g} → {r['new_test_mae']:.4g}"] for r in show]
        ax=fig.add_axes([.05,.32,.90,.45]);ax.axis('off')
        if cells:
            table=ax.table(cellText=cells,colLabels=['Модель / данные','Шаги','LID','λ: исходная → фикс','Test MAE: исходная → фикс'],
                cellLoc='left',colLoc='left',loc='upper left',colWidths=[.30,.075,.06,.25,.315])
            table.auto_set_font_size(False);table.set_fontsize(9.5);table.scale(1,1.5)
            for (row,col),cell in table.get_celld().items():
                cell.set_edgecolor('#dddddd')
                if row==0:cell.set_facecolor('#e8f1f2');cell.set_text_props(weight='bold')
        fig.text(.05,.24,'Реальные изменения: нормирование внутри train-only ковариационной оболочки;\n'
                 'аналитическое гауссовское восстановление вне неё; residual-бэкбоун; точный trace.\n'
                 'В отмеченных прогонах: пары Z и −Z и последние 8k шагов с cosine/EMA.\n'
                 'Полная спецификация каждой пары, хеши весов и парные интервалы: paired_results.csv.',linespacing=1.7)
        if missing:fig.text(.05,.08,'ПРОМЕЖУТОЧНЫЙ АРТЕФАКТ: ещё не завершены '+', '.join(missing),color='#a64a36',fontsize=8)
        fig.text(.05,.035,'Сравнение носит диагностический характер: это не полный повтор бенчмарка и не оценка разброса по переобучениям.',fontsize=9)
        pdf.savefig(fig);fig.savefig(a.output/'paired_results.png',dpi=160);plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(11.7,8.3));fig.suptitle('Почему изменилась выбранная λ: сохранённые holdout-кривые',fontsize=16)
        for ax,index in zip(axes.flat,CURVE_PAIRS):
            name,*_=PAIRS[index];ax.set_title(name)
            if index not in by_pair:ax.text(.5,.5,'Прогон ещё не завершён',transform=ax.transAxes,ha='center');continue
            old,fixed=by_pair[index]
            for label,item,color in [('Исходная модель, P16',old,COLORS['original']),
                                      ('Новая модель, P16',fixed[16],COLORS['fixed16']),
                                      ('Новая модель, точный trace',fixed[0],COLORS['fixed0'])]:
                rec,_,v,_=item;err=np.abs(v['full']-rec['target_lid']).mean(0)
                ax.plot(v['scales'],err,label=label,color=color,lw=1.7)
                col=np.argmin(abs(v['scales']-rec['selected_lambda']))
                ax.scatter([v['scales'][col]],[err[col]],color=color,s=65,marker='*',zorder=4)
            reference_name={6:'bridge_spaghetti_original32_run',4:'postrect_spiral_original32'}.get(index)
            if reference_name:
                oracle_path=a.runs/reference_name/'oracle_selection/arrays.npz'
                if oracle_path.exists():
                    oracle_values=np.load(oracle_path)
                    ax.plot(oracle_values['scales'],abs(oracle_values['full_exact']-oracle_values['target'][:,None]).mean(0),
                            '--',color='#252525',lw=1.3,label='Точный posterior и trace')
            ax.set(xscale='log',yscale='log',xlabel='λ в исходных единицах нормировки',ylabel='Holdout MAE LID (меньше лучше)')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout(rect=[0,.085,1,.95]);fig.text(.045,.035,
            'Каждая кривая: те же 1 000 holdout-наблюдений; P16 — 16 проб Радемахера. Звезда — bounded-v2 выбор.\n'
            'Разница красной и синей кривых — изменение модели; синей и зелёной — устранение шума trace. Граница λ не сужалась.',fontsize=9)
        pdf.savefig(fig);fig.savefig(a.output/'holdout_curves.png',dpi=160);plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(11.7,8.3));fig.suptitle('Проверка обучения отдельно от скалярной ошибки LID',fontsize=16)
        for ax,index in zip(axes.flat,CURVE_PAIRS):
            name,old,new,_,_=PAIRS[index];ax.set_title(name)
            for run,label,color in [(old,'Исходная модель',COLORS['original']),(new,'Новая модель',COLORS['fixed0'])]:
                path=a.runs/run/'denoising/summary.json'
                if not path.exists():continue
                j=json.loads(path.read_text());values=j['rows'];s=np.array([x['lambda_value'] for x in values]);risk=np.array([x['mean_risk'] for x in values])
                ax.plot(s,risk,'o-',ms=3,label=label,color=color)
            ax.set(xscale='log',yscale='log',xlabel='λ в исходных единицах нормировки',ylabel='Денойзинг: E‖q(X+λZ)−X‖² / λ²')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout(rect=[0,.09,1,.95]);fig.text(.045,.03,
            '128 фиксированных holdout-точек × 8 одинаковых шумов на точку; сумма по координатам, меньше лучше.\n'
            'Эта метрика проверяет восстановление зашумлённых данных и не использует производные или метки LID.\n'
            'Успех на выбранной конечной λ не доказывает точность сети при λ → 0; все малые масштабы показаны явно.',fontsize=9)
        pdf.savefig(fig);fig.savefig(a.output/'denoising.png',dpi=160);plt.close(fig)

        oracle=a.runs/'bridge_spaghetti_original32_run/oracle_selection'
        if (oracle/'summary.json').exists():
            summary=json.loads((oracle/'summary.json').read_text())
            measured={row['estimator']:row for row in summary['results']}
            fig=plt.figure(figsize=(11.7,8.3));fig.suptitle('Контроль селектора: идеальная модель, меняется только trace',fontsize=17,y=.95)
            fig.text(.06,.82,'Spaghetti, все 1 000 holdout-точек. Непрерывный posterior вычислен по закону генератора.\n'
                'Модель не обучается. Сетка λ, bounded-v2 выбор и целевой LID=1 остаются прежними.',linespacing=1.8,fontsize=12)
            lines=[f"{label}: λ={measured[key]['selected_lambda']:.5g}; MAE={measured[key]['holdout_mae']:.5g}."
                   for key,label in [('full_exact','Точный trace'),('full_p16','16 исходных случайных проб'),
                                     ('full_p64','64 случайные пробы')]]
            factor=measured['full_p16']['selected_lambda']/measured['full_exact']['selected_lambda']
            for k,line in enumerate(lines):fig.text(.085,.66-.095*k,line,fontsize=17)
            fig.text(.06,.29,f'Вывод этого контроля: шум trace способен сдвинуть выбранный масштаб в {factor:g} раз\n'
                'даже при точном posterior. Здесь P16 не выбирает λ≥16: этот контроль не объясняет\n'
                'все случаи λ=64 и не отменяет обнаруженные ошибки обученных полей.\n\n'
                'Числа и полный массив: bridge_spaghetti_original32_run/oracle_selection/.\n'
                'Точный trace в патче использует глобальный affine-rank модели, а не истинный LID.',linespacing=1.8,fontsize=11)
            pdf.savefig(fig);plt.close(fig)

        spiral=a.runs/'postrect_spiral_original32/oracle_selection'
        if (spiral/'arrays.npz').exists():
            v=np.load(spiral/'arrays.npz');err=abs(v['full_exact']-v['target'][:,None]).mean(0);best=int(err.argmin())
            fig,axes=plt.subplots(1,2,figsize=(11.7,8.3));fig.suptitle('Spiral: большая λ возможна и у точного posterior',fontsize=17,y=.95)
            fig.text(.06,.82,'Контроль без обучения: интегрирование непрерывного закона генератора, все 1 000 holdout-точек.\n'
                'Spiral: (sin(t²)/t, cos(t²)/t), t равномерно на [1,100]; истинный LID=1.\n'
                'R — trace производной posterior; C — квадрат смещения posterior / λ²; Full=R+C.',linespacing=1.7,fontsize=10)
            axes[0].plot(v['scales'],err,color='#252525');axes[0].scatter(v['scales'][best],err[best],marker='*',s=100,color='#006d77')
            axes[0].set(xscale='log',xlabel='λ в исходных единицах',ylabel='MAE точного Full к LID=1')
            axes[1].plot(v['scales'],v['response_exact'].mean(0),label='Среднее R')
            axes[1].plot(v['scales'],v['correction'].mean(0),label='Среднее C')
            axes[1].axhline(1,color='black',ls='--',lw=1,label='LID=1')
            axes[1].set(xscale='log',xlabel='λ в исходных единицах',ylabel='Среднее по 1 000 точек');axes[1].legend()
            for ax in axes:ax.grid(alpha=.2)
            fig.tight_layout(rect=[.03,.27,.98,.73])
            fig.text(.06,.15,f'Глобальный минимум на всех 29 масштабах: λ={v["scales"][best]:.5g}, MAE={err[best]:.5g}.\n'
                f'На нижней границе λ=1/256: MAE={err[0]:.5g}. Сетка ещё не разрешает все тесные витки.\n'
                'Поэтому λ≈5,66 здесь нельзя автоматически считать багом. Исправление сети не доказывает\n'
                'точность при λ→0; для такого утверждения нужны более мелкие масштабы и достаточная плотность данных.',fontsize=11,linespacing=1.8)
            sensitivity_path=a.runs/'oracle_selector_sensitivity.json'
            if sensitivity_path.exists():
                rows_sensitivity=json.loads(sensitivity_path.read_text())
                item=next(row for row in rows_sensitivity if row['run']=='postrect_spiral_original32' and row['readout']=='full_exact')
                detail='; '.join(f"λ={row['lambda_value']:.4g}: {row['n']}/500" for row in item['bootstrap'])
                fig.text(.06,.035,'Bootstrap глобального выбора: '+detail+'. Эти два минимума статистически не разделены.',fontsize=9)
            pdf.savefig(fig);fig.savefig(a.output/'spiral_oracle.png',dpi=160);plt.close(fig)

        nf_rows=[]
        nf_runs=[('Исходная','nf_exp_original32'),('Ковариация + EMA','nf_exp_cov_terminal32'),
                 ('Ковариация + EMA + ω≤1','nf_exp_lowfreq_terminal32'),('Исходная','nf_exp_original128'),
                 ('Ковариация, постоянный LR','nf_exp_cov128'),('Ковариация + EMA + ω≤1','nf_exp_lowfreq_terminal128')]
        for name,run in nf_runs:
            for suffix,readout in [('float32','OLS5'),('float64_exact','Точная производная')]:
                path=a.runs/run/f'evaluation_nf_{suffix}'
                if not (path/'test_metrics.json').exists():continue
                receipt=json.loads((path/'test_metrics.json').read_text());arrays=np.load(path/'test_selected.npz')
                np.testing.assert_allclose(abs(arrays['full'][:,0]-arrays['target']).mean(),receipt['test_mae'],rtol=0,atol=1e-10)
                nf_rows.append(dict(state=name,run=run,dataset=receipt['dataset'],target_lid=receipt['target_lid'],readout=readout,steps=receipt['total_steps'],
                    lambda_value=receipt['selected_lambda'],holdout_mae=receipt['holdout_mae'],test_mae=receipt['test_mae'],
                    condition_max_frequency=receipt['condition_max_frequency'],checkpoint_sha256=receipt['checkpoint_sha256']))
        if nf_rows:
            with (a.output/'nf_results.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(nf_rows[0]));writer.writeheader();writer.writerows(nf_rows)
            fig=plt.figure(figsize=(11.7,8.3));fig.suptitle('NF: исправление зависимости плотности от масштаба',fontsize=17,y=.95)
            fig.text(.05,.82,'Exp, коэффициенты, LID=2; 99k fit / 1k holdout / 1k test; один seed. Веса выбираются по native NLL.\n'
                'Оценка: D + производная log p(x,λ) по log λ при фиксированном x.\n'
                'OLS5 — линейная регрессия по пяти log-шкалам с шагом 0,05; точная производная — autograd.\n'
                'Для каждого readout λ отдельно выбрана прежним bounded-v2 правилом на holdout.',fontsize=10,linespacing=1.6)
            cells=[[r['state'],str(r['steps']//1000)+'k',r['readout'],f"{r['lambda_value']:.4g}",f"{r['test_mae']:.4g}"] for r in nf_rows]
            ax=fig.add_axes([.045,.31,.91,.43]);ax.axis('off')
            table=ax.table(cellText=cells,colLabels=['Обученная модель','Шаги','Readout','λ','Test MAE ↓'],
                           colWidths=[.35,.07,.25,.14,.19],loc='upper left',cellLoc='left',colLoc='left')
            table.auto_set_font_size(False);table.set_fontsize(9.5);table.scale(1,1.45)
            for (row,col),cell in table.get_celld().items():
                cell.set_edgecolor('#ddd')
                if row==0:cell.set_facecolor('#e8f1f2');cell.set_text_props(weight='bold')
            fig.text(.05,.16,'У исходного NF на λ=1/16, 64 одинаковых holdout-точках: OLS5 MAE=0,233,\n'
                'точная производная MAE=24,529. Это осцилляция выученной плотности; OLS5 её сглаживает.\n'
                'Новая граница частот кодирования log λ: 100 → 1, с тем же условным NLL.\n'
                'Сглаживание зависимости не следует выдавать за доказательство точного Gaussian heat flow.',fontsize=10,linespacing=1.7)
            pdf.savefig(fig);fig.savefig(a.output/'nf_derivatives.png',dpi=160);plt.close(fig)

        spag_nf=[]
        for name,run in [('Исходная','nf_spaghetti_original32'),('Общий гладкий NF','nf_spaghetti_smooth_terminal32')]:
            for suffix,readout in [('float32','OLS5'),('float64_exact','Точная производная')]:
                path=a.runs/run/f'evaluation_nf_{suffix}'
                if not (path/'test_metrics.json').exists():continue
                receipt=json.loads((path/'test_metrics.json').read_text());arrays=np.load(path/'test_selected.npz')
                np.testing.assert_allclose(abs(arrays['full'][:,0]-arrays['target']).mean(),receipt['test_mae'],rtol=0,atol=1e-10)
                spag_nf.append(dict(state=name,run=run,dataset=receipt['dataset'],target_lid=receipt['target_lid'],readout=readout,
                    steps=receipt['total_steps'],lambda_value=receipt['selected_lambda'],holdout_mae=receipt['holdout_mae'],
                    test_mae=receipt['test_mae'],condition_max_frequency=receipt['condition_max_frequency'],
                    checkpoint_sha256=receipt['checkpoint_sha256']))
        if len(spag_nf)==4:
            nf_rows+=spag_nf
            with (a.output/'nf_results.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(nf_rows[0]));writer.writeheader();writer.writerows(nf_rows)
            fig,axes=plt.subplots(1,2,figsize=(11.7,8.3));fig.suptitle('NF / Spaghetti: тот же рецепт на втором outlier',fontsize=17,y=.95)
            fig.text(.055,.83,'Коэффициенты, LID=1; 32k обновлений в обеих моделях, 99k fit / 1k holdout / 1k test.\n'
                'Веса выбираются по нативному NLL; для каждого readout λ затем выбирается на holdout.\n'
                'Изменены ковариационное нормирование, максимальная частота log λ (100→1) и terminal/EMA.',fontsize=10,linespacing=1.7)
            for k,(suffix,title) in enumerate([('float32','OLS5'),('float64_exact','Точная производная')]):
                for run,label,color in [('nf_spaghetti_original32','Исходная',COLORS['original']),
                                        ('nf_spaghetti_smooth_terminal32','Общий гладкий NF',COLORS['fixed0'])]:
                    path=a.runs/run/f'evaluation_nf_{suffix}';v=np.load(path/'holdout.npz')
                    rec=json.loads((path/'test_metrics.json').read_text());err=abs(v['full']-v['target'][:,None]).mean(0)
                    axes[k].plot(v['scales'],err,label=label,color=color)
                    j=np.argmin(abs(v['scales']-rec['selected_lambda']));axes[k].scatter(v['scales'][j],err[j],marker='*',s=70,color=color)
                axes[k].set(title=title,xscale='log',yscale='log',xlabel='λ в исходных единицах',ylabel='Holdout MAE к LID=1')
                axes[k].grid(alpha=.2);axes[k].legend(fontsize=9)
            fig.tight_layout(rect=[.03,.27,.98,.74])
            for k,row in enumerate(spag_nf):
                fig.text(.065,.205-k*.038,f"{row['state']}, {row['readout']}: λ={row['lambda_value']:.5g}, testMAE={row['test_mae']:.5g}.",fontsize=11)
            fig.text(.055,.035,'Один seed; это независимая проверка общей настройки на второй геометрии, не полный повтор NF-бенчмарка.',fontsize=9)
            pdf.savefig(fig);fig.savefig(a.output/'nf_spaghetti.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in verification.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
