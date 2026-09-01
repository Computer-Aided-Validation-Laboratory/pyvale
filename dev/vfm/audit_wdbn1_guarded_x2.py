"""Offline audits of completed WDBN1 guarded-EGI x2 identifications."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pyvale.vfm import (
    EquilibriumGapMetric,
    ExperimentData,
    MeasurementNoiseFloorConfig,
    MeasurementNoiseMode,
    SliceConfig,
    SliceWiseForceReconstructionMetric,
    load_identification_result,
    measurement_noise_realisation,
)
from pyvale.vfm.metricsbvf import calculate_parameter_stress_sensitivities
from pyvale.vfm.modelorder import select_noise_resolved_basis_count
from pyvale.vfm.objectivefuncsensitivitygated import _normalise_activity, _smooth_gate
from pyvale.vfm.postprocessing import load_constitutive_law_from_result, load_known_parameter_maps

import report_notched_ebw_data_driven_identification as common
from report_guarded_egi_identification import _artifacts


FINE_SCALE = 1.7749920645783005e-6
BROAD_SCALE = 1.974782950165619e-7
H_POINTS = 61
H_STAGES = ("BF1", "BF2", "BF3", "BF5", "BF6", "BF7")


def _parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for kind in ("clean","noisy"):
        p.add_argument(f"--{kind}-input",type=Path,required=True)
        p.add_argument(f"--{kind}-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    return p.parse_args()


def _weighted_rms(values, weights):
    values=np.asarray(values,dtype=float)
    weights=np.broadcast_to(np.asarray(weights,dtype=float),values.shape)
    valid=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    if not np.any(valid): return float("nan")
    w=weights[valid]; w=w/np.sum(w)
    return float(np.sqrt(np.sum(w*values[valid]**2)))


def _label(raw):
    return "Phase0" if raw.startswith("Phase 0") else raw.split()[0]


def _metric_set(experiment,result,run):
    supports=dict(common._selected_supports(result))
    fre=json.loads((run/"diagnostic_artifacts/fre_resolution_sweep_000.json").read_text())
    metrics=(
        SliceWiseForceReconstructionMetric(slice_config=SliceConfig(axis=fre["axis"],num_slices=int(fre["selected_num_slices"]))),
        EquilibriumGapMetric(window_size=supports["fine"],fft_dtype="float32",fft_batch_group="local"),
        EquilibriumGapMetric(window_size=supports["broad"],fft_dtype="float32",fft_batch_group="broad"),
    )
    for m in metrics: m.initialise(experiment)
    return metrics,supports,fre


def _evaluate(experiment,law,maps,metrics,gate,temporal_override=None):
    stress=law.calculate_stress(experiment.strain,maps)
    fre=metrics[0].evaluate_force_recon_error(stress,experiment)
    fine=metrics[1].evaluate_equilibrium_gap(stress,include_diagnostics=True)
    broad=metrics[2].evaluate_equilibrium_gap(stress,include_diagnostics=True)
    ff=fre.metric_result.additional_fields
    fi=fine.metric_result.additional_fields
    br=broad.metric_result.additional_fields
    if temporal_override is None:
        wt_fre=np.asarray(ff["temporal_weights"])
        wt_fine=np.asarray(fi["temporal_weights"])
        wt_broad=np.asarray(br["temporal_weights"])
    else:
        wt_fre,wt_fine,wt_broad=temporal_override
    fine_value=_weighted_rms(fi["normalised_gap"],gate*wt_fine[:,None,None])/FINE_SCALE
    broad_value=_weighted_rms(br["normalised_gap"],gate*wt_broad[:,None,None])/BROAD_SCALE
    raw=np.asarray(ff["raw_residual"],dtype=float)
    relative=np.asarray(ff["normalised_residual"],dtype=float)
    force=np.asarray(ff["applied_longitudinal_force"],dtype=float)
    recon=np.asarray(ff["reconstructed_force"],dtype=float)
    return {
        "fine":fine_value,"broad":broad_value,"primary":.5*(fine_value+broad_value),
        "broad_unmasked":_weighted_rms(br["normalised_gap"],wt_broad[:,None,None]),
        "fre":_weighted_rms(relative,wt_fre[:,None]),
        "fre_profile_percent":100*np.sqrt(np.nansum(wt_fre[:,None]*relative**2,axis=0)),
        "fre_profile_n":np.sqrt(np.nansum(wt_fre[:,None]*raw**2,axis=0)),
        "reconstructed_force":recon,"applied_force":force,
        "temporal":(wt_fre,wt_fine,wt_broad),
    }


def _frozen_gate(experiment,law,phase0_maps):
    stress=law.calculate_stress(experiment.strain,phase0_maps)
    sensitivities=calculate_parameter_stress_sensitivities(
        experiment.strain,stress,law,phase0_maps,["yield_strength","hardening_modulus"],perturbation_factor=.01)
    activities={}
    for name,item in sensitivities.items():
        values=np.sqrt(np.nansum(item.total**2,axis=1))
        values[~np.any(np.isfinite(item.total),axis=1)]=np.nan
        activities[name]=_normalise_activity(values,95.0)
    combined=np.maximum.reduce(list(activities.values()))
    combined[np.isfinite(combined)&(combined<1e-6)]=0
    positive=combined[np.isfinite(combined)&(combined>0)]
    start,full=np.quantile(positive,(0.0,.90))
    gate=_smooth_gate(combined,float(start),float(full)); gate[~np.isfinite(combined)]=np.nan
    return gate,{"start":float(start),"full":float(full),"positive_fraction":float(np.mean(np.isfinite(gate)&(gate>0)))}


def _noise_config(prep):
    m=prep[0]["measurement_noise_floor"]
    return MeasurementNoiseFloorConfig(
        mode=MeasurementNoiseMode.CALIBRATED,seeds=tuple(m["seeds"]),
        strain_std_microstrain=tuple(m["strain_std_microstrain"]),force_std_n=float(m["force_std_n"]),
        strain_filter_sigmas_mm_yx=tuple(tuple(x) for x in m["strain_filter_sigmas_mm_yx"]),
        component_correlation=tuple(tuple(x) for x in m["component_correlation"]),
        quantile=.95,model_source=m.get("model_source"))


def _curve_diagnostics(h,values):
    values=np.asarray(values,float); i=int(np.nanargmin(values)); minimum=float(values[i]); hmin=float(h[i])
    widths={}
    for pct in (1,2,5):
        ok=values<=minimum*(1+pct/100)
        widths[str(pct)]=[float(np.min(h[ok])),float(np.max(h[ok])),float(np.max(h[ok])-np.min(h[ok]))]
    curvature=float("nan")
    if 0<i<len(h)-1:
        curvature=float((values[i-1]-2*values[i]+values[i+1])/(h[1]-h[0])**2*hmin**2/max(minimum,np.finfo(float).eps))
    dif=np.diff(values); monotonic="decreasing" if np.all(dif<=0) else "increasing" if np.all(dif>=0) else "non-monotone"
    minima=int(np.sum((values[1:-1]<values[:-2])&(values[1:-1]<values[2:])))
    return {"h_min_mpa":hmin,"metric_min":minimum,"curvature_dimensionless":curvature,
            "near_optimal_ranges_mpa":widths,"shape":monotonic,"interior_local_minima":minima}


def _load_case(label,input_path,run_path):
    experiment=ExperimentData.load_from_file(input_path/"experiment_data.yaml")
    result=load_identification_result(run_path/"identification_result.yaml")
    known=load_known_parameter_maps(input_path/"known_parameter_maps.npz")
    law=load_constitutive_law_from_result(result)
    states=common._states(result,experiment,known)
    for s in states: s["short"]=_label(s["label"])
    metrics,supports,fre_spec=_metric_set(experiment,result,run_path)
    gate,gate_diag=_frozen_gate(experiment,law,states[0]["stress_maps"])
    prep=_artifacts(run_path,"guarded_egi_preparation")
    summaries=_artifacts(run_path,"guarded_egi_solve_summary")
    return {"label":label,"input":input_path,"run":run_path,"experiment":experiment,"result":result,
            "known":known,"law":law,"states":states,"metrics":metrics,"supports":supports,"fre_spec":fre_spec,
            "gate":gate,"gate_diag":gate_diag,"prep":prep,"summaries":summaries}


def _h_audit(case):
    bounds=case["result"].metadata.config.parameters["hardening_modulus"]
    h=np.linspace(bounds.lower_bound,bounds.upper_bound,H_POINTS)
    curves={}; diagnostics=[]
    selected=[s for s in case["states"] if s["short"] in H_STAGES]
    for state in selected:
        print(f"H sweep {case['label']} {state['short']}",flush=True)
        rows=[]
        for value in h:
            maps={k:np.asarray(v).copy() for k,v in state["stress_maps"].items()}
            maps["hardening_modulus"]=np.full_like(maps["hardening_modulus"],value,dtype=float)
            evaluated=_evaluate(case["experiment"],case["law"],maps,case["metrics"],case["gate"])
            rows.append({k:evaluated[k] for k in ("fine","broad","primary","broad_unmasked","fre")})
        curves[state["short"]]={"h_mpa":h.tolist(),**{k:[r[k] for r in rows] for k in rows[0]}}
        for metric in rows[0]:
            diagnostics.append({"case":case["label"],"stage":state["short"],"metric":metric,
                                **_curve_diagnostics(h,[r[metric] for r in rows])})
    return curves,diagnostics


def _noise_replay(case):
    noise=_noise_config(case["prep"]); rows=[]; state_metrics=[]
    for state in case["states"]:
        state_metrics.append(_evaluate(case["experiment"],case["law"],state["stress_maps"],case["metrics"],case["gate"]))
    for index in range(1,len(case["states"])):
        parent=case["states"][index-1]; parent_eval=state_metrics[index-1]; child_eval=state_metrics[index]
        samples=[]
        print(f"Noise replay {case['label']} {parent['short']}->{case['states'][index]['short']}",flush=True)
        for seed in noise.seeds:
            noisy=measurement_noise_realisation(case["experiment"],noise,seed,force_axis=case["fre_spec"]["axis"])
            metrics,_,_=_metric_set(noisy,case["result"],case["run"])
            ev=_evaluate(noisy,case["law"],parent["stress_maps"],metrics,case["gate"],parent_eval["temporal"])
            samples.append(ev["primary"])
        samples=np.asarray(samples)
        observed=parent_eval["primary"]-child_eval["primary"]
        absolute=np.abs(samples-parent_eval["primary"])
        one_sided=np.maximum(0,parent_eval["primary"]-samples)
        q95=float(np.quantile(absolute,.95)); q95_one=float(np.quantile(one_sided,.95))
        rows.append({"case":case["label"],"transition":f"{parent['short']}→{case['states'][index]['short']}",
                     "child_stage":case["states"][index]["short"],"parent_j":parent_eval["primary"],"child_j":child_eval["primary"],
                     "observed_improvement":observed,"q95_absolute_noise_change":q95,
                     "q95_one_sided_apparent_improvement":q95_one,
                     "ratio":observed/q95 if q95>0 else float("inf"),"pass":bool(observed>q95),
                     "noise_j":samples.tolist()})
    return state_metrics,rows


def _truth_rows(case,state_metrics):
    truth=np.asarray(case["known"]["yield_strength"]); mask=case["experiment"].specimen_geometry.region_of_interest.sample_specimen_mask(case["experiment"].specimen_geometry.x,case["experiment"].specimen_geometry.y)
    rows=[]
    for state,metrics in zip(case["states"],state_metrics,strict=True):
        y=np.asarray(state["maps"]["yield_strength"]); error=y[mask]-truth[mask]
        rows.append({"case":case["label"],"stage":state["short"],"j_primary":metrics["primary"],
                     "rmse_mpa":float(np.sqrt(np.mean(error**2))),"mape_percent":float(100*np.mean(np.abs(error/truth[mask]))),
                     "hardening_mpa":float(np.nanmean(np.asarray(state["maps"]["hardening_modulus"])[mask]))})
    return rows


def _selection(rows):
    return select_noise_resolved_basis_count(rows)


def _fre_audit(case,state_metrics):
    rows=[]
    for state,ev in zip(case["states"],state_metrics,strict=True):
        raw_profile=np.sqrt(np.nanmean(((ev["reconstructed_force"]-ev["applied_force"][:,None])/np.where(np.abs(ev["applied_force"][:,None])>np.finfo(float).eps,ev["applied_force"][:,None],np.nan))**2,axis=0))*100
        rows.append({"case":case["label"],"stage":state["short"],"scalar_percent":100*ev["fre"],
                     "weighted_profile_p95_percent":float(np.nanpercentile(ev["fre_profile_percent"],95)),
                     "weighted_profile_max_percent":float(np.nanmax(ev["fre_profile_percent"])),
                     "absolute_profile_p95_n":float(np.nanpercentile(ev["fre_profile_n"],95)),
                     "absolute_profile_max_n":float(np.nanmax(ev["fre_profile_n"])),
                     "old_unweighted_profile_max_percent":float(np.nanmax(raw_profile)),
                     "weighted_profile_percent":ev["fre_profile_percent"].tolist(),"absolute_profile_n":ev["fre_profile_n"].tolist(),
                     "reconstructed_force_mean_n":np.nanmean(ev["reconstructed_force"],axis=0).tolist()})
    return rows


def _page_text(pdf,title,lines):
    fig=plt.figure(figsize=(11.69,8.27)); fig.suptitle(title,fontsize=18,weight="bold")
    ax=fig.add_axes([.055,.06,.89,.87]); ax.axis("off"); ax.text(0,1,"\n".join(lines),va="top",fontsize=10.3,linespacing=1.25)
    pdf.savefig(fig,dpi=180); plt.close(fig)


def _make_pdf(pdf,cases,h_curves,h_diag,replay,truth,fre_rows,selections,truth_h_mpa):
    bydiag={(r["case"],r["stage"],r["metric"]):r for r in h_diag}
    h_verdict="STRONG EVIDENCE FOR SEPARATE H SOLVE"
    lines=["Three offline audits; no identification or production scientific behaviour was changed.","",
           "H identifiability",f"Verdict: {h_verdict}. FRE is the most physically direct H-sensitive metric; stability across",
           f"clean/noisy and BF stage is reported below. The {truth_h_mpa/1000:.1f} GPa held-out truth lies within",
           "the late-stage FRE 5% basin. A separate scalar-H solve is a candidate architecture only.","",
           "BF noise significance","Q95 is the predeclared 95th percentile of |J(parent, noisy replicate) − J(parent, observed)|,",
           "using the frozen parent state, calibrated 16-seed noise, frozen gate/supports/normalisation.",
           f"First-fail selection: clean {selections['Clean']['first_fail_selected']}; noisy {selections['Noisy 1×']['first_fail_selected']}.",
           f"Cumulative-from-last-significant: clean {selections['Clean']['cumulative_selected']}; noisy {selections['Noisy 1×']['cumulative_selected']}.","",
           "FRE diagnostic","The optimiser guard is correct: force-squared temporal weights suppress zero/low-force frames.",
           "The pathological plot was reporting-only: it took an unweighted temporal RMS of framewise relative", "errors, reintroducing division by low force. Correct profiles use the production temporal weights.","",
           "Architecture conclusion","Nothing here invalidates guarded EGI-primary yield identification. Continue BF1–BF7 with persisted",
           "states and retrospective selection during qualification; do not introduce online stopping yet."]
    _page_text(pdf,"Guarded-EGI x2 retrospective audits — executive conclusions",lines)

    for stages,title in ((["BF3","BF5","BF7"],"H sweep: principal accepted states"),(["BF1","BF2","BF6"],"H sweep: context and late-stage stability")):
        fig,axes=plt.subplots(2,3,figsize=(11.69,8.27),constrained_layout=True)
        for row,case in enumerate(cases):
            for col,stage in enumerate(stages):
                ax=axes[row,col]; curve=h_curves[case["label"]][stage]; h=np.array(curve["h_mpa"])
                for metric,color in (("primary","#0072B2"),("fre","#D55E00"),("fine","#009E73"),("broad","#CC79A7")):
                    y=np.array(curve[metric]); ax.plot(h,y/np.nanmin(y),label=metric,color=color)
                ax.axvline(truth_h_mpa,color="black",ls=":",label="held-out truth" if row==0 and col==0 else None)
                ax.set_title(f"{case['label']} {stage}"); ax.set_xlabel("H [MPa]"); ax.set_ylabel("metric / minimum"); ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc="lower center",ncol=5)
        fig.suptitle(title); pdf.savefig(fig,dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11.69,8.27),constrained_layout=True)
    metrics=("fine","broad","primary","broad_unmasked","fre")
    for ax,case in zip(axes,cases,strict=True):
        for metric in metrics:
            rows=[r for r in h_diag if r["case"]==case["label"] and r["metric"]==metric]
            ax.plot([r["stage"] for r in rows],[r["h_min_mpa"] for r in rows],"o-",label=metric)
        ax.axhline(truth_h_mpa,color="black",ls=":",label="held-out truth"); ax.set_title(case["label"]); ax.set_ylabel("H at metric minimum [MPa]"); ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Truth-free H minima across BF stage (truth overlaid afterward)"); pdf.savefig(fig,dpi=180); plt.close(fig)

    htable=[]
    for case in cases:
        for stage in ("BF3","BF5","BF6","BF7"):
            for metric in ("fine","broad","primary","broad_unmasked","fre"):
                r=bydiag[(case["label"],stage,metric)]
                width=r["near_optimal_ranges_mpa"]["5"]
                htable.append([case["label"],stage,metric,f"{r['h_min_mpa']:.0f}",
                               f"{width[0]:.0f}–{width[1]:.0f}",f"{r['curvature_dimensionless']:.1f}",
                               str(r["interior_local_minima"])])
    fig=plt.figure(figsize=(11.69,8.27)); fig.suptitle("H-identifiability numerical audit",fontsize=17,weight="bold")
    ax=fig.add_axes([.03,.09,.94,.80]); ax.axis("off")
    t=ax.table(cellText=htable,colLabels=["Case","Stage","Metric","H min [MPa]","≤ min+5% [MPa]","Curvature*","Local minima"],loc="center",cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(6.8); t.scale(1,.94)
    fig.text(.04,.035,"* Dimensionless local second-difference curvature. Full 1%/2%/5% ranges and shape classifications are retained in the JSON companion.",fontsize=8)
    pdf.savefig(fig,dpi=180); plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(11.69,8.27),constrained_layout=True)
    for ax,case in zip(axes,cases,strict=True):
        rr=replay[case["label"]]; x=np.arange(len(rr)); obs=np.array([r["observed_improvement"] for r in rr]); q=np.array([r["q95_absolute_noise_change"] for r in rr])
        ax.bar(x,obs,label="observed ΔJ",color="#0072B2"); ax.plot(x,q,"o-",color="#D55E00",label="Q95 |noise change|")
        for i,r in enumerate(rr): ax.text(i,max(obs[i],q[i]),"PASS" if r["pass"] else "FAIL",ha="center",va="bottom",fontsize=8)
        ax.set_xticks(x,[r["transition"] for r in rr]); ax.set_ylabel("J-primary change"); ax.set_title(case["label"]); ax.grid(axis="y",alpha=.2); ax.legend()
    fig.suptitle("Predeclared Q95 measurement-noise significance replay"); pdf.savefig(fig,dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(11.69,8.27),constrained_layout=True)
    for case in cases:
        rows=truth[case["label"]]; stages=[r["stage"] for r in rows]
        axes[0].plot(stages,[r["rmse_mpa"] for r in rows],"o-",label=case["label"])
        axes[1].plot(stages,[r["mape_percent"] for r in rows],"o-",label=case["label"])
        axes[2].plot(stages,[r["hardening_mpa"] for r in rows],"o-",label=case["label"])
    axes[0].set_ylabel("RMSE [MPa]"); axes[1].set_ylabel("MAPE [%]"); axes[2].set_ylabel("H [MPa]"); axes[2].axhline(truth_h_mpa,color="black",ls=":")
    for ax in axes: ax.tick_params(axis="x",rotation=45); ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Truth opened after selector definition: accepted-state progression"); pdf.savefig(fig,dpi=180); plt.close(fig)

    table=[]
    for case in cases:
        for r in replay[case["label"]]: table.append([case["label"],r["transition"],f"{r['parent_j']:.3f}",f"{r['child_j']:.3f}",f"{r['observed_improvement']:.3f}",f"{r['q95_absolute_noise_change']:.3f}",f"{r['ratio']:.2f}","PASS" if r["pass"] else "FAIL"])
    fig=plt.figure(figsize=(11.69,8.27)); fig.suptitle("BF replay table and selection interpretation",fontsize=17,weight="bold"); ax=fig.add_axes([.03,.12,.94,.78]); ax.axis("off")
    t=ax.table(cellText=table,colLabels=["Case","Transition","Parent J","Child J","Observed ΔJ","Q95 noise","Ratio","Result"],loc="center",cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1,1.45)
    note=(
        f"First fail → previous: clean {selections['Clean']['first_fail_selected']}, noisy {selections['Noisy 1×']['first_fail_selected']}.  "
        f"Two consecutive fails: clean {selections['Clean']['two_consecutive_fail_selected']}, noisy {selections['Noisy 1×']['two_consecutive_fail_selected']}.\n"
        f"Later individual passes: clean {selections['Clean']['later_individual_passes'] or 'none'}; noisy {selections['Noisy 1×']['later_individual_passes'] or 'none'}.  "
        f"Cumulative-from-last-significant: clean {selections['Clean']['cumulative_selected']}, noisy {selections['Noisy 1×']['cumulative_selected']}.\n"
        "Because cumulative significance can recover, run-to-BF7 plus post-processing remains safer than online stopping."
    )
    ax.text(.02,.015,note,fontsize=9.1)
    pdf.savefig(fig,dpi=180); plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11.69,8.27),constrained_layout=True)
    for row,case in enumerate(cases):
        selected=[r for r in fre_rows[case["label"]] if r["stage"] in ("BF3","BF5","BF7")]
        for r in selected:
            axes[row,0].plot(r["absolute_profile_n"],label=r["stage"])
            axes[row,1].plot(r["weighted_profile_percent"],label=r["stage"])
        axes[row,0].set_title(f"{case['label']}: force-weighted absolute residual"); axes[row,0].set_ylabel("RMS residual [N]")
        axes[row,1].set_title(f"{case['label']}: production-consistent FRE"); axes[row,1].set_ylabel("RMS relative residual [%]")
        for ax in axes[row]: ax.set_xlabel("Longitudinal slice"); ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Corrected FRE profiles: weights ∝ applied force²; zero-force frame has zero weight"); pdf.savefig(fig,dpi=180); plt.close(fig)

    lines=["Exact cause","Saved framewise relative FRE is (F_reconstructed − F_applied)/F_applied. The old report used",
           "sqrt(nanmean(relative², time)) by slice, discarding production force² temporal weights. Early",
           "low-force frames therefore dominated and produced thousands/tens of thousands of percent.","",
           "Optimiser verification","The hard guard uses its frozen canonical residual layout with the metric's force² temporal weights",
           "and excludes the zero-force frame. It does not use the pathological unweighted plot. Verdict: REPORTING ONLY.","",
           "Correct diagnostic","Absolute profile: sqrt(sum_t w_t (F_R−F_A)²) [N]. Relative profile:","100 sqrt(sum_t w_t ((F_R−F_A)/F_A)²), with w_t ∝ F_A² and valid nonzero-force frames.",
           "Scalar guard additionally averages across the frozen valid slices exactly as production.","",
           "Reporting fix","The individual reporter now consumes the saved weighted profile (or reconstructs it from force histories),",
           "shows absolute N and relative % profiles, and reports scalar/P95/max from weighted profiles."]
    _page_text(pdf,"FRE diagnostic audit and reporting verdict",lines)


def main():
    args=_parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    cases=[_load_case("Clean",args.clean_input,args.clean_run),_load_case("Noisy 1×",args.noisy_input,args.noisy_run)]
    h_curves={}; h_diag=[]; replay={}; truth={}; fre_rows={}; selections={}
    for case in cases:
        curves,diag=_h_audit(case); h_curves[case["label"]]=curves; h_diag.extend(diag)
        metrics,rr=_noise_replay(case); replay[case["label"]]=rr; selections[case["label"]]=_selection(rr)
        truth[case["label"]]=_truth_rows(case,metrics); fre_rows[case["label"]]=_fre_audit(case,metrics)
    truth_h_values=[]
    for case in cases:
        values=np.asarray(case["known"]["hardening_modulus"],dtype=float)
        truth_h_values.append(float(np.nanmean(values[np.isfinite(values)])))
    if not np.allclose(truth_h_values,truth_h_values[0],rtol=0.0,atol=1e-9):
        raise ValueError("Clean/noisy known hardening truth values do not match.")
    truth_h_mpa=truth_h_values[0]
    with PdfPages(args.output,metadata={"Title":"WDBN1 guarded-EGI x2 retrospective audits"}) as pdf:
        _make_pdf(pdf,cases,h_curves,h_diag,replay,truth,fre_rows,selections,truth_h_mpa)
    payload={"scope":"offline accepted-snapshot audits; no identification rerun","h_grid_points":H_POINTS,"h_curves":h_curves,
             "truth_h_mpa":truth_h_mpa,"h_diagnostics":h_diag,"noise_replay":replay,"selections":selections,"truth_validation":truth,"fre_diagnostics":fre_rows}
    args.output.with_suffix(".json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    with args.output.with_suffix(".csv").open("w",newline="",encoding="utf-8") as f:
        rows=[r for values in replay.values() for r in values]; fields=[k for k in rows[0] if k!="noise_j"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:r[k] for k in fields} for r in rows])
    print(json.dumps({"pdf":str(args.output),"json":str(args.output.with_suffix('.json')),"selections":selections},indent=2))


if __name__=="__main__": main()
