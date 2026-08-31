"""Standalone manual EGI-resolution diagnostic; never selects a fine support."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pyvale.vfm import (
    EgiSupportBankConfig, EquilibriumGapMetric, ExperimentData,
    calculate_parameter_stress_sensitivities, generate_odd_pixel_egi_support_bank,
)
from investigate_wdbn1_automatic_resolution import _curve_knee, _load_json, _noise_realisation, _homogeneous_result
import call_notched_ebw_bivariate_identification as runner


ROOT=Path(__file__).resolve().parents[2]
DATA=Path("/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe")
DEFAULT_OUTPUT=ROOT/"dev/vfm/output/egi_resolution_diagnostics_20260901"
NOISE_MODEL=ROOT/"dev/vfm/data/wdbn1_noise_model_20260828.yaml"
DATASETS=(
    ("corrected_clean_x2",DATA/"wdbn1-representative-fe-v1-clean-fe-roi-spatial-x2/pyvale-vfm/prepared","clean synthetic; WDBN1 reference noise"),
    ("corrected_noisy_x2",DATA/"wdbn1-representative-fe-v1-noisy-1x-seed20260830-fe-roi-spatial-x2/pyvale-vfm/prepared","noisy synthetic; calibrated construction noise"),
    ("corrected_clean_full",DATA/"wdbn1-representative-fe-v1-final-h0125-r4-fe-roi/pyvale-vfm/prepared","clean synthetic; WDBN1 reference noise"),
    ("corrected_noisy_full",DATA/"wdbn1-representative-fe-v1-noisy-1x-seed20260830-r2-fe-roi/pyvale-vfm/prepared","noisy synthetic; calibrated construction noise"),
    ("experimental_wdbn1",Path("/media/data/3_Resources/gr91-weld-dic-results/wdbn1/pyvale-input/vfm-input-data_2026-08-17_04-10"),"experimental measurement ROI; stationary/low-load calibration"),
)


def derive_manual_supports(x: np.ndarray,y: np.ndarray,fine: int,maximum_bbox_fraction: float=.5):
    bank=generate_odd_pixel_egi_support_bank(x,y,EgiSupportBankConfig(maximum_bbox_fraction=maximum_bbox_fraction)); by={s.window_size[0]:s for s in bank}
    if fine not in by: raise ValueError("fine support is outside the geometry bank")
    first=by[fine]; broad=bank[-1]; target=np.sqrt(first.nominal_side_length*broad.nominal_side_length)
    middle=min((s for s in bank if fine<s.window_size[0]<broad.window_size[0]),key=lambda s:abs(np.log(s.nominal_side_length)-np.log(target)))
    return first,middle,broad


def write_sweep_csv(path: Path,rows: list[dict]) -> None:
    fields=list(rows[0]);
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _gate(strain,stress,law,maps):
    sensitivities=calculate_parameter_stress_sensitivities(strain,stress,law,maps,["yield_strength","hardening_modulus"],perturbation_factor=.01)
    scaled=[]
    for item in sensitivities.values():
        activity=np.sqrt(np.nansum(item.total**2,axis=1)); positive=activity[np.isfinite(activity)&(activity>0)]; scale=np.quantile(positive,.95); scaled.append(np.clip(activity/scale,0,1))
    combined=np.maximum.reduce(scaled); positive=combined[np.isfinite(combined)&(combined>0)]; start,full=np.quantile(positive,(0,.9)); gate=np.clip((combined-start)/(full-start),0,1); gate[~np.isfinite(combined)]=np.nan
    return gate


def _noise_inputs(experiment,mask,count,seed):
    model=_load_json(NOISE_MODEL); x=experiment.specimen_geometry.x; y=experiment.specimen_geometry.y; dx=float(np.nanmedian(np.diff(x,axis=1))); dy=float(np.nanmedian(np.diff(y,axis=0)))
    sigmas=tuple(float(model["components"][n]["sigma"]) for n in ("exx","eyy","exy")); filters=tuple((float(model["components"][n]["gaussian_filter_sigma_mm"]["y"])/dy,float(model["components"][n]["gaussian_filter_sigma_mm"]["x"])/dx) for n in ("exx","eyy","exy"))
    return [_noise_realisation(experiment.strain,mask,sigmas,filters,seed+i) for i in range(count)]


def analyse_dataset(name,path,label,output,noise_replicates=3,maximum_bbox_fraction=.5):
    started=time.perf_counter(); experiment=ExperimentData.load_from_file(path/"experiment_data.yaml"); mask=experiment.specimen_geometry.region_of_interest.sample_specimen_mask(experiment.specimen_geometry.x,experiment.specimen_geometry.y)
    phase0=_homogeneous_result(experiment,output/name,path); maps=phase0.parameter_maps; law=runner._create_constitutive_law("cython",minimum_yield_strength=200); stress=law.calculate_stress(experiment.strain,maps); gate=_gate(experiment.strain,stress,law,maps); spatial_gate=np.nansum(np.where(np.isfinite(gate),gate,0),axis=0)>0
    noise_stress=[law.calculate_stress(value,maps) for value in _noise_inputs(experiment,mask,noise_replicates,20260920)]
    supports=generate_odd_pixel_egi_support_bank(experiment.specimen_geometry.x,experiment.specimen_geometry.y,EgiSupportBankConfig(maximum_bbox_fraction=maximum_bbox_fraction)); representative_indices=sorted(set(np.linspace(0,len(supports)-1,8,dtype=int))); rows=[]; maps_out={}; previous=None
    for index,support in enumerate(supports):
        metric=EquilibriumGapMetric(window_size=np.asarray(support.window_size,dtype=np.uint32),fft_dtype="float32",compute_temporal_rms=True,compute_spatiotemporal_rms=True); metric.initialise(experiment); result=metric.evaluate_equilibrium_gap(stress); weights=np.asarray(result.metric_result.additional_fields["force_weights"]); signal=np.sqrt(np.nansum(weights[:,None,None]*result.normalised_gap**2,axis=0)); deltas=[]
        for value in noise_stress:
            noisy=metric.evaluate_equilibrium_gap(value); deltas.append(np.sqrt(np.nansum(weights[:,None,None]*(noisy.normalised_gap-result.normalised_gap)**2,axis=0)))
        noise=np.sqrt(np.nanmean(np.asarray(deltas)**2,axis=0)); valid=spatial_gate&np.isfinite(signal)&np.isfinite(noise)&(noise>0); correlation=np.nan if previous is None else float(np.corrcoef(previous[valid],signal[valid])[0,1]); previous=signal.copy()
        rows.append({"support_pixels":support.window_size[0],"support_mm":support.nominal_side_length,"informative_coverage":float(np.mean(valid[spatial_gate])),"gated_egi_rms":float(np.sqrt(np.mean(signal[valid]**2))),"propagated_noise_rms":float(np.sqrt(np.mean(noise[valid]**2))),"diagnostic_snr":float(np.sqrt(np.mean(signal[valid]**2))/np.sqrt(np.mean(noise[valid]**2))),"previous_support_correlation":correlation})
        if index in representative_indices: maps_out[str(support.window_size[0])]=signal
        print(f"{name}: {support.window_size[0]} ({index+1}/{len(supports)})",flush=True)
    lengths=np.asarray([r["support_mm"] for r in rows]); noise=np.asarray([r["propagated_noise_rms"] for r in rows]); knee=_curve_knee(lengths,noise); csv_path=output/f"EGI_RESOLUTION_DIAGNOSTIC_{name.upper()}_SUPPORT_SWEEP.csv"; write_sweep_csv(csv_path,rows)
    return {"name":name,"label":label,"path":str(path),"rows":rows,"noise_knee_pixels":rows[knee]["support_pixels"],"broad_pixels":rows[-1]["support_pixels"],"representative_maps":maps_out,"x":experiment.specimen_geometry.x,"y":experiment.specimen_geometry.y,"spatial_gate":spatial_gate,"runtime_seconds":time.perf_counter()-started,"noise_replicates":noise_replicates}


def render_report(results,output):
    report=output/"EGI_RESOLUTION_DIAGNOSTICS_20260901.pdf"
    with PdfPages(report) as pdf:
        fig=plt.figure(figsize=(11.69,8.27)); fig.text(.06,.94,"EGI resolution diagnostics",fontsize=20,weight="bold"); fig.text(.07,.84,"Manual interim workflow",fontsize=15,weight="bold"); fig.text(.08,.78,"1. Run this diagnostic.\n2. Inspect curves and representative maps.\n3. Choose the fine window manually.\n4. Pass --fine-egi-window N to identification.\n5. Identification derives logarithmic middle and geometry broad supports.\n6. All supports freeze before BF1.\n\nNo curve in this report automatically selects or recommends a fine support.",fontsize=12,va="top",linespacing=1.5); pdf.savefig(fig); plt.close(fig)
        for case in results:
            rows=case["rows"]; pixels=np.asarray([r["support_pixels"] for r in rows]); fig,axes=plt.subplots(2,2,figsize=(11.69,8.27),constrained_layout=True); fig.suptitle(f"{case['name']} — {case['label']}",fontsize=16)
            axes[0,0].plot(pixels,[r["propagated_noise_rms"] for r in rows]); axes[0,0].set(title="Support-specific propagated noise",ylabel="noise RMS")
            axes[0,1].plot(pixels,[r["gated_egi_rms"] for r in rows]); axes[0,1].set(title="Gated Phase-0 EGI magnitude",ylabel="EGI RMS")
            axes[1,0].plot(pixels,[r["diagnostic_snr"] for r in rows]); axes[1,0].axvline(case["noise_knee_pixels"],ls="--",color="k",label="noise knee (diagnostic)"); axes[1,0].set(title="Diagnostic signal/noise — not a selector",xlabel="support [pixels]"); axes[1,0].legend()
            axes[1,1].plot(pixels,[r["previous_support_correlation"] for r in rows],label="adjacent correlation"); axes[1,1].plot(pixels,[r["informative_coverage"] for r in rows],label="coverage"); axes[1,1].set(title="Scale similarity and coverage",xlabel="support [pixels]"); axes[1,1].legend()
            for ax in axes.flat: ax.grid(alpha=.25); pdf.savefig(fig); plt.close(fig)
            maps=case["representative_maps"]; values=np.concatenate([v[case["spatial_gate"]] for v in maps.values()]); vmax=np.quantile(values,.98); fig,axes=plt.subplots(2,4,figsize=(11.69,6.2),constrained_layout=True); extent=(np.nanmin(case["x"]),np.nanmax(case["x"]),np.nanmin(case["y"]),np.nanmax(case["y"]))
            for ax,(support,value) in zip(axes.flat,maps.items(),strict=True):
                shown=np.where(case["spatial_gate"],value,np.nan); ax.imshow(shown,origin="lower",extent=extent,vmin=0,vmax=vmax,cmap="magma",aspect="auto"); ax.set_title(f"{support}x{support}"); ax.set(xlabel="x [mm]",ylabel="y [mm]")
            fig.suptitle(f"{case['name']}: representative gated EGI maps (shared scale)"); pdf.savefig(fig); plt.close(fig)
        fig,ax=plt.subplots(figsize=(11.69,8.27)); ax.axis("off"); table=[[r["name"],r["noise_knee_pixels"],r["broad_pixels"],f"{r['runtime_seconds']:.1f}","--fine-egi-window N"] for r in results]; ax.table(cellText=table,colLabels=["dataset","diagnostic knee","geometry broad","runtime s","manual syntax"],loc="center",cellLoc="center").scale(1,1.5); fig.suptitle("Cross-dataset comparison — no automatic fine recommendation",fontsize=17); pdf.savefig(fig); plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--noise-replicates",type=int,default=3); parser.add_argument("--maximum-bbox-fraction",type=float,default=.5); args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    results=[analyse_dataset(name,path,label,args.output,args.noise_replicates,args.maximum_bbox_fraction) for name,path,label in DATASETS]; report=render_report(results,args.output); summary={"fine_support_policy":"manual user input only","identification_syntax":"--fine-egi-window N","datasets":[{k:v for k,v in r.items() if k not in ("representative_maps","x","y","spatial_gate","rows")} for r in results],"report":str(report)}; (args.output/"EGI_RESOLUTION_DIAGNOSTICS_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n"); print(report)


if __name__=="__main__": main()
