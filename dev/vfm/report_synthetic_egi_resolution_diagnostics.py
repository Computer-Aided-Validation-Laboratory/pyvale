"""Build the synthetic-only consolidated EGI report from completed CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from investigate_wdbn1_automatic_resolution import _curve_knee


ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/"dev/vfm/output/egi_resolution_diagnostics_20260901"
CASES=(
    ("corrected_clean_x2","Corrected synthetic clean spatial-x2","REFERENCE WDBN1 measurement-noise model"),
    ("corrected_noisy_x2","Corrected synthetic noisy spatial-x2","calibrated WDBN1-like construction noise"),
    ("corrected_clean_full","Corrected synthetic clean full resolution","REFERENCE WDBN1 measurement-noise model"),
    ("corrected_noisy_full","Corrected synthetic noisy full resolution","calibrated WDBN1-like construction noise"),
)


def load_case(name,title,noise_label):
    path=OUTPUT/f"EGI_RESOLUTION_DIAGNOSTIC_{name.upper()}_SUPPORT_SWEEP.csv"
    with path.open(newline="") as stream: rows=[{key:float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    lengths=np.asarray([r["support_mm"] for r in rows]); noise=np.asarray([r["propagated_noise_rms"] for r in rows]); knee=_curve_knee(lengths,noise)
    return {"name":name,"title":title,"noise_label":noise_label,"path":str(path),"rows":rows,"noise_knee_pixels":int(rows[knee]["support_pixels"]),"broad_pixels":int(rows[-1]["support_pixels"]),"broad_mm":rows[-1]["support_mm"]}


def main():
    cases=[load_case(*item) for item in CASES]; report=OUTPUT/"EGI_RESOLUTION_DIAGNOSTICS_SYNTHETIC_ONLY_20260901.pdf"
    with PdfPages(report) as pdf:
        fig=plt.figure(figsize=(11.69,8.27)); fig.text(.06,.94,"Synthetic EGI resolution diagnostics",fontsize=20,weight="bold"); fig.text(.07,.84,"Interim manual workflow",fontsize=15,weight="bold"); fig.text(.08,.78,"1. Inspect these complete support curves.\n2. Choose the fine EGI window manually for the dataset.\n3. Pass --fine-egi-window N to identification.\n4. Identification derives logarithmic middle and geometry broad supports.\n5. Fine/middle/broad freeze before BF1.\n\nNoise knee and SNR are diagnostic only. This report does not select or recommend a fine support.\n\nScope: four corrected FE-derived synthetic ROI products only. Experimental WDBN1 was not run. Representative maps are not included because they were transient in the interrupted all-dataset process; no EGI field was recomputed to create this report.",fontsize=11.5,va="top",linespacing=1.45); pdf.savefig(fig); plt.close(fig)
        for case in cases:
            rows=case["rows"]; pixels=np.asarray([r["support_pixels"] for r in rows]); mm=np.asarray([r["support_mm"] for r in rows]); fig,axes=plt.subplots(2,2,figsize=(11.69,8.27),constrained_layout=True); fig.suptitle(f"{case['title']}\nNoise: {case['noise_label']}",fontsize=16)
            axes[0,0].plot(mm,[r["propagated_noise_rms"] for r in rows]); axes[0,0].axvline(rows[(pixels==case['noise_knee_pixels']).nonzero()[0][0]]["support_mm"],color="k",ls="--",label="diagnostic knee"); axes[0,0].set(xlabel="support [mm]",ylabel="noise RMS",yscale="log",title="Support-specific propagated noise"); axes[0,0].legend()
            axes[0,1].plot(mm,[r["gated_egi_rms"] for r in rows]); axes[0,1].set(xlabel="support [mm]",ylabel="EGI RMS",yscale="log",title="Gated Phase-0 EGI magnitude")
            axes[1,0].plot(pixels,[r["diagnostic_snr"] for r in rows]); axes[1,0].axvline(case["noise_knee_pixels"],color="k",ls="--"); axes[1,0].set(xlabel="support [pixels]",ylabel="signal/noise",yscale="log",title="Diagnostic SNR — not a selector")
            axes[1,1].plot(pixels,[r["previous_support_correlation"] for r in rows],label="adjacent-map correlation"); axes[1,1].plot(pixels,[r["informative_coverage"] for r in rows],label="informative coverage"); axes[1,1].set(xlabel="support [pixels]",ylabel="fraction / correlation",ylim=(0,1.02),title="Scale similarity and coverage"); axes[1,1].legend()
            for ax in axes.flat: ax.grid(alpha=.25)
            pdf.savefig(fig); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11.69,5.6),constrained_layout=True)
        for case in cases:
            rows=case["rows"]; mm=[r["support_mm"] for r in rows]; axes[0].plot(mm,[r["propagated_noise_rms"] for r in rows],label=case["name"]); axes[1].plot(mm,[r["diagnostic_snr"] for r in rows],label=case["name"])
        axes[0].set(xlabel="support [mm]",ylabel="noise RMS",yscale="log",title="Noise comparison"); axes[1].set(xlabel="support [mm]",ylabel="diagnostic SNR",yscale="log",title="Diagnostic only — no fine selection")
        for ax in axes: ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.suptitle("Four corrected synthetic datasets",fontsize=17); pdf.savefig(fig); plt.close(fig)
        fig,ax=plt.subplots(figsize=(11.69,8.27)); ax.axis("off"); table=[[c["name"],len(c["rows"]),c["noise_knee_pixels"],c["broad_pixels"],f"{c['broad_mm']:.3f}","--fine-egi-window N"] for c in cases]; tab=ax.table(cellText=table,colLabels=["dataset","supports","diagnostic knee","broad px","broad mm","manual syntax"],loc="center",cellLoc="center"); tab.scale(1,1.5); fig.suptitle("Synthetic sweep inventory — no automatic fine recommendation",fontsize=17); pdf.savefig(fig); plt.close(fig)
    summary={"scope":"four corrected synthetic datasets only; experimental WDBN1 not run","fine_support_policy":"manual user input","identification_syntax":"--fine-egi-window N","representative_maps":"not persisted by interrupted run; deliberately not recomputed","datasets":[{k:v for k,v in c.items() if k!="rows"} for c in cases],"report":str(report)}; (OUTPUT/"EGI_RESOLUTION_DIAGNOSTICS_SYNTHETIC_ONLY_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n"); print(report)


if __name__=="__main__": main()
