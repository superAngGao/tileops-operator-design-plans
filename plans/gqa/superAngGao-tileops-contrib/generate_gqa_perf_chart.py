#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


OUT = Path("/home/ga/tileops-gqa-plan/superAngGao-tileops-contrib")
REPO = Path("/home/ga/TileOPs")
RUN_ID = "25259450950"
RUN_URL = f"https://github.com/tile-ai/TileOPs/actions/runs/{RUN_ID}"
ARTIFACT_DIR = Path(f"/tmp/tileops_run_{RUN_ID}/benchmark")
XML_PATH = ARTIFACT_DIR / "bench_results.xml"
EXCLUDED_WORKLOADS = {"llama8b-1k"}

FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    CJK_FONT = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    CJK_FONT = "DejaVu Sans"

mpl.rcParams["font.family"] = ["DejaVu Sans", CJK_FONT]
mpl.rcParams["svg.fonttype"] = "path"
mpl.rcParams["axes.unicode_minus"] = False

COLORS = {
    "ink": "#172026",
    "muted": "#63707a",
    "line": "#dce4ea",
    "tileops": "#00a896",
    "flashinfer": "#7c5cff",
    "fa3": "#2f6fed",
    "gain": "#f5a524",
}


def ensure_artifact() -> None:
    if XML_PATH.exists():
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "gh",
            "run",
            "download",
            RUN_ID,
            "-R",
            "tile-ai/TileOPs",
            "-n",
            f"tileops_benchmark_{RUN_ID}",
            "-D",
            str(ARTIFACT_DIR),
        ],
        cwd=REPO,
    )


def parse_rows() -> list[dict]:
    ensure_artifact()
    rows: list[dict] = []
    for tc in ET.parse(XML_PATH).iter("testcase"):
        cls = tc.attrib.get("classname", "")
        name = tc.attrib.get("name", "")
        if not cls.endswith("bench_gqa") or not name.startswith("test_gqa_fwd_bench["):
            continue
        props = {p.attrib["name"]: p.attrib.get("value", "") for p in tc.findall("./properties/property")}
        if "fa3_tflops" not in props or "flashinfer_tflops" not in props:
            continue
        workload = name.split("[", 1)[1].rstrip("]")
        if (
            workload.startswith("ws-")
            or workload in EXCLUDED_WORKLOADS
            or workload.startswith("train-")
            or workload.startswith("sft-")
        ):
            continue
        tileops_tflops = float(props["tileops_tflops"])
        fa3_tflops = float(props["fa3_tflops"])
        flashinfer_tflops = float(props["flashinfer_tflops"])
        rows.append(
            {
                "workload": workload,
                "tileops_variant": props.get("tileops_variant", ""),
                "tileops_latency_ms": float(props["tileops_latency_ms"]),
                "tileops_tflops": tileops_tflops,
                "fa3_latency_ms": float(props["fa3_latency_ms"]),
                "fa3_tflops": fa3_tflops,
                "flashinfer_latency_ms": float(props["flashinfer_latency_ms"]),
                "flashinfer_tflops": flashinfer_tflops,
                "tileops_pct_fa3": tileops_tflops / fa3_tflops * 100,
                "flashinfer_pct_fa3": flashinfer_tflops / fa3_tflops * 100,
                "tileops_pct_flashinfer": tileops_tflops / flashinfer_tflops * 100,
            }
        )
    if not rows:
        raise SystemExit(f"No GQA FlashInfer rows parsed from {XML_PATH}")
    return rows


def save_csv(rows: list[dict]) -> None:
    with (OUT / "gqa_perf_run25259450950.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_chart(rows: list[dict]) -> None:
    df = pd.DataFrame(rows).sort_values("tileops_pct_fa3", ascending=True).reset_index(drop=True)
    mean_tileops_fa3 = df["tileops_pct_fa3"].mean()
    mean_tileops_fi = df["tileops_pct_flashinfer"].mean()
    best = df.loc[df["tileops_pct_fa3"].idxmax()]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 4, width_ratios=[3.1, 0.08, 0.95, 0.05], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])

    y = list(range(len(df)))
    offset = 0.18
    ax.barh([i - offset for i in y], df["flashinfer_pct_fa3"], height=0.31,
            color=COLORS["flashinfer"], alpha=0.72, label="FlashInfer")
    bars = ax.barh([i + offset for i in y], df["tileops_pct_fa3"], height=0.31,
                   color=COLORS["tileops"], label="TileOps")

    for bar, pct in zip(bars, df["tileops_pct_fa3"]):
        ax.text(pct + 0.8, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%",
                va="center", fontsize=10.5, color=COLORS["ink"])

    ax.axvline(100, color=COLORS["fa3"], linewidth=2, linestyle=(0, (4, 3)), alpha=0.75)
    ax.text(100.8, len(df) - 0.75, "FA3 = 100%", color=COLORS["fa3"], fontsize=12, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(df["workload"], fontsize=11)
    ax.set_xlim(0, 112)
    ax.set_xlabel("性能占 FA3 的比例（%）", fontsize=12, color=COLORS["muted"])
    ax.set_title("GQA Dense Prefill 性能对比：TileOps / FlashInfer / FA3", loc="left",
                 fontsize=24, fontweight="bold", color=COLORS["ink"], pad=34)
    ax.text(
        0,
        1.012,
        f"Nightly run {RUN_ID} artifact · causal inference dense prefill workloads · H200 · FA3 归一化为 100%",
        transform=ax.transAxes,
        fontsize=12.2,
        color=COLORS["muted"],
    )
    ax.grid(axis="x", color=COLORS["line"], linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["line"])
    ax.tick_params(axis="x", colors=COLORS["muted"])
    ax.tick_params(axis="y", length=0, colors=COLORS["ink"])
    ax.legend(loc="lower right", frameon=False, fontsize=12)

    kpi = fig.add_subplot(gs[0, 2])
    kpi.axis("off")
    kpi.text(0, 0.92, "TileOps 平均相对 FA3", fontsize=14, color=COLORS["muted"])
    kpi.text(0, 0.79, f"{mean_tileops_fa3:.1f}%", fontsize=52, fontweight="bold", color=COLORS["tileops"])
    kpi.text(0, 0.67, "of FA3", fontsize=18, color=COLORS["muted"])

    kpi.text(0, 0.52, "TileOps 平均相对 FlashInfer", fontsize=14, color=COLORS["muted"])
    kpi.text(0, 0.41, f"{mean_tileops_fi:.1f}%", fontsize=38, fontweight="bold", color=COLORS["flashinfer"])

    kpi.text(0, 0.25, "最佳 workload", fontsize=14, color=COLORS["muted"])
    kpi.text(0, 0.16, best["workload"], fontsize=21, fontweight="bold", color=COLORS["ink"])
    kpi.text(0, 0.07, f"{best['tileops_pct_fa3']:.1f}% of FA3", fontsize=23,
             fontweight="bold", color=COLORS["gain"])
    kpi.text(0, -0.02, f"{best['tileops_pct_flashinfer']:.1f}% of FlashInfer",
             fontsize=12.5, color=COLORS["muted"])
    kpi.text(0, -0.13, f"Source: {RUN_URL}", fontsize=9.5, color=COLORS["muted"])

    fig.savefig(OUT / "05_gqa_perf_pr871.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "05_gqa_perf_pr871.svg", bbox_inches="tight")
    fig.savefig(OUT / "05_gqa_perf_run25259450950.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "05_gqa_perf_run25259450950.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = parse_rows()
    save_csv(rows)
    save_chart(rows)
    df = pd.DataFrame(rows)
    print(f"rows={len(rows)}")
    print(f"mean_tileops_fa3={df['tileops_pct_fa3'].mean():.1f}")
    print(f"mean_tileops_flashinfer={df['tileops_pct_flashinfer'].mean():.1f}")


if __name__ == "__main__":
    main()
