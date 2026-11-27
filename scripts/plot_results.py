
"""Build the v4 comparison figure from the archived metrics."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "runs" / "main_experiment_v4" / "metrics"


def read_rows(group: str) -> list[dict[str, str]]:
    with (METRICS / f"metrics_by_{group}.csv").open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    model_labels = {
        "gpt-5.4-mini": "GPT-5.4-mini", "gpt-5.4": "GPT-5.4", "gpt-5.5": "GPT-5.5",
        "deepseek-v4-flash": "DeepSeek V4-Flash", "deepseek-v4-pro": "DeepSeek V4-Pro",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    }
    models = sorted(read_rows("model"), key=lambda row: float(row["detection_accuracy"]), reverse=True)
    prompts = read_rows("prompt")
    fields = ("detection_accuracy", "correction_accuracy", "false_positive_rate")
    legend = ("Обнаружение ↑", "Исправление ↑", "Ложные срабатывания ↓")
    colors = ("#226a67", "#687cb4", "#c26345")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, rows, labels, title in (
        (axes[0], models, [model_labels[row["value"]] for row in models], "Модели"),
        (axes[1], prompts, [row["value"] for row in prompts], "Промпты"),
    ):
        for index, (field, label, color) in enumerate(zip(fields, legend, colors)):
            positions = [position + (index - 1) * 0.23 for position in range(len(rows))]
            axis.barh(positions, [float(row[field]) for row in rows], height=0.2, label=label, color=color)
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_xlim(0, 1)
        axis.set_title(title, loc="left")
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), frameon=False, ncol=1)
    figure.suptitle("Оценка LLM на русских минимальных парах · v4", fontsize=15, fontweight="bold")
    path = ROOT / "docs" / "figures" / "v4-comparison.png"
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path, dpi=145, bbox_inches="tight")


if __name__ == "__main__":
    main()
