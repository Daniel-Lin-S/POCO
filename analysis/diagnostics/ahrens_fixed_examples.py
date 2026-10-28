"""Render matched POCO Ahrens test traces and diagnostic figures.

Input is a completed fixed-reference report with fixed_reference_per_fit.csv.
Output is a new diagnostics/<UTC timestamp> folder with three PDFs and a
manifest. Fit checkpoints and scientific results remain read only.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from analysis.session_scaling_extended import subject_errors
from configs.config_global import (
    ARTIFACT_ROOT, DEVICE,
)
from models.model_utils import model_init
from utils.ahrens_sessions import processed_directory
from utils.artifacts import validate_complete
from utils.config_utils import load_config
from utils.split_utils import chronological_bounds

SUBJECTS = ("subject_8", "subject_09", "subject_11")
SIZES = (3, 9, 18)
PCS = (0, 20, 50)
SEED = 0
COLORS = ("#d1495b", "#306e9a", "#6b9d45")
WEIGHT_KEYS = (
    "in_proj.0.weight",
    "conditioning_alpha.weight",
    "conditioning_beta.weight",
    "out_proj.weight",
)


def selected_fits(table: pd.DataFrame) -> dict[int, str]:
    """Return complete seed-zero POCO IDs, choosing the best size-nine score."""
    rows = table.loc[(table.model == "POCO") & (table.seed == SEED)]
    selected = {}
    for size in SIZES:
        candidates = rows.loc[rows.sessions_per_fit == size]
        if candidates.empty:
            raise ValueError(f"No POCO fit with {size} sessions.")
        if size != 9 and len(candidates) != 1:
            raise ValueError(
                f"Expected one POCO fit with {size} sessions, "
                f"got {len(candidates)}."
            )
        scores = candidates.test_score.to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite score at {size} sessions.")
        chosen = candidates.sort_values(
            ["test_score", "fit_id"], ascending=[False, True]
        ).iloc[0]
        selected[size] = str(chosen.fit_id)
    return selected


def read_window(subject: str, dataset: object) -> np.ndarray:
    """Return first chronological test window, shape (time, PCs)."""
    path = processed_directory("lowpass") / f"{subject}.npz"
    with np.load(path) as archive:
        if archive["filter_mode"].item() != "lowpass":
            raise ValueError(f"Expected low-pass data in {path}.")
        activity = archive["PC"][:dataset.pc_dim]
    _, start = chronological_bounds(
        activity.shape[1], dataset.train_split,
        dataset.val_split, dataset.test_split,
    )
    window = activity[:, start:start + dataset.seq_length].T.copy()
    expected = (dataset.seq_length, dataset.pc_dim)
    if window.shape != expected or not np.isfinite(window).all():
        raise ValueError(
            f"Expected finite test window {expected} in {path}, "
            f"got {window.shape}."
        )
    return window


def predict(
    fit_path: Path, windows: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Evaluate one checkpoint on matched windows, returning (time, PCs)."""
    config = load_config(str(fit_path))
    if not validate_complete(config):
        raise ValueError(f"Incomplete fit: {fit_path}.")
    label = config.dataset_label[0]
    dataset = config.dataset_config[label]
    subjects = dataset.session_ids
    if not set(SUBJECTS).issubset(subjects):
        raise ValueError(f"Fit lacks reference subjects: {fit_path}.")
    units = dataset.pc_dim
    model = model_init(
        config, [[units] * len(subjects)],
        [[np.arange(units) for _ in subjects]],
    )
    state = torch.load(
        fit_path / "net_best.pth", map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    result = {}
    for subject, window in windows.items():
        inputs = [
            torch.zeros(
                dataset.seq_length - dataset.pred_length,
                0, units, device=DEVICE,
            ) for _ in subjects
        ]
        index = subjects.index(subject)
        history = window[:-dataset.pred_length].copy()
        inputs[index] = torch.from_numpy(
            history
        ).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            values = model(inputs)[index][:, 0].cpu().numpy()
        expected = (dataset.pred_length, units)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(
                f"Expected finite prediction {expected} for "
                f"{subject}, got {values.shape}."
            )
        result[subject] = values
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def prediction_figure(
    windows: dict[str, np.ndarray],
    predictions: dict[int, dict[str, np.ndarray]],
    path: Path,
) -> None:
    """Plot identical test targets beside three checkpoint predictions."""
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    for row, subject in enumerate(SUBJECTS):
        for column, pc in enumerate(PCS):
            ax = axes[row, column]
            horizon = predictions[SIZES[0]][subject].shape[0]
            target = windows[subject][-horizon:, pc]
            ax.plot(
                np.arange(target.size), target, color="#222222",
                linewidth=2.8, label="Target",
            )
            for size, color in zip(SIZES, COLORS):
                values = predictions[size][subject][:, pc]
                ax.plot(
                    np.arange(values.size), values, color=color,
                    linewidth=2.4, label=f"{size} sessions",
                    linestyle="--" if size == 9 else "-",
                )
            ax.set_title(f"{subject}, PC {pc}", fontsize=17, pad=12)
            ax.set_xlabel("Forecast step", fontsize=15)
            ax.set_ylabel("PC activity", fontsize=15)
            ax.tick_params(labelsize=13)
            sns.despine(ax=ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        bbox_to_anchor=(0.5, 0.005), ncol=4,
        frameon=False, fontsize=16,
    )
    fig.suptitle(fill(
        "POCO on identical Ahrens test windows; seed 0; "
        "best-scoring nine-session group shown", 75,
    ), y=0.995, fontsize=20)
    fig.tight_layout(rect=(0, 0.075, 1, 0.91))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def training_figure(paths: dict[int, Path], path: Path) -> None:
    """Plot saved pooled-group training and validation MSE histories."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    max_update = 0.0
    for size, color in zip(SIZES, COLORS):
        table = pd.read_table(paths[size] / "progress.txt")
        if table.empty:
            raise ValueError(f"Empty progress table in {paths[size]}.")
        x = table.BatchNum.to_numpy(dtype=float)
        max_update = max(max_update, float(x.max()))
        for ax, field in zip(axes, ("TrainLoss", "ValLoss")):
            values = table[field].to_numpy(dtype=float)
            if not np.isfinite(x).all() or not np.isfinite(values).all():
                raise ValueError(f"Non-finite {field} in {paths[size]}.")
            sns.lineplot(
                x=x, y=values, ax=ax, color=color,
                linewidth=2.5, label=f"{size} sessions",
                estimator=None,
            )
    for ax, label in zip(
        axes, ("Training loss (MSE)", "Validation loss (MSE)")
    ):
        ax.set_ylabel(label, fontsize=16)
        ax.set_xlim(0, max_update)
        ax.tick_params(labelsize=13)
        ax.legend().remove()
        sns.despine(ax=ax)
    axes[1].set_xlabel("Training update", fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        bbox_to_anchor=(0.5, 0.01), ncol=3,
        frameon=False, fontsize=15,
    )
    fig.suptitle(fill(
        "POCO pooled-group loss histories; validation "
        "population differs by group size", 75,
    ), y=0.99, fontsize=19)
    fig.tight_layout(rect=(0, 0.09, 1, 0.9))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def weights_figure(paths: dict[int, Path], path: Path) -> None:
    """Plot RMS and magnitudes of shared local MLP checkpoint weights."""
    weights = {}
    for size, fit_path in paths.items():
        state = torch.load(
            fit_path / "net_best.pth", map_location="cpu",
            weights_only=True,
        )
        missing = set(WEIGHT_KEYS) - set(state)
        if missing:
            raise ValueError(
                f"Missing shared MLP weights {sorted(missing)} "
                f"in {fit_path}."
            )
        weights[size] = {
            key: state[key].float().numpy() for key in WEIGHT_KEYS
        }
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    positions = np.arange(len(WEIGHT_KEYS))
    width = 0.24
    for offset, (size, color) in enumerate(zip(SIZES, COLORS)):
        rms = [
            float(np.sqrt(np.mean(weights[size][key] ** 2)))
            for key in WEIGHT_KEYS
        ]
        if not np.isfinite(rms).all():
            raise ValueError(f"Non-finite checkpoint weights at {size}.")
        axes[0].bar(
            positions + (offset - 1) * width, rms,
            width=width, color=color, label=f"{size} sessions",
        )
        values = np.abs(weights[size][WEIGHT_KEYS[0]].ravel())
        sns.kdeplot(
            x=values, ax=axes[1], color=color,
            linewidth=2.5, label=f"{size} sessions",
        )
    axes[0].set_xticks(positions, [
        key.removesuffix(".weight") for key in WEIGHT_KEYS
    ], rotation=15, ha="right")
    axes[0].set_ylabel("Weight RMS", fontsize=16)
    axes[1].set_xlabel("|in_proj.0.weight|", fontsize=16)
    axes[1].set_ylabel("Density", fontsize=16)
    for ax in axes:
        ax.tick_params(labelsize=13)
        ax.legend().remove()
        sns.despine(ax=ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        bbox_to_anchor=(0.5, 0.01), ncol=3,
        frameon=False, fontsize=15,
    )
    fig.suptitle(fill(
        "POCO shared local MLP checkpoint weight magnitudes; seed 0",
        75,
    ), y=0.99, fontsize=19)
    fig.tight_layout(rect=(0, 0.1, 1, 0.91))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def subject_loss_figure(table: pd.DataFrame, path: Path) -> None:
    """Plot fixed-subject test MSE, averaging groups within each seed."""
    rows = table.loc[
        (table.model == "POCO") &
        (table.sessions_per_fit.isin((3, 6, 9, 18)))
    ]
    records = []
    for row in rows.itertuples():
        fit_path = Path(ARTIFACT_ROOT) / "fits" / row.fit_id
        config = load_config(str(fit_path))
        if not validate_complete(config):
            raise ValueError(f"Incomplete fit: {fit_path}.")
        errors = subject_errors(config, "test")
        for subject in SUBJECTS:
            total, count = errors[subject]
            records.append({
                "subject": subject,
                "size": row.sessions_per_fit,
                "seed": row.seed,
                "mse": total / count,
            })
    values = pd.DataFrame(records)
    seed_means = values.groupby(
        ["subject", "size", "seed"], as_index=False
    ).mse.mean()
    summary = seed_means.groupby(
        ["subject", "size"], as_index=False
    ).mse.agg(["mean", "std", "count"]).reset_index()
    if (summary["count"] != 2).any():
        raise ValueError("Expected two seed means for every subject/size.")
    if not np.isfinite(summary[["mean", "std"]]).all().all():
        raise ValueError("Non-finite per-subject test MSE summary.")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, subject in zip(axes, SUBJECTS):
        selected = summary.loc[summary.subject == subject]
        selected = selected.sort_values("size")
        x = selected["size"].to_numpy(dtype=float)
        mean = selected["mean"].to_numpy(dtype=float)
        sem = selected["std"].to_numpy(dtype=float) / np.sqrt(2)
        ax.errorbar(
            x, mean, yerr=sem, color="#306e9a",
            linewidth=2.5, marker="o", markersize=8,
            capsize=5,
        )
        ax.set_xticks((3, 6, 9, 18))
        ax.set_xlim(2, 19)
        ax.set_title(subject, fontsize=17, pad=12)
        ax.set_xlabel("Training sessions", fontsize=15)
        ax.set_ylabel("Test loss (MSE)", fontsize=15)
        ax.tick_params(labelsize=13)
        sns.despine(ax=ax)
    fig.suptitle(fill(
        "POCO fixed-subject test loss; groups averaged within seed; "
        "error bars are SEM across two seeds", 75,
    ), y=0.99, fontsize=19)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate figures from the completed fixed-reference report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    root = parser.parse_args().report_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Missing fixed-reference report: {root}.")
    table = pd.read_csv(root / "fixed_reference_per_fit.csv")
    selected = selected_fits(table)
    paths = {
        size: Path(ARTIFACT_ROOT) / "fits" / fit_id
        for size, fit_id in selected.items()
    }
    datasets = {}
    for size, fit_path in paths.items():
        config = load_config(str(fit_path))
        if not validate_complete(config):
            raise ValueError(f"Incomplete selected fit: {fit_path}.")
        label = config.dataset_label[0]
        datasets[size] = config.dataset_config[label]
    settings = {
        (item.pc_dim, item.seq_length, item.pred_length,
         item.train_split, item.val_split, item.test_split)
        for item in datasets.values()
    }
    if len(settings) != 1:
        raise ValueError(
            f"Incompatible selected fit settings: {settings}."
        )
    windows = {
        subject: read_window(subject, datasets[3])
        for subject in SUBJECTS
    }
    predictions = {
        size: predict(fit_path, windows)
        for size, fit_path in paths.items()
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "diagnostics" / stamp
    output.mkdir(parents=True, exist_ok=False)
    sns.set_theme(style="ticks", context="talk")
    prediction_figure(
        windows, predictions, output / "predictions.pdf"
    )
    training_figure(paths, output / "training_dynamics.pdf")
    weights_figure(paths, output / "shared_mlp_weights.pdf")
    subject_loss_figure(table, output / "subject_test_loss.pdf")
    manifest = {
        "seed": SEED,
        "reference_subjects": SUBJECTS,
        "pc_indices": PCS,
        "test_window_offset": 0,
        "fit_ids_by_session_count": selected,
        "nine_session_selection":
            "maximum fixed-reference test score",
        "training_curves":
            "saved pooled-group train and validation MSE",
        "subject_test_loss":
            "mean groups within seed, then SEM across two seeds",
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"Saved diagnostic figures: {output.resolve()}")


if __name__ == "__main__":
    main()
