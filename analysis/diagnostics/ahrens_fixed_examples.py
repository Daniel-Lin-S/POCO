"""Render matched POCO Ahrens test traces and diagnostic figures.

Input is a completed fixed-reference report with fixed_reference_per_fit.csv.
Output is a diagnostics/<UTC timestamp> folder with five figures, an
effective-offset CSV, and a manifest. Saved fits remain read only.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from analysis.session_scaling_extended import subject_errors
from analysis.diagnostics.figure_style import (
    FIGURE_SUFFIX, LABEL_FONT, PANEL_FONT, TICK_FONT, SESSION_COLORS,
    SESSION_WIDTHS, save_diagnostic,
)
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
CONTEXT_FRAMES = 8
CONTEXT_COLOR = "#e8e8e8"
SHARED_HEAD_LAYERS = (
    "in_proj.0",
    "conditioning_alpha",
    "conditioning_beta",
    "out_proj",
)
PARAMETER_KINDS = ("weight", "bias")
WEIGHT_VIEW_QUANTILES = (0.005, 0.995)
OFFSET_PAD_FRACTION = 0.08
OFFSET_MIN_PAD = 0.005


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
    """Plot the last eight observed frames and all matched forecast frames.

    Parameters
    ----------
    windows : dict[str, numpy.ndarray]
        Reference subject to (context plus horizon, PCs) test window.
    predictions : dict[int, dict[str, numpy.ndarray]]
        Session count and subject to (horizon, PCs) predictions.
    path : pathlib.Path
        Absolute output PDF destination.
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5))
    for row, subject in enumerate(SUBJECTS):
        for column, pc in enumerate(PCS):
            ax = axes[row, column]
            horizon = predictions[SIZES[0]][subject].shape[0]
            window = windows[subject]
            if window.shape[0] - horizon < CONTEXT_FRAMES:
                raise ValueError(f"Insufficient context for {subject}.")
            observed = window[-horizon - CONTEXT_FRAMES:, pc]
            x = np.arange(-CONTEXT_FRAMES, horizon)
            ax.axvspan(
                -CONTEXT_FRAMES, -0.5, color=CONTEXT_COLOR, zorder=0,
            )
            ax.axvline(-0.5, color="#777777", linewidth=1.2, zorder=1)
            ax.plot(
                x, observed, color="#111111", linestyle="--",
                linewidth=2.5, marker="o", markersize=5,
                markerfacecolor="white", markeredgewidth=1.3,
                markevery=2, label="Target", zorder=5,
            )
            for size in SIZES:
                values = predictions[size][subject][:, pc]
                ax.plot(
                    np.arange(horizon), values,
                    color=SESSION_COLORS[size],
                    linewidth=SESSION_WIDTHS[size], linestyle="-",
                    label=f"{size} sessions", zorder=2,
                )
            ax.set_title(
                f"{subject} · PC {pc}", fontsize=PANEL_FONT, pad=9,
            )
            ax.set_xlim(-CONTEXT_FRAMES, horizon - 1)
            ax.set_xticks((-CONTEXT_FRAMES, 0, 8, horizon - 1))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.set_xlabel("Relative frame", fontsize=LABEL_FONT)
            ax.set_ylabel("PC activity", fontsize=LABEL_FONT)
            ax.tick_params(labelsize=TICK_FONT)
            sns.despine(ax=ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(facecolor=CONTEXT_COLOR))
    labels.append("Context (8 frames)")
    save_diagnostic(
        fig, path,
        "POCO | 3, 9, 18 training sessions | seed 0\n"
        "Best 9-session test score; forecast begins at frame 0",
        handles, labels,
    )


def training_figure(paths: dict[int, Path], path: Path) -> None:
    """Plot saved pooled-group training and validation MSE histories."""
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8))
    max_update = 0.0
    for size in SIZES:
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
                x=x, y=values, ax=ax, color=SESSION_COLORS[size],
                linewidth=SESSION_WIDTHS[size], linestyle="-",
                label=f"{size} sessions", estimator=None,
            )
    for ax, label in zip(
        axes, ("Training loss (MSE)", "Validation loss (MSE)")
    ):
        ax.set_ylabel(label, fontsize=LABEL_FONT)
        ax.set_xlim(0, max_update)
        ax.tick_params(labelsize=TICK_FONT)
        ax.legend().remove()
        sns.despine(ax=ax)
    axes[1].set_xlabel("Training update", fontsize=LABEL_FONT)
    handles, labels = axes[0].get_legend_handles_labels()
    save_diagnostic(
        fig, path,
        "POCO loss histories | seed 0\n"
        "Each curve uses its full training / validation group",
        handles, labels,
    )


def weights_figure(paths: dict[int, Path], path: Path) -> None:
    """Plot signed ECDFs of global forecast-head parameters.

    Parameters
    ----------
    paths : dict[int, pathlib.Path]
        Session count to completed POCO fit directory.
    path : pathlib.Path
        Absolute output PDF path.
    """
    keys = tuple(
        f"{layer}.{kind}"
        for layer in SHARED_HEAD_LAYERS
        for kind in PARAMETER_KINDS
    )
    parameters = {}
    parameter_shapes = {}
    for size, fit_path in paths.items():
        state = torch.load(
            fit_path / "net_best.pth", map_location="cpu",
            weights_only=True,
        )
        missing = set(keys) - set(state)
        if missing:
            raise ValueError(
                f"Missing shared forecast-head parameters {sorted(missing)} "
                f"in {fit_path}."
            )
        parameters[size] = {}
        parameter_shapes[size] = {}
        for key in keys:
            parameter_shapes[size][key] = tuple(state[key].shape)
            values = state[key].float().numpy().ravel()
            if values.size < 2 or not np.isfinite(values).all():
                raise ValueError(
                    f"Expected at least two finite values for "
                    f"{key} in {fit_path}."
                )
            parameters[size][key] = values
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 14))
    for row, layer in enumerate(SHARED_HEAD_LAYERS):
        for column, kind in enumerate(PARAMETER_KINDS):
            key = f"{layer}.{kind}"
            ax = axes[row, column]
            shapes = {
                parameter_shapes[size][key] for size in SIZES
            }
            if len(shapes) != 1:
                raise ValueError(
                    f"Shared parameter shape differs across fits "
                    f"for {key}: {sorted(shapes)}."
                )
            count = parameters[SIZES[0]][key].size
            for size in SIZES:
                sns.ecdfplot(
                    x=parameters[size][key], ax=ax,
                    color=SESSION_COLORS[size],
                    linewidth=SESSION_WIDTHS[size], linestyle="-",
                    label=f"{size} sessions",
                )
            pooled = np.concatenate([
                parameters[size][key] for size in SIZES
            ])
            lower, upper = np.quantile(
                pooled, WEIGHT_VIEW_QUANTILES
            )
            if not np.isfinite((lower, upper)).all() or lower >= upper:
                raise ValueError(
                    f"No finite parameter range for {key}."
                )
            padding = (upper - lower) * 0.04
            ax.set_xlim(lower - padding, upper + padding)
            ax.axvline(0, color="#333333", linewidth=1, alpha=0.45)
            ax.set_title(
                f"{key} · shared\n{count:,} parameters",
                fontsize=PANEL_FONT, pad=9,
            )
            ax.set_xlabel(
                "Signed parameter value",
                fontsize=LABEL_FONT,
            )
            ax.set_ylabel("Cumulative fraction", fontsize=LABEL_FONT)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=TICK_FONT)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.legend().remove()
            sns.despine(ax=ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    save_diagnostic(
        fig, path,
        "POCO shared forecast-head parameters | seed 0\n"
        "α/β parameters shared; their outputs can vary by session",
        handles, labels,
    )


def effective_offset_table(paths: dict[int, Path]) -> pd.DataFrame:
    """Compute the shared constant forecast term at each horizon step.

    Parameters
    ----------
    paths : dict[int, pathlib.Path]
        Session count to completed fit directory.

    Returns
    -------
    pandas.DataFrame
        One row per session count and forecast step, with output-bias,
        beta-bias contribution, and their sum in PC activity units.
    """
    records = []
    for size, fit_path in paths.items():
        config = load_config(str(fit_path))
        if not validate_complete(config):
            raise ValueError(f"Incomplete fit: {fit_path}.")
        if (config.conditioning != "mlp" or
                config.normalize_input or
                config.mu_module_mode != "none" or
                config.std_module_mode != "none"):
            raise ValueError(
                f"Expected MLP conditioning and identity output "
                f"normalization in {fit_path}."
            )
        state = torch.load(
            fit_path / "net_best.pth", map_location="cpu",
            weights_only=True,
        )
        keys = (
            "out_proj.weight", "conditioning_beta.bias",
            "out_proj.bias",
        )
        missing = set(keys) - set(state)
        if missing:
            raise ValueError(
                f"Missing effective-offset tensors {sorted(missing)} "
                f"in {fit_path}."
            )
        output_weight = state[keys[0]].double().numpy()
        beta_bias = state[keys[1]].double().numpy()
        output_bias = state[keys[2]].double().numpy()
        expected_weight = (
            config.pred_length, config.conditioning_dim
        )
        if (output_weight.shape != expected_weight or
                beta_bias.shape != (config.conditioning_dim,) or
                output_bias.shape != (config.pred_length,)):
            raise ValueError(
                f"Expected W_o {expected_weight}, b_beta "
                f"{(config.conditioning_dim,)}, and b_o "
                f"{(config.pred_length,)} in {fit_path}; got "
                f"{output_weight.shape}, {beta_bias.shape}, "
                f"and {output_bias.shape}."
            )
        beta_contribution = output_weight @ beta_bias
        offset = beta_contribution + output_bias
        arrays = (output_weight, beta_bias, output_bias, offset)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError(
                f"Non-finite effective offset in {fit_path}."
            )
        for step, (beta_term, bias, total) in enumerate(zip(
            beta_contribution, output_bias, offset
        )):
            records.append({
                "training_sessions": size,
                "forecast_step": step,
                "W_o_times_b_beta": float(beta_term),
                "b_o": float(bias),
                "effective_constant_offset": float(total),
            })
    return pd.DataFrame(records)


def effective_offset_figure(
    table: pd.DataFrame, path: Path,
) -> None:
    """Plot the checkpoint constant term by forecast step and scale.

    Parameters
    ----------
    table : pandas.DataFrame
        Effective-offset table with one row per scale and step.
    path : pathlib.Path
        Absolute PDF destination.
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    for size in SIZES:
        rows = table.loc[table.training_sessions == size]
        if rows.empty:
            raise ValueError(f"No offset rows for {size} sessions.")
        rows = rows.sort_values("forecast_step")
        steps = rows.forecast_step.to_numpy(dtype=int)
        if not np.array_equal(steps, np.arange(len(steps))):
            raise ValueError(
                f"Expected consecutive forecast steps for "
                f"{size} sessions, got {steps.tolist()}."
            )
        values = rows.effective_constant_offset.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite constant offset for {size} sessions."
            )
        sns.lineplot(
            x=steps, y=values, ax=ax,
            color=SESSION_COLORS[size],
            linewidth=SESSION_WIDTHS[size], marker="o",
            markersize=7, estimator=None,
            label=f"{size} sessions",
        )
    values = table.effective_constant_offset.to_numpy(dtype=float)
    lower, upper = float(values.min()), float(values.max())
    padding = max((upper - lower) * OFFSET_PAD_FRACTION,
                  OFFSET_MIN_PAD)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xlim(0, int(table.forecast_step.max()))
    ax.set_xticks(np.arange(0, int(table.forecast_step.max()) + 1, 3))
    ax.axhline(0, color="#333333", linewidth=1.5, linestyle=":")
    ax.set_xlabel("Forecast step", fontsize=LABEL_FONT)
    ax.set_ylabel("Constant offset (PC activity)", fontsize=LABEL_FONT)
    ax.tick_params(labelsize=TICK_FONT)
    sns.despine(ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    save_diagnostic(
        fig, path,
        "POCO shared-head constant offset | seed 0\n"
        r"$W_o b_\beta + b_o$; input and embedding terms excluded",
        handles, labels,
    )


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
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
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
        ax.set_title(subject, fontsize=PANEL_FONT, pad=9)
        ax.set_xlabel("Training sessions", fontsize=LABEL_FONT)
        ax.set_ylabel("Test loss (MSE)", fontsize=LABEL_FONT)
        ax.tick_params(labelsize=TICK_FONT)
        sns.despine(ax=ax)
    save_diagnostic(
        fig, path,
        "POCO test loss on the fixed reference subjects\n"
        "Groups averaged within seed; error bars: SEM across two seeds",
    )


def main() -> None:
    """Generate figures from the completed fixed-reference report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--replace-dir", type=Path)
    args = parser.parse_args()
    root = args.report_root.resolve()
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
    if args.replace_dir is not None:
        output = args.replace_dir.resolve()
        if output.parent != root / "diagnostics":
            raise ValueError(
                f"Replacement directory must be under {root}."
            )
        with (output / "manifest.json").open() as stream:
            previous = json.load(stream)
        expected = {str(size): fit_id for size, fit_id in selected.items()}
        if previous.get("fit_ids_by_session_count") != expected:
            raise ValueError(
                f"Selected fit IDs differ from {output} manifest."
            )
        sns.set_theme(style="ticks", context="talk")
        pending_figures = {
            "predictions.pdf": output / ".predictions.pending.pdf",
            "training_dynamics.pdf":
                output / ".training_dynamics.pending.pdf",
            "shared_mlp_weights.pdf":
                output / ".shared_mlp_weights.pending.pdf",
            "subject_test_loss.pdf":
                output / ".subject_test_loss.pending.pdf",
            "effective_constant_offset.pdf":
                output / ".effective_constant_offset.pending.pdf",
        }
        try:
            prediction_figure(
                windows, predictions, pending_figures["predictions.pdf"]
            )
            training_figure(
                paths, pending_figures["training_dynamics.pdf"]
            )
            weights_figure(
                paths, pending_figures["shared_mlp_weights.pdf"]
            )
            subject_loss_figure(
                table, pending_figures["subject_test_loss.pdf"]
            )
            offset_table = effective_offset_table(paths)
            effective_offset_figure(
                offset_table,
                pending_figures["effective_constant_offset.pdf"],
            )
            csv_pending = output / ".effective_constant_offset.pending.csv"
            offset_table.to_csv(csv_pending, index=False)
            for name, pending in pending_figures.items():
                os.replace(pending, output / name)
            os.replace(
                csv_pending, output / "effective_constant_offset.csv"
            )
        finally:
            for pending in pending_figures.values():
                pending.unlink(missing_ok=True)
            csv_pending = output / ".effective_constant_offset.pending.csv"
            csv_pending.unlink(missing_ok=True)
        previous["prediction_line_styles"] = {
            "3": "solid", "9": "solid", "18": "solid"
        }
        previous["weight_distribution"] = (
            "Signed ECDF of globally shared forecast-head weights "
            "and biases; x axis shows pooled central 99 percent"
        )
        previous["plotted_shared_parameter_keys"] = [
            f"{layer}.{kind}"
            for layer in SHARED_HEAD_LAYERS
            for kind in PARAMETER_KINDS
        ]
        previous["session_dependent_tensors_excluded"] = [
            "decoder.session_emb.weight", "decoder.unit_emb.weight"
        ]
        previous["conditioning_scope"] = (
            "Alpha and beta projection parameters are globally shared; "
            "their per-unit outputs depend on decoder embeddings, "
            "including session embeddings"
        )
        previous["effective_constant_offset"] = (
            "W_o @ b_beta + b_o, one value per forecast step; "
            "identity output normalization for these fits"
        )
        previous["displayed_context_frames"] = CONTEXT_FRAMES
        previous["prediction_colors"] = SESSION_COLORS
        previous["prediction_widths_points"] = SESSION_WIDTHS
        previous["target_style"] = "black dashed with open circles"
        previous["layout"] = {
            "title_gap_points": 16, "minimum_font_points": 18,
        }
        metadata_pending = output / ".manifest.pending.json"
        with metadata_pending.open("x") as stream:
            json.dump(previous, stream, indent=2)
        os.replace(metadata_pending, output / "manifest.json")
        print(f"Updated diagnostic figures: {output.resolve()}")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "diagnostics" / stamp
    output.mkdir(parents=True, exist_ok=False)
    sns.set_theme(style="ticks", context="talk")
    prediction_figure(
        windows, predictions, output / f"predictions{FIGURE_SUFFIX}"
    )
    training_figure(paths, output / f"training_dynamics{FIGURE_SUFFIX}")
    weights_figure(paths, output / f"shared_mlp_weights{FIGURE_SUFFIX}")
    subject_loss_figure(table, output / f"subject_test_loss{FIGURE_SUFFIX}")
    offset_table = effective_offset_table(paths)
    effective_offset_figure(
        offset_table, output / f"effective_constant_offset{FIGURE_SUFFIX}"
    )
    offset_table.to_csv(
        output / "effective_constant_offset.csv", index=False
    )
    manifest = {
        "seed": SEED,
        "reference_subjects": SUBJECTS,
        "pc_indices": PCS,
        "test_window_offset": 0,
        "effective_constant_offset": (
            "W_o @ b_beta + b_o, one value per forecast step; "
            "identity output normalization for these fits"
        ),
        "displayed_context_frames": CONTEXT_FRAMES,
        "prediction_colors": SESSION_COLORS,
        "prediction_widths_points": SESSION_WIDTHS,
        "target_style": "black dashed with open circles",
        "layout": {"title_gap_points": 16, "minimum_font_points": 18},
        "fit_ids_by_session_count": selected,
        "nine_session_selection":
            "maximum fixed-reference test score",
        "training_curves":
            "saved pooled-group train and validation MSE",
        "subject_test_loss":
            "mean groups within seed, then SEM across two seeds",
        "prediction_line_styles": {
            "3": "solid", "9": "solid", "18": "solid"
        },
        "weight_distribution": (
            "Signed ECDF of globally shared forecast-head weights "
            "and biases; x axis shows pooled central 99 percent"
        ),
        "plotted_shared_parameter_keys": [
            f"{layer}.{kind}"
            for layer in SHARED_HEAD_LAYERS
            for kind in PARAMETER_KINDS
        ],
        "session_dependent_tensors_excluded": [
            "decoder.session_emb.weight", "decoder.unit_emb.weight"
        ],
        "conditioning_scope": (
            "Alpha and beta projection parameters are globally shared; "
            "their per-unit outputs depend on decoder embeddings, "
            "including session embeddings"
        ),
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"Saved diagnostic figures: {output.resolve()}")


if __name__ == "__main__":
    main()
