"""Compare POCO head distributions and fixed-subject output components.

Read the completed fixed-reference CSV, fit checkpoints, and processed test
segments. Write distribution and component CSV/PDF diagnostics in place.
Completed fit artifacts are read only.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
import torch
import torch.nn.functional as functional

from analysis.diagnostics.ahrens_fixed_examples import (
    PARAMETER_KINDS, SHARED_HEAD_LAYERS, SUBJECTS,
)
from analysis.diagnostics.figure_style import (
    LABEL_FONT, SESSION_COLORS, TICK_FONT, save_diagnostic,
)
from analysis.session_scaling_extended import subject_errors
from configs.config_global import ARTIFACT_ROOT
from models.model_utils import model_init
from utils.ahrens_sessions import processed_directory
from utils.artifacts import validate_complete
from utils.config_utils import load_config
from utils.split_utils import chronological_bounds

SIZES = (3, 9, 18)
TERMS = ("input_scaled", "embedding_beta", "beta_bias", "output_bias")
TERM_LABELS = ("Input × scale", "Embedding β", "Wₒ bβ", "bₒ")
BATCH_SIZE = 32
RECONSTRUCTION_TOLERANCE = 0.001
MSE_TOLERANCE = 0.002


def complete_fit(fit_id: str) -> tuple[Path, object]:
    """Return a validated fit directory and configuration.

    Parameters
    ----------
    fit_id : str
        Completed scientific fit hash.

    Returns
    -------
    tuple[pathlib.Path, object]
        Absolute fit directory and saved configuration.
    """
    path = Path(ARTIFACT_ROOT) / "fits" / fit_id
    config = load_config(str(path))
    if not validate_complete(config):
        raise ValueError(f"Incomplete fit: {path}.")
    return path, config


def head_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load globally shared head parameters as flat finite arrays.

    Parameters
    ----------
    path : pathlib.Path
        Complete fit directory.

    Returns
    -------
    dict[str, numpy.ndarray]
        Parameter name to flattened values.
    """
    state = torch.load(
        path / "net_best.pth", map_location="cpu", weights_only=True,
    )
    result = {}
    for layer in SHARED_HEAD_LAYERS:
        for kind in PARAMETER_KINDS:
            name = f"{layer}.{kind}"
            if name not in state:
                raise ValueError(f"Missing {name} in {path}.")
            values = state[name].numpy().astype(np.float64).ravel()
            if values.size < 2 or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name} in {path}.")
            result[name] = values
    return result


def distribution_results(rows: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame
]:
    """Compare all nine- and eighteen-session fits to same-seed triples.

    Parameters
    ----------
    rows : pandas.DataFrame
        Fixed-reference per-fit table.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Per-fit parameter metrics and two-seed summary. Six nine-session
        groups are averaged within seed before summarizing seeds.
    """
    rows = rows.loc[
        (rows.model == "POCO") & rows.sessions_per_fit.isin(SIZES)
    ]
    counts = rows.groupby(["seed", "sessions_per_fit"]).size()
    seeds = sorted(rows.seed.unique())
    if len(seeds) != 2:
        raise ValueError(f"Expected two seeds, got {seeds}.")
    records = []
    for seed in seeds:
        for size, expected in ((3, 1), (9, 6), (18, 1)):
            actual = counts.get((seed, size), 0)
            if actual != expected:
                raise ValueError(
                    f"Expected {expected} size-{size} seed-{seed} "
                    f"fits, got {actual}."
                )
        base_id = rows.loc[
            (rows.seed == seed) & (rows.sessions_per_fit == 3),
            "fit_id",
        ].iloc[0]
        base_path, _ = complete_fit(str(base_id))
        baseline = head_arrays(base_path)
        candidates = rows.loc[
            (rows.seed == seed) & rows.sessions_per_fit.isin((9, 18))
        ]
        for row in candidates.itertuples():
            path, _ = complete_fit(str(row.fit_id))
            current = head_arrays(path)
            for name, reference in baseline.items():
                values = current[name]
                if values.shape != reference.shape:
                    raise ValueError(
                        f"Shape differs for {name} in {path}: "
                        f"{values.shape} versus {reference.shape}."
                    )
                sd = float(reference.std())
                rms = float(np.sqrt(np.mean(reference ** 2)))
                if sd <= 0 or rms <= 0:
                    raise ValueError(
                        f"Zero baseline spread for {name} in "
                        f"{base_path}."
                    )
                records.append({
                    "fit_id": str(row.fit_id),
                    "seed": int(seed),
                    "sessions_per_fit": int(row.sessions_per_fit),
                    "parameter": name,
                    "parameter_count": values.size,
                    "wasserstein_over_baseline_sd":
                        wasserstein_distance(
                            reference, values,
                        ) / sd,
                    "rms_ratio": np.sqrt(np.mean(values ** 2)) / rms,
                })
    details = pd.DataFrame(records)
    metrics = ["wasserstein_over_baseline_sd", "rms_ratio"]
    seed_means = details.groupby(
        ["seed", "sessions_per_fit", "parameter"],
        as_index=False,
    )[metrics].mean()
    summary = seed_means.groupby(
        ["sessions_per_fit", "parameter"], as_index=False,
    )[metrics].agg(["mean", "std", "count"])
    summary.columns = [
        "sessions_per_fit", "parameter",
        "distance_mean", "distance_sd", "distance_n",
        "rms_ratio_mean", "rms_ratio_sd", "rms_ratio_n",
    ]
    if (summary.distance_n != 2).any():
        raise ValueError("Expected two seed means per parameter.")
    return details, summary


def distribution_figure(
    details: pd.DataFrame, summary: pd.DataFrame,
    selected: dict[int, str], path: Path,
) -> None:
    """Plot normalized distribution distance for all shared-head tensors.

    Parameters
    ----------
    details : pandas.DataFrame
        Per-fit parameter distances.
    summary : pandas.DataFrame
        Two-seed means and standard deviations.
    selected : dict[int, str]
        Displayed seed-zero fit ID by session count.
    path : pathlib.Path
        Absolute PDF destination.
    """
    names = [
        f"{layer}.{kind}" for layer in SHARED_HEAD_LAYERS
        for kind in PARAMETER_KINDS
    ]
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    x = np.arange(len(names))
    for size, shift, hatch in ((9, -0.18, ""), (18, 0.18, "//")):
        data = summary.loc[
            summary.sessions_per_fit == size
        ].set_index("parameter").loc[names]
        ax.bar(
            x + shift, data.distance_mean,
            yerr=data.distance_sd / np.sqrt(2),
            width=0.34, color=SESSION_COLORS[size], hatch=hatch,
            edgecolor="#263746", capsize=4,
            label=f"{size} sessions: mean ± SEM",
        )
        shown = details.loc[
            details.fit_id == selected[size]
        ].set_index("parameter").loc[names]
        ax.scatter(
            x + shift, shown.wasserstein_over_baseline_sd,
            s=72, facecolors="white", edgecolors="#111111",
            linewidths=1.7, zorder=4,
            label=f"{size} sessions: displayed fit",
        )
    ax.set_xticks(x, names, rotation=28, ha="right")
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_ylabel(
        "Distribution distance / 3-session SD", fontsize=LABEL_FONT,
    )
    ax.set_xlabel(
        "Globally shared head parameter", fontsize=LABEL_FONT,
    )
    ax.tick_params(labelsize=TICK_FONT)
    sns.despine(ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    save_diagnostic(
        fig, path,
        "POCO shared-head shifts | fixed Ahrens subjects\n"
        "Group mean within seed; comparison with same-seed triple",
        handles, labels,
    )


def load_models(
    selected: dict[int, str], device: torch.device,
) -> tuple[dict[int, tuple[object, object]], object]:
    """Load three checkpoint models with identical fixed-test settings.

    Parameters
    ----------
    selected : dict[int, str]
        Selected fit ID by session count.
    device : torch.device
        Evaluation device.

    Returns
    -------
    tuple[dict[int, tuple[object, object]], object]
        Model/config by session count and common dataset config.
    """
    models = {}
    settings = set()
    dataset = None
    for size in SIZES:
        path, config = complete_fit(selected[size])
        item = config.dataset_config[config.dataset_label[0]]
        if (
            config.conditioning != "mlp" or config.normalize_input or
            config.mu_module_mode != "none" or
            config.std_module_mode != "none"
        ):
            raise ValueError(f"Unexpected output normalization in {path}.")
        if item.test_set_window_stride != 1:
            raise ValueError(f"Expected test stride one in {path}.")
        if not set(SUBJECTS).issubset(item.session_ids):
            raise ValueError(f"Missing fixed subjects in {path}.")
        settings.add((
            item.pc_dim, item.seq_length, item.pred_length,
            item.train_split, item.val_split, item.test_split,
        ))
        units = item.pc_dim
        model = model_init(
            config, [[units] * len(item.session_ids)],
            [[np.arange(units) for _ in item.session_ids]],
        )
        state = torch.load(
            path / "net_best.pth", map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        models[size] = model, config
        dataset = item
    if len(settings) != 1:
        raise ValueError(f"Fixed-test settings differ: {settings}.")
    return models, dataset


def test_segment(subject: str, dataset: object) -> np.ndarray:
    """Read the fixed chronological test segment as (time, PCs).

    Parameters
    ----------
    subject : str
        Reference subject name.
    dataset : object
        Saved dataset configuration.

    Returns
    -------
    numpy.ndarray
        Finite test data with no train/validation samples.
    """
    path = processed_directory("lowpass") / f"{subject}.npz"
    with np.load(path) as archive:
        if archive["filter_mode"].item() != "lowpass":
            raise ValueError(f"Expected low-pass data in {path}.")
        activity = archive["PC"][:dataset.pc_dim]
    _, start = chronological_bounds(
        activity.shape[1], dataset.train_split,
        dataset.val_split, dataset.test_split,
    )
    result = np.ascontiguousarray(activity[:, start:].T)
    if result.shape[0] <= dataset.seq_length:
        raise ValueError(f"Test segment too short in {path}.")
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite test data in {path}.")
    return result


def decompose(
    model: object, config: object, subject: str,
    history: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Split one POCO forecast into four output-space terms.

    Parameters
    ----------
    model : object
        Loaded POCO checkpoint.
    config : object
        Matching fit configuration.
    subject : str
        Evaluated reference subject.
    history : torch.Tensor
        Activity of shape (batch, history frames, PCs).
    device : torch.device
        Model device.

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        Prediction and additive terms, each (batch, PCs, horizon).
    """
    dataset = config.dataset_config[config.dataset_label[0]]
    index = dataset.session_ids.index(subject)
    inputs = [
        torch.empty(
            history.shape[1], 0, dataset.pc_dim, device=device,
        ) for _ in dataset.session_ids
    ]
    inputs[index] = history.permute(1, 0, 2).contiguous()
    cache = {}
    handles = []
    for name in ("in_proj", "conditioning_alpha",
                 "conditioning_beta"):
        def capture(_module, _arguments, output, key=name):
            cache.setdefault(key, []).append(output)

        handles.append(
            getattr(model, name).register_forward_hook(capture)
        )
    try:
        with torch.no_grad():
            prediction = model(inputs)[index].permute(1, 2, 0)
    finally:
        for handle in handles:
            handle.remove()
    for key in ("in_proj", "conditioning_alpha",
                "conditioning_beta"):
        if len(cache.get(key, [])) != len(inputs):
            raise ValueError(
                f"Expected {len(inputs)} {key} invocations."
            )
    hidden = cache["in_proj"][index]
    alpha = cache["conditioning_alpha"][index]
    beta = cache["conditioning_beta"][index]
    weight = model.out_proj.weight
    beta_bias = model.conditioning_beta.bias
    terms = {
        "input_scaled": functional.linear(
            hidden * alpha, weight,
        ),
        "embedding_beta": functional.linear(
            beta - beta_bias, weight,
        ),
        "beta_bias": functional.linear(beta_bias, weight),
        "output_bias": model.out_proj.bias,
    }
    terms = {
        key: value.expand_as(prediction)
        for key, value in terms.items()
    }
    error = (sum(terms.values()) - prediction).abs().max().item()
    if error > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"Decomposition mismatch for {subject}: {error:.6g}."
        )
    return prediction, terms


def component_results(
    selected: dict[int, str], device: torch.device,
) -> pd.DataFrame:
    """Allocate test-MSE differences over all fixed-subject test windows.

    Parameters
    ----------
    selected : dict[int, str]
        Three selected checkpoints.
    device : torch.device
        Evaluation device.

    Returns
    -------
    pandas.DataFrame
        Per-subject and pooled component MSE-change allocation. Each
        component receives mean(2*baseline_error*delta_component +
        delta_component*delta_prediction).
    """
    models, dataset = load_models(selected, device)
    history_length = dataset.seq_length - dataset.pred_length
    totals = {
        (subject, size): {
            "n": 0, "baseline": 0.0, "current": 0.0,
            "impact": {key: 0.0 for key in TERMS},
            "change_sq": {key: 0.0 for key in TERMS},
        }
        for subject in SUBJECTS for size in (9, 18)
    }
    for subject in SUBJECTS:
        activity = test_segment(subject, dataset)
        windows = activity.shape[0] - dataset.seq_length
        for start in range(0, windows, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, windows)
            history = np.stack([
                activity[pos:pos + history_length]
                for pos in range(start, stop)
            ])
            target = np.stack([
                activity[
                    pos + history_length:pos + dataset.seq_length
                ].T for pos in range(start, stop)
            ])
            inputs = torch.from_numpy(history).to(device)
            truth = torch.from_numpy(target).to(device)
            predictions = {}
            components = {}
            for size, (model, config) in models.items():
                predictions[size], components[size] = decompose(
                    model, config, subject, inputs, device,
                )
            baseline_error = predictions[3] - truth
            for size in (9, 18):
                error = predictions[size] - truth
                total_delta = predictions[size] - predictions[3]
                entry = totals[(subject, size)]
                entry["n"] += truth.numel()
                entry["baseline"] += float(
                    baseline_error.double().square().sum()
                )
                entry["current"] += float(
                    error.double().square().sum()
                )
                for key in TERMS:
                    delta = components[size][key] - components[3][key]
                    entry["change_sq"][key] += float(
                        delta.double().square().sum()
                    )
                    entry["impact"][key] += float((
                        2 * baseline_error.double() * delta.double()
                        + delta.double() * total_delta.double()
                    ).sum())
        for size in SIZES:
            saved_error, saved_count = subject_errors(
                models[size][1], "test",
            )[subject]
            expected_count = (
                windows * dataset.pc_dim * dataset.pred_length
            )
            if saved_count != expected_count:
                raise ValueError(
                    f"Test count mismatch for {subject}, {size} "
                    f"sessions: {saved_count} versus {expected_count}."
                )
            observed_sum = (
                totals[(subject, 9)]["baseline"]
                if size == 3 else totals[(subject, size)]["current"]
            )
            observed = observed_sum / saved_count
            expected = saved_error / saved_count
            if abs(observed - expected) > MSE_TOLERANCE:
                raise ValueError(
                    f"Test MSE mismatch for {subject}, {size} "
                    f"sessions: {observed:.6f} versus {expected:.6f}."
                )
    records = []
    for size in (9, 18):
        for subject in (*SUBJECTS, "pooled_reference"):
            selected_entries = (
                [totals[(subject, size)]]
                if subject != "pooled_reference" else
                [totals[(item, size)] for item in SUBJECTS]
            )
            count = sum(item["n"] for item in selected_entries)
            baseline = sum(
                item["baseline"] for item in selected_entries
            ) / count
            current = sum(
                item["current"] for item in selected_entries
            ) / count
            impact = {
                key: sum(
                    item["impact"][key] for item in selected_entries
                ) / count for key in TERMS
            }
            if abs(sum(impact.values()) - (current - baseline)) > 0.001:
                raise ValueError(
                    f"Allocated MSE mismatch for {subject}, "
                    f"{size} sessions."
                )
            for key in TERMS:
                records.append({
                    "subject": subject,
                    "training_sessions": size,
                    "component": key,
                    "test_elements": count,
                    "mse_3_sessions": baseline,
                    "mse_current": current,
                    "mse_change": current - baseline,
                    "allocated_mse_change": impact[key],
                    "rms_component_change": np.sqrt(
                        sum(
                            item["change_sq"][key]
                            for item in selected_entries
                        ) / count
                    ),
                })
    return pd.DataFrame(records)


def component_figure(table: pd.DataFrame, path: Path) -> None:
    """Plot four output-component contributions to fixed-test MSE change.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-subject and pooled component impact table.
    path : pathlib.Path
        Absolute PDF destination.
    """
    pooled = table.loc[table.subject == "pooled_reference"]
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6.8),
        gridspec_kw={"width_ratios": (2.2, 1)},
    )
    x = np.arange(len(TERMS))
    bias_values = []
    for size, shift, hatch in ((9, -0.18, ""), (18, 0.18, "//")):
        values = pooled.loc[
            pooled.training_sessions == size
        ].set_index("component").loc[list(TERMS)]
        impacts = values.allocated_mse_change.to_numpy()
        axes[0].bar(
            x + shift, impacts, width=0.34,
            color=SESSION_COLORS[size], hatch=hatch,
            edgecolor="#263746",
            label=f"{size} versus 3 sessions",
        )
        biases = np.array((
            impacts[2], impacts[3], impacts[2] + impacts[3],
        ))
        bias_values.extend(biases)
        axes[1].bar(
            np.arange(3) + shift, biases, width=0.34,
            color=SESSION_COLORS[size], hatch=hatch,
            edgecolor="#263746",
        )
    for ax in axes:
        ax.axhline(0, color="#222222", linewidth=1.6)
        ax.tick_params(labelsize=TICK_FONT)
        sns.despine(ax=ax)
    axes[0].set_xticks(x, TERM_LABELS, rotation=18, ha="right")
    axes[0].set_xlim(-0.6, len(TERMS) - 0.4)
    axes[0].set_ylabel(
        "Allocated test MSE change (PC activity²)",
        fontsize=LABEL_FONT,
    )
    axes[0].set_title("All output terms", fontsize=LABEL_FONT)
    axes[1].set_xticks(
        np.arange(3), ("Wₒ bβ", "bₒ", "Combined"),
        rotation=18, ha="right",
    )
    axes[1].set_xlim(-0.6, 2.4)
    limit = max(abs(value) for value in bias_values) * 1.25
    if limit <= 0 or not np.isfinite(limit):
        raise ValueError("Invalid bias contribution range.")
    axes[1].set_ylim(-limit, limit)
    axes[1].set_title("Bias terms enlarged", fontsize=LABEL_FONT)
    handles, labels = axes[0].get_legend_handles_labels()
    save_diagnostic(
        fig, path,
        "POCO output components | fixed Ahrens test windows\n"
        "Seed 0; negative reduces MSE; descriptive allocation",
        handles, labels,
    )


def save_table(table: pd.DataFrame, path: Path) -> None:
    """Atomically save a nonempty finite derived CSV.

    Parameters
    ----------
    table : pandas.DataFrame
        Derived results with numerical metrics.
    path : pathlib.Path
        Absolute CSV destination.
    """
    numeric = table.select_dtypes(include="number")
    if (
        table.empty or not np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError(f"Invalid derived table for {path}.")
    pending = path.with_name(f".{path.stem}.pending.csv")
    table.to_csv(pending, index=False)
    os.replace(pending, path)


def main() -> None:
    """Generate in-place derived diagnostic tables and PDFs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    root = args.report_root.resolve()
    output = args.diagnostics_dir.resolve()
    if output.parent != root / "diagnostics":
        raise ValueError(f"Diagnostics must be within {root}.")
    with (output / "manifest.json").open() as stream:
        manifest = json.load(stream)
    selected = {
        int(key): value for key, value in
        manifest["fit_ids_by_session_count"].items()
    }
    if set(selected) != set(SIZES):
        raise ValueError(f"Expected selected sizes {SIZES}.")
    rows = pd.read_csv(root / "fixed_reference_per_fit.csv")
    sns.set_theme(style="ticks", context="talk")
    details, summary = distribution_results(rows)
    components = component_results(
        selected, torch.device(args.device),
    )
    distribution_pdf = output / ".parameter_shift.pending.pdf"
    component_pdf = output / ".component_impact.pending.pdf"
    distribution_figure(
        details, summary, selected, distribution_pdf,
    )
    component_figure(components, component_pdf)
    save_table(
        details, output / "parameter_distribution_shift.csv",
    )
    save_table(
        summary, output / "parameter_distribution_summary.csv",
    )
    save_table(
        components, output / "forecast_component_impact.csv",
    )
    os.replace(
        distribution_pdf, output / "parameter_distribution_shift.pdf",
    )
    os.replace(
        component_pdf, output / "forecast_component_impact.pdf",
    )
    manifest["parameter_shift_analysis"] = (
        "All six nine-session groups and both seeds; group mean within "
        "seed. Wasserstein distance is divided by same-seed three-session "
        "parameter SD. Open circles identify displayed seed-zero fits."
    )
    manifest["component_analysis"] = (
        "All fixed-subject test windows for selected seed-zero fits. "
        "Four additive output terms. MSE-change allocation is algebraic "
        "and does not establish causal parameter effects."
    )
    pending = output / ".parameter_manifest.pending.json"
    with pending.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(pending, output / "manifest.json")
    print(f"Saved parameter and component diagnostics: {output}")


if __name__ == "__main__":
    main()
