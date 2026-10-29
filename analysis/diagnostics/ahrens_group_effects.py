"""Evaluate checkpoint terms across all fixed-reference Ahrens POCO fits.

Input: fixed_reference_per_fit.csv, completed 3/9/18-session POCO
checkpoints, and processed low-pass archives for subjects 8, 09, and 11.

Output in an existing diagnostics directory:
- checkpoint_term_removal.csv: fit_id, seed, sessions_per_fit, subject,
  removed_term, test_elements, fitted MSE, removal MSE, and their difference.
- checkpoint_term_removal_summary.csv: group-within-seed and two-seed
  summaries of each removal difference.
- group_component_impact.csv: fit_id, seed, sessions_per_fit, subject,
  component, test_elements, both MSEs, their change, allocated change,
  and RMS component change relative to the same-seed triple.
- group_component_impact_summary.csv: group-within-seed and two-seed
  summaries of each component's allocated MSE change.
- group_component_impact.pdf, fitted_term_removal.pdf, and
  scale_term_removal.pdf: figures derived from those tables.
Completed fit artifacts remain read only.
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
import torch

from analysis.diagnostics.ahrens_fixed_examples import SUBJECTS
from analysis.diagnostics.ahrens_parameter_mechanism import (
    BATCH_SIZE, MSE_TOLERANCE, SIZES, TERM_LABELS, TERMS,
    decompose, load_checkpoint, save_table, test_segment,
)
from analysis.diagnostics.figure_style import (
    FIGURE_SUFFIX,
    LABEL_FONT, SESSION_COLORS, TICK_FONT, save_diagnostic,
)
from analysis.session_scaling_extended import subject_errors

ABLATIONS = (
    "scale_all", "scale_embedding", "scale_bias",
    "beta_all", "embedding_beta", "beta_bias",
    "output_bias", "constant_offset",
)
ABLATION_LABELS = (
    "Entire scale branch", "Input × Wαe", "Input × bα",
    "Entire β branch", "Embedding β", "Wₒ bβ",
    "bₒ", "Combined offset",
)
REFERENCE_SIZE = 3
SEED_COUNT = 2
GROUP_COUNTS = {3: 1, 9: 6, 18: 1}
IMPACT_TOLERANCE = 0.001


def study_fits(
    table: pd.DataFrame, device: torch.device,
) -> tuple[list[dict], object]:
    """Load every completed POCO fit at the three requested sizes.

    Parameters
    ----------
    table : pandas.DataFrame
        Fixed-reference per-fit table.
    device : torch.device
        Evaluation device.

    Returns
    -------
    tuple[list[dict], object]
        Loaded fit records and shared dataset settings.
    """
    rows = table.loc[
        (table.model == "POCO") &
        table.sessions_per_fit.isin(SIZES)
    ].copy()
    if rows.fit_id.duplicated().any():
        raise ValueError("Repeated POCO fit ID in fixed-reference table.")
    seeds = sorted(rows.seed.unique())
    if len(seeds) != SEED_COUNT:
        raise ValueError(f"Expected two seeds, got {seeds}.")
    counts = rows.groupby(["seed", "sessions_per_fit"]).size()
    for seed in seeds:
        for size, expected in GROUP_COUNTS.items():
            actual = counts.get((seed, size), 0)
            if actual != expected:
                raise ValueError(
                    f"Expected {expected} size-{size} seed-{seed} "
                    f"fits, got {actual}."
                )
    rows = rows.sort_values(
        ["seed", "sessions_per_fit", "fit_id"]
    )
    fits = []
    settings = set()
    dataset = None
    for row in rows.itertuples():
        model, config, item = load_checkpoint(
            str(row.fit_id), device,
        )
        settings.add((
            item.pc_dim, item.seq_length, item.pred_length,
            item.train_split, item.val_split, item.test_split,
        ))
        fits.append({
            "fit_id": str(row.fit_id),
            "seed": int(row.seed),
            "size": int(row.sessions_per_fit),
            "model": model,
            "config": config,
        })
        dataset = item
    if len(settings) != 1:
        raise ValueError(f"Fixed test settings differ: {settings}.")
    return fits, dataset


def ablation_terms(
    components: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Form eight output-space terms for checkpoint removal checks.

    Parameters
    ----------
    components : dict[str, torch.Tensor]
        Five additive forecast terms, each (batch, PCs, horizon).

    Returns
    -------
    dict[str, torch.Tensor]
        Terms removed from the fitted prediction.
    """
    return {
        "scale_all": (
            components["scale_embedding"] + components["scale_bias"]
        ),
        "scale_embedding": components["scale_embedding"],
        "scale_bias": components["scale_bias"],
        "beta_all": (
            components["embedding_beta"] + components["beta_bias"]
        ),
        "embedding_beta": components["embedding_beta"],
        "beta_bias": components["beta_bias"],
        "output_bias": components["output_bias"],
        "constant_offset": (
            components["beta_bias"] + components["output_bias"]
        ),
    }


def new_totals(size: int) -> dict:
    """Make finite-error accumulators for one fit and subject.

    Parameters
    ----------
    size : int
        Training session count.

    Returns
    -------
    dict
        Counts, squared-error sums, and conditional comparison sums.
    """
    totals = {
        "n": 0,
        "mse_sum": 0.0,
        "ablation_sum": {key: 0.0 for key in ABLATIONS},
    }
    if size != REFERENCE_SIZE:
        totals["baseline_sum"] = 0.0
        totals["impact_sum"] = {key: 0.0 for key in TERMS}
        totals["change_sq_sum"] = {key: 0.0 for key in TERMS}
    return totals


def accumulate_ablation(
    totals: dict, error: torch.Tensor,
    components: dict[str, torch.Tensor],
) -> None:
    """Accumulate fitted and output-removal squared errors.

    Parameters
    ----------
    totals : dict
        Mutable per-fit, per-subject error sums.
    error : torch.Tensor
        Fitted forecast minus target, (batch, PCs, horizon).
    components : dict[str, torch.Tensor]
        Five exact output-space forecast terms.
    """
    totals["n"] += error.numel()
    totals["mse_sum"] += float(error.double().square().sum())
    for key, term in ablation_terms(components).items():
        changed_error = error.double() - term.double()
        totals["ablation_sum"][key] += float(
            changed_error.square().sum()
        )


def accumulate_comparison(
    totals: dict, error: torch.Tensor,
    prediction: torch.Tensor,
    components: dict[str, torch.Tensor],
    baseline_prediction: torch.Tensor,
    baseline_terms: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> None:
    """Allocate a fit's MSE difference to five exact output terms.

    Parameters
    ----------
    totals : dict
        Mutable per-fit, per-subject comparison sums.
    error : torch.Tensor
        Current fitted forecast error, (batch, PCs, horizon).
    prediction : torch.Tensor
        Current fitted forecast, (batch, PCs, horizon).
    components : dict[str, torch.Tensor]
        Current output-space terms.
    baseline_prediction : torch.Tensor
        Same-seed triple forecast, (batch, PCs, horizon).
    baseline_terms : dict[str, torch.Tensor]
        Same-seed triple output-space terms.
    target : torch.Tensor
        Test target, (batch, PCs, horizon).
    """
    baseline_error = baseline_prediction - target
    prediction_delta = prediction - baseline_prediction
    totals["baseline_sum"] += float(
        baseline_error.double().square().sum()
    )
    for key in TERMS:
        delta = components[key] - baseline_terms[key]
        totals["change_sq_sum"][key] += float(
            delta.double().square().sum()
        )
        totals["impact_sum"][key] += float((
            2 * baseline_error.double() * delta.double()
            + delta.double() * prediction_delta.double()
        ).sum())


def check_saved_losses(
    fits: list[dict], totals: dict,
    dataset: object, subject: str, windows: int,
) -> None:
    """Require recomputed per-subject MSE to match saved test metrics.

    Parameters
    ----------
    fits : list[dict]
        Loaded fit records.
    totals : dict
        Per-fit, per-subject accumulated errors.
    dataset : object
        Common dataset settings.
    subject : str
        Evaluated reference subject.
    windows : int
        Number of evaluated chronological test windows.
    """
    expected_count = (
        windows * dataset.pc_dim * dataset.pred_length
    )
    for fit in fits:
        fit_id = fit["fit_id"]
        entry = totals[(fit_id, subject)]
        saved_sum, saved_count = subject_errors(
            fit["config"], "test",
        )[subject]
        if entry["n"] != expected_count or saved_count != expected_count:
            raise ValueError(
                f"Test element count mismatch for {fit_id}, "
                f"{subject}: {entry['n']}, {saved_count}, "
                f"expected {expected_count}."
            )
        observed = entry["mse_sum"] / expected_count
        expected = saved_sum / expected_count
        if abs(observed - expected) > MSE_TOLERANCE:
            raise ValueError(
                f"Test MSE mismatch for {fit_id}, {subject}: "
                f"{observed:.6f} versus {expected:.6f}."
            )


def evaluate(
    fits: list[dict], dataset: object,
    device: torch.device,
) -> dict[tuple[str, str], dict]:
    """Evaluate all fixed-subject test windows with all checkpoints.

    Parameters
    ----------
    fits : list[dict]
        Complete 3/9/18-session POCO fit records.
    dataset : object
        Shared dataset settings.
    device : torch.device
        Evaluation device.

    Returns
    -------
    dict[tuple[str, str], dict]
        Error sums keyed by fit ID and reference subject.
    """
    totals = {
        (fit["fit_id"], subject): new_totals(fit["size"])
        for fit in fits for subject in SUBJECTS
    }
    baselines = {
        fit["seed"]: fit["fit_id"]
        for fit in fits if fit["size"] == REFERENCE_SIZE
    }
    history_length = dataset.seq_length - dataset.pred_length
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
            results = {}
            for fit in fits:
                prediction, components = decompose(
                    fit["model"], fit["config"],
                    subject, inputs, device,
                )
                results[fit["fit_id"]] = prediction, components
                entry = totals[(fit["fit_id"], subject)]
                accumulate_ablation(
                    entry, prediction - truth, components,
                )
            for fit in fits:
                if fit["size"] == REFERENCE_SIZE:
                    continue
                prediction, components = results[fit["fit_id"]]
                baseline_prediction, baseline_terms = results[
                    baselines[fit["seed"]]
                ]
                accumulate_comparison(
                    totals[(fit["fit_id"], subject)],
                    prediction - truth, prediction, components,
                    baseline_prediction, baseline_terms, truth,
                )
        check_saved_losses(
            fits, totals, dataset, subject, windows,
        )
        print(f"Validated all checkpoints on {subject} test windows.")
    return totals


def pooled_totals(
    totals: dict, fit_id: str, size: int,
) -> dict:
    """Pool three fixed subjects by evaluated forecast element count.

    Parameters
    ----------
    totals : dict
        Per-fit and subject error accumulators.
    fit_id : str
        Fit to pool.
    size : int
        Training session count.

    Returns
    -------
    dict
        Summed error and comparison totals.
    """
    pooled = new_totals(size)
    for subject in SUBJECTS:
        item = totals[(fit_id, subject)]
        pooled["n"] += item["n"]
        pooled["mse_sum"] += item["mse_sum"]
        for key in ABLATIONS:
            pooled["ablation_sum"][key] += item["ablation_sum"][key]
        if size != REFERENCE_SIZE:
            pooled["baseline_sum"] += item["baseline_sum"]
            for key in TERMS:
                pooled["impact_sum"][key] += item["impact_sum"][key]
                pooled["change_sq_sum"][key] += item[
                    "change_sq_sum"
                ][key]
    return pooled


def result_tables(
    fits: list[dict], totals: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert checkpoint sums into per-fit ablation and impact tables.

    Parameters
    ----------
    fits : list[dict]
        Complete fit records.
    totals : dict
        Accumulated fixed-test errors.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Per-fit ablation and comparison rows, including subject and
        pooled-reference rows.
    """
    ablations = []
    impacts = []
    for fit in fits:
        fit_id = fit["fit_id"]
        size = fit["size"]
        seed = fit["seed"]
        for subject in (*SUBJECTS, "pooled_reference"):
            item = (
                totals[(fit_id, subject)]
                if subject != "pooled_reference" else
                pooled_totals(totals, fit_id, size)
            )
            count = item["n"]
            full_mse = item["mse_sum"] / count
            for key in ABLATIONS:
                changed = item["ablation_sum"][key] / count
                ablations.append({
                    "fit_id": fit_id,
                    "seed": seed,
                    "sessions_per_fit": size,
                    "subject": subject,
                    "removed_term": key,
                    "test_elements": count,
                    "full_mse": full_mse,
                    "mse_after_removal": changed,
                    "removal_delta_mse": changed - full_mse,
                })
            if size == REFERENCE_SIZE:
                continue
            baseline = item["baseline_sum"] / count
            effect = {
                key: item["impact_sum"][key] / count
                for key in TERMS
            }
            if abs(
                sum(effect.values()) - (full_mse - baseline)
            ) > IMPACT_TOLERANCE:
                raise ValueError(
                    f"Impact terms do not sum for {fit_id}, "
                    f"{subject}."
                )
            for key in TERMS:
                impacts.append({
                    "fit_id": fit_id,
                    "seed": seed,
                    "sessions_per_fit": size,
                    "subject": subject,
                    "component": key,
                    "test_elements": count,
                    "mse_3_sessions": baseline,
                    "mse_current": full_mse,
                    "mse_change": full_mse - baseline,
                    "allocated_mse_change": effect[key],
                    "rms_component_change": np.sqrt(
                        item["change_sq_sum"][key] / count
                    ),
                })
    return pd.DataFrame(ablations), pd.DataFrame(impacts)


def summarize(
    table: pd.DataFrame, key: str, metric: str,
) -> pd.DataFrame:
    """Average groups within seed, then summarize two seed means.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-fit pooled-reference metric table.
    key : str
        Categorical component or removal-term column.
    metric : str
        Numerical column to summarize.

    Returns
    -------
    pandas.DataFrame
        Group-mean within seed, two-seed mean and SEM.
    """
    pooled = table.loc[
        table.subject == "pooled_reference"
    ]
    seed_means = pooled.groupby(
        ["sessions_per_fit", "seed", key],
        as_index=False,
    )[metric].mean()
    summary = seed_means.groupby(
        ["sessions_per_fit", key], as_index=False,
    )[metric].agg(["mean", "std", "count"])
    summary.columns = [
        "sessions_per_fit", key, "mean", "sd", "seed_count",
    ]
    if (summary.seed_count != SEED_COUNT).any():
        raise ValueError(
            f"Expected {SEED_COUNT} seed means for {key}."
        )
    summary["sem"] = summary.sd / np.sqrt(SEED_COUNT)
    return summary


def impact_figure(
    summary: pd.DataFrame, path: Path,
) -> None:
    """Plot five exact MSE-change allocations for all 9/18 groups.

    Parameters
    ----------
    summary : pandas.DataFrame
        Group-within-seed, then two-seed impact summary.
    path : pathlib.Path
        Absolute PDF destination.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6.8),
        gridspec_kw={"width_ratios": (2.2, 1)},
    )
    all_bias = []
    for size, shift, hatch in ((9, -0.18, ""), (18, 0.18, "//")):
        data = summary.loc[
            summary.sessions_per_fit == size
        ].set_index("component").loc[list(TERMS)]
        values = data["mean"].to_numpy()
        errors = data["sem"].to_numpy()
        axes[0].bar(
            np.arange(len(TERMS)) + shift, values, yerr=errors,
            width=0.34, color=SESSION_COLORS[size], hatch=hatch,
            edgecolor="#263746", capsize=4,
            label=f"{size} versus 3 sessions",
        )
        bias = np.array((
            values[3], values[4], values[3] + values[4],
        ))
        all_bias.extend(bias)
        axes[1].bar(
            np.arange(3) + shift, bias, width=0.34,
            color=SESSION_COLORS[size], hatch=hatch,
            edgecolor="#263746",
        )
    for ax in axes:
        ax.axhline(0, color="#222222", linewidth=1.6)
        ax.tick_params(labelsize=TICK_FONT)
        sns.despine(ax=ax)
    axes[0].set_xticks(
        np.arange(len(TERMS)), TERM_LABELS,
        rotation=20, ha="right",
    )
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
    limit = max(abs(value) for value in all_bias) * 1.4
    if limit <= 0 or not np.isfinite(limit):
        raise ValueError("Invalid all-group bias range.")
    axes[1].set_ylim(-limit, limit)
    axes[1].set_title("Bias terms enlarged", fontsize=LABEL_FONT)
    handles, labels = axes[0].get_legend_handles_labels()
    save_diagnostic(
        fig, path,
        "POCO checkpoint term changes | fixed Ahrens test set\n"
        "Groups within seed; two-seed means; left bars show SEM",
        handles, labels,
    )



def scale_ablation_figure(
    summary: pd.DataFrame, path: Path,
) -> None:
    """Plot test-MSE effects of removing fitted scale terms.

    Parameters
    ----------
    summary : pandas.DataFrame
        Group-within-seed, then two-seed removal summary.
    path : pathlib.Path
        Absolute PDF destination.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    positions = np.arange(len(SIZES))
    for key, shift, color, hatch in (
        ("scale_all", -0.18, "#397aaa", ""),
        ("scale_embedding", 0.18, "#073b65", "//"),
    ):
        data = summary.loc[
            summary.removed_term == key
        ].set_index("sessions_per_fit").loc[list(SIZES)]
        axes[0].bar(
            positions + shift, data["mean"], yerr=data["sem"],
            width=0.34, color=color, hatch=hatch,
            edgecolor="#263746", capsize=4,
            label=ABLATION_LABELS[ABLATIONS.index(key)],
        )
    bias = summary.loc[
        summary.removed_term == "scale_bias"
    ].set_index("sessions_per_fit").loc[list(SIZES)]
    axes[1].bar(
        positions, bias["mean"], yerr=bias["sem"],
        width=0.45, color="#80aed1", edgecolor="#263746",
        capsize=4, label="Input × bα",
    )
    for ax in axes:
        ax.axhline(0, color="#222222", linewidth=1.6)
        ax.set_xticks(positions, SIZES)
        ax.set_xlim(-0.6, len(SIZES) - 0.4)
        ax.set_xlabel("Training sessions", fontsize=LABEL_FONT)
        ax.set_ylabel(
            "Test MSE increase after removal (PC activity²)",
            fontsize=LABEL_FONT,
        )
        ax.tick_params(labelsize=TICK_FONT)
        sns.despine(ax=ax)
    axes[0].set_title("Full and embedding scale",
                      fontsize=LABEL_FONT)
    axes[1].set_title("Constant scale enlarged",
                      fontsize=LABEL_FONT)
    handles = []
    labels = []
    for ax in axes:
        one, two = ax.get_legend_handles_labels()
        handles.extend(one)
        labels.extend(two)
    save_diagnostic(
        fig, path,
        "POCO scale-term removal | fixed Ahrens test set\n"
        "Positive means worse; groups within seed, then two seeds",
        handles, labels,
    )

def ablation_figure(
    summary: pd.DataFrame, path: Path,
) -> None:
    """Plot test-MSE effects of removing fitted beta and bias terms.

    Parameters
    ----------
    summary : pandas.DataFrame
        Group-within-seed, then two-seed ablation summary.
    path : pathlib.Path
        Absolute PDF destination.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, keys, title in (
        (axes[0], ("beta_all",), "Entire β branch"),
        (
            axes[1],
            ("beta_bias", "output_bias", "constant_offset"),
            "Constant terms enlarged",
        ),
    ):
        positions = np.arange(len(SIZES))
        offsets = np.linspace(-0.25, 0.25, len(keys))
        for key, shift in zip(keys, offsets):
            data = summary.loc[
                summary.removed_term == key
            ].set_index("sessions_per_fit").loc[list(SIZES)]
            ax.bar(
                positions + shift, data["mean"],
                yerr=data["sem"], width=0.42 / len(keys),
                color=SESSION_COLORS[9] if key == "beta_all"
                else ("#80aed1" if key == "beta_bias"
                      else "#397aaa" if key == "output_bias"
                      else "#073b65"),
                hatch="//" if key == "constant_offset" else "",
                edgecolor="#263746", capsize=4,
                label=ABLATION_LABELS[ABLATIONS.index(key)],
            )
        ax.axhline(0, color="#222222", linewidth=1.6)
        ax.set_xticks(positions, SIZES)
        ax.set_xlim(-0.6, len(SIZES) - 0.4)
        ax.set_xlabel("Training sessions", fontsize=LABEL_FONT)
        ax.set_title(title, fontsize=LABEL_FONT)
        ax.tick_params(labelsize=TICK_FONT)
        sns.despine(ax=ax)
    axes[0].set_ylabel(
        "Test MSE increase after removal (PC activity²)",
        fontsize=LABEL_FONT,
    )
    axes[1].set_ylabel(
        "Test MSE increase after removal (PC activity²)",
        fontsize=LABEL_FONT,
    )
    handles = []
    labels = []
    for ax in axes:
        one, two = ax.get_legend_handles_labels()
        handles.extend(one)
        labels.extend(two)
    save_diagnostic(
        fig, path,
        "POCO β and bias-term removal | fixed Ahrens test set\n"
        "Positive means worse; groups within seed, then two seeds",
        handles, labels,
    )


def main() -> None:
    """Evaluate checkpoints and save derived diagnostics in place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    root = args.report_root.resolve()
    output = args.diagnostics_dir.resolve()
    if output.parent != root / "diagnostics":
        raise ValueError(f"Diagnostics must be within {root}.")
    table = pd.read_csv(root / "fixed_reference_per_fit.csv")
    fits, dataset = study_fits(
        table, torch.device(args.device),
    )
    totals = evaluate(fits, dataset, torch.device(args.device))
    ablation, impact = result_tables(fits, totals)
    ablation_summary = summarize(
        ablation, "removed_term", "removal_delta_mse",
    )
    impact_summary = summarize(
        impact, "component", "allocated_mse_change",
    )
    sns.set_theme(style="ticks", context="talk")
    pending_impact = output / f".group_component_impact.pending{FIGURE_SUFFIX}"
    pending_ablation = output / f".fitted_term_removal.pending{FIGURE_SUFFIX}"
    pending_scale = output / f".scale_term_removal.pending{FIGURE_SUFFIX}"
    impact_figure(impact_summary, pending_impact)
    ablation_figure(ablation_summary, pending_ablation)
    scale_ablation_figure(ablation_summary, pending_scale)
    for name, data in (
        ("checkpoint_term_removal.csv", ablation),
        ("checkpoint_term_removal_summary.csv", ablation_summary),
        ("group_component_impact.csv", impact),
        ("group_component_impact_summary.csv", impact_summary),
    ):
        save_table(data, output / name)
    os.replace(
        pending_impact, output / f"group_component_impact{FIGURE_SUFFIX}",
    )
    os.replace(
        pending_ablation, output / f"fitted_term_removal{FIGURE_SUFFIX}",
    )
    os.replace(
        pending_scale, output / f"scale_term_removal{FIGURE_SUFFIX}",
    )
    with (output / "manifest.json").open() as stream:
        manifest = json.load(stream)
    manifest["all_checkpoint_components"] = (
        "All complete 3/9/18-session POCO checkpoints on the "
        "same three reference test subjects; five exact output terms. "
        "9-session groups averaged within seed, then two seeds."
    )
    manifest["fitted_term_removal"] = (
        "Subtract fitted scale, beta, and bias output terms at "
        "inference; no refitting. Positive delta means test loss "
        "increased. Scale and beta removal PDFs show the effects. "
        "This measures fitted-model reliance, not causal training "
        "contribution."
    )
    pending = output / ".checkpoint_manifest.pending.json"
    with pending.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(pending, output / "manifest.json")
    print(f"Saved all-checkpoint term analysis: {output}")


if __name__ == "__main__":
    main()
