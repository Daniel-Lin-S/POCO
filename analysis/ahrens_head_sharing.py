"""Report the complete nested Ahrens POCO sharing factorial.

Input is 96 complete scientific fits from the dedicated campaign, plus
completed reference-subject singleton fits for fixed copy baselines.
Output under figures/ahrens_head_sharing_factorial is per-fit,
per-subject, exposure, and factorial-contrast CSVs, six PDFs, and
manifest.json. Existing model-fit artifacts are read only.
"""

import json
import os
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.diagnostics.figure_style import (
    LABEL_FONT, PANEL_FONT, TICK_FONT, save_diagnostic,
)
from analysis.session_scaling_extended import (
    fixed_metrics, reference_baselines, subject_errors,
)
from configs.config_global import FIG_DIR
from configs.experiments import (
    ahrens_head_sharing_factorial,
    ahrens_session_lowpass_extended,
)
from utils.artifacts import fit_id, validate_complete
from utils.config_utils import configs_dict_unpack

REFERENCE = ("subject_8", "subject_09", "subject_11")
SIZES = (3, 6, 9, 18)
FACTOR_BITS = {
    "W_o": (1, "out_weight_scope"),
    "in_proj": (2, "in_proj_scope"),
    "W_alpha": (4, "alpha_weight_scope"),
}
FACTOR_TITLES = {
    "W_o": r"Output $W_{\mathrm{o}}$",
    "in_proj": r"Input $W_{\mathrm{in}},\,b_{\mathrm{in}}$",
    "W_alpha": r"Scale $W_{\alpha}$",
}
METRICS = ("test_score", "test_loss")
METRIC_YLABELS = {
    "test_score": "Test score",
    "test_loss": "Test MSE",
}
CONTRAST_YLABELS = {
    "test_score": r"$D_j$ (score)",
    "test_loss": r"$D_j$ (MSE)",
}
INTERACTION_YLABELS = {
    "test_score": r"$I$ (score)",
    "test_loss": r"$I$ (MSE)",
}
INTERACTION_TITLES = {
    "W_o+in_proj":
        r"$W_{\mathrm{o}} \times "
        r"(W_{\mathrm{in}},b_{\mathrm{in}})$",
    "W_o+W_alpha":
        r"$W_{\mathrm{o}} \times W_{\alpha}$",
    "in_proj+W_alpha":
        r"$(W_{\mathrm{in}},b_{\mathrm{in}})"
        r" \times W_{\alpha}$",
    "W_o+in_proj+W_alpha":
        r"$W_{\mathrm{o}} \times "
        r"(W_{\mathrm{in}},b_{\mathrm{in}})$"
        "\n" r"$\times W_{\alpha}$",
}
POLICY_ORDER = (0, 1, 2, 4, 3, 5, 6, 7)
POLICY_LABELS = {
    0: "All shared",
    1: r"$W_{\mathrm{o}}$",
    2: r"$W_{\mathrm{in}},b_{\mathrm{in}}$",
    4: r"$W_{\alpha}$",
    3: r"$W_{\mathrm{o}}+(W_{\mathrm{in}},b_{\mathrm{in}})$",
    5: r"$W_{\mathrm{o}}+W_{\alpha}$",
    6: r"$(W_{\mathrm{in}},b_{\mathrm{in}})+W_{\alpha}$",
    7: "All private",
}
HEATMAP_MEAN_DIGITS = {
    "test_score": 3,
    "test_loss": 2,
}
HEATMAP_SEM_DIGITS = {
    "test_score": 4,
    "test_loss": 3,
}
HEATMAP_SIZE = (14, 11)
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)
LUMINANCE_THRESHOLD = 0.5
HEATMAP_PALETTES = {
    "test_score": "viridis",
    "test_loss": "viridis_r",
}
EXPECTED_FITS = 96
SEED_COUNT = 3
SHARED_COLOR = "#165a83"
PRIVATE_COLOR = "#c86829"
CONTRAST_COLOR = "#165a83"
ZERO_COLOR = "#666666"
ZERO_LINE_WIDTH = 1.8
ZERO_LABEL = "0 = equal added-session gain"
ZERO_INTERACTION_LABEL = "0 = no interaction in scaling gain"
TRIPLE_INTERACTION_COLOR = "#684c8b"
JITTER = (-0.12, 0.0, 0.12)


def policy_mask(config: object) -> int:
    """Encode private parameter groups in one fit.

    Parameters
    ----------
    config : object
        Resolved POCO sharing fit configuration.

    Returns
    -------
    int
        Three-bit private-module mask, zero for all shared.
    """
    mask = 0
    for bit, field in FACTOR_BITS.values():
        scope = getattr(config, field)
        if scope not in ("shared", "session"):
            raise ValueError(f"Invalid {field} scope: {scope}.")
        if scope == "session":
            mask |= bit
    return mask


def reference_copy_baselines() -> dict[str, dict[str, float]]:
    """Read validated reference-subject persistence baselines.

    Returns
    -------
    dict[str, dict[str, float]]
        Phase to subject to copy-MSE baseline.
    """
    configs = configs_dict_unpack(ahrens_session_lowpass_extended())
    return {
        phase: reference_baselines(configs, list(REFERENCE), phase)
        for phase in ("val", "test")
    }


def read_exposure(
    config: object, total: int,
) -> list[dict]:
    """Read actual session presentations for a new completed fit.

    Parameters
    ----------
    config : object
        Completed fit configuration.
    total : int
        Total examples in the saved progress table.

    Returns
    -------
    list[dict]
        One subject/count record per new fit; legacy baselines return
        an empty list because they did not save per-subject exposure.
    """
    path = Path(config.save_path) / "exposure.json"
    if not path.is_file():
        if Path(config.save_path).name != fit_id(config):
            raise ValueError(f"Missing new-fit exposure: {path}.")
        return []
    values = json.loads(path.read_text())
    subjects = config.dataset_config[
        config.dataset_label[0]
    ].session_ids
    if (
        not isinstance(values, dict) or
        list(values) != subjects or
        any(type(count) is not int or count < 0
            for count in values.values()) or
        sum(values.values()) != total
    ):
        raise ValueError(
            f"Exposure must match subjects and {total} examples: "
            f"{path}."
        )
    return [
        {
            "fit_id": fit_id(config),
            "subject": subject,
            "training_examples_seen": values[subject],
        } for subject in subjects
    ]


def read_fits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate all factorial fits and read fixed-reference metrics.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Per-fit fixed metrics/resource counts, per-subject metrics,
        and actual per-subject exposure for new fits.
    """
    configs = configs_dict_unpack(ahrens_head_sharing_factorial())
    identities = [fit_id(config) for config in configs]
    if len(configs) != EXPECTED_FITS or len(set(identities)) != EXPECTED_FITS:
        raise ValueError(
            f"Expected {EXPECTED_FITS} distinct fits, "
            f"got {len(configs)} with {len(set(identities))} IDs."
        )
    baselines = reference_copy_baselines()
    per_fit = []
    per_subject = []
    exposures = []
    for config in configs:
        if not validate_complete(config):
            raise ValueError(f"Incomplete fit: {config.save_path}.")
        label = config.dataset_label[0]
        dataset = config.dataset_config[label]
        size = len(dataset.session_ids)
        if (size not in SIZES or
                not set(REFERENCE) <= set(dataset.session_ids)):
            raise ValueError(
                f"Invalid fixed-reference group: {dataset.session_ids}."
            )
        progress = pd.read_table(Path(config.save_path) / "progress.txt")
        total = int(progress.TrainExamplesSeen.iloc[-1])
        if total != config.max_batch * dataset.batch_size:
            raise ValueError(
                f"Expected {config.max_batch * dataset.batch_size} "
                f"examples, got {total} in {config.save_path}."
            )
        flops = json.loads(
            (Path(config.save_path) / "flops.json").read_text()
        )
        row = {
            "fit_id": fit_id(config),
            "seed": config.seed,
            "sessions_per_fit": size,
            "private_mask": policy_mask(config),
            "subjects": "+".join(dataset.session_ids),
            "train_examples_seen": total,
            "trainable_parameters": flops["trainable_parameters"],
            "forward_backward_flops_per_step":
                flops["forward_backward_flops_per_step"],
            "exposure_detail_available": (
                (Path(config.save_path) / "exposure.json").is_file()
            ),
        }
        for phase in ("val", "test"):
            errors = subject_errors(config, phase)
            score, loss = fixed_metrics(
                errors, list(REFERENCE), baselines[phase],
            )
            row[f"{phase}_score"] = score
            row[f"{phase}_loss"] = loss
            for subject in REFERENCE:
                squared, count = errors[subject]
                mse = squared / count
                per_subject.append({
                    "fit_id": row["fit_id"],
                    "seed": config.seed,
                    "sessions_per_fit": size,
                    "private_mask": row["private_mask"],
                    "phase": phase,
                    "subject": subject,
                    "prediction_elements": count,
                    "mse": mse,
                    "score": 1 - mse / baselines[phase][subject],
                })
        per_fit.append(row)
        exposures.extend(read_exposure(config, total))
    result = pd.DataFrame(per_fit)
    numeric = result.select_dtypes(include="number").to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite fixed-reference fit results.")
    expected = {
        (seed, size, mask)
        for seed in range(SEED_COUNT)
        for size in SIZES for mask in range(8)
    }
    observed = set(zip(
        result.seed, result.sessions_per_fit, result.private_mask
    ))
    if observed != expected:
        raise ValueError(
            f"Missing or repeated factorial cells: {expected ^ observed}."
        )
    if not exposures:
        raise ValueError("No new fits have exposure records.")
    return (
        result, pd.DataFrame(per_subject),
        pd.DataFrame(exposures),
    )


def gain(
    lookup: dict, seed: int, size: int,
    mask: int, metric: str,
) -> float:
    """Return fixed-subject improvement from three sessions.

    Parameters
    ----------
    lookup : dict
        (seed, size, mask) to completed per-fit record.
    seed : int
        Random seed.
    size : int
        Larger training cohort size.
    mask : int
        Private-factor bit mask.
    metric : str
        test_score or test_loss.

    Returns
    -------
    float
        Score increase or MSE reduction.
    """
    baseline = lookup[(seed, 3, mask)][metric]
    current = lookup[(seed, size, mask)][metric]
    return (
        current - baseline if metric == "test_score"
        else baseline - current
    )


def contrast_value(
    lookup: dict, seed: int, size: int,
    metric: str, factors: tuple[int, ...], context: int,
) -> float:
    """Calculate a factorial sharing contrast in one context.

    Parameters
    ----------
    lookup : dict
        Complete policy/size/seed result mapping.
    seed : int
        Random seed.
    size : int
        Larger training cohort size.
    metric : str
        test_score or test_loss.
    factors : tuple[int, ...]
        Bits whose sharing effects interact.
    context : int
        Private bits outside the tested factors.

    Returns
    -------
    float
        Signed inclusion-exclusion sharing effect on the gain.
    """
    value = 0.0
    for count in range(len(factors) + 1):
        for subset in combinations(factors, count):
            mask = context
            for bit in subset:
                mask |= bit
            value += (-1) ** count * gain(
                lookup, seed, size, mask, metric,
            )
    return value


def factorial_table(per_fit: pd.DataFrame) -> pd.DataFrame:
    """Compute original-context, marginal, and interaction contrasts.

    Parameters
    ----------
    per_fit : pandas.DataFrame
        One complete result per seed, size, and private mask.

    Returns
    -------
    pandas.DataFrame
        One row per seed, size, metric, and contrast.
    """
    lookup = {
        (int(row.seed), int(row.sessions_per_fit),
         int(row.private_mask)): {
            "test_score": row.test_score,
            "test_loss": row.test_loss,
        }
        for row in per_fit.itertuples()
    }
    factors = list(FACTOR_BITS)
    records = []
    for seed in range(SEED_COUNT):
        for size in SIZES[1:]:
            for metric in METRICS:
                for order in (1, 2, 3):
                    for names in combinations(factors, order):
                        bits = tuple(FACTOR_BITS[name][0]
                                     for name in names)
                        remaining = [
                            FACTOR_BITS[name][0] for name in factors
                            if name not in names
                        ]
                        contexts = [
                            sum(bits_in_context)
                            for count in range(len(remaining) + 1)
                            for bits_in_context in combinations(
                                remaining, count,
                            )
                        ]
                        marginal = np.mean([
                            contrast_value(
                                lookup, seed, size, metric,
                                bits, context,
                            ) for context in contexts
                        ])
                        records.append({
                            "seed": seed,
                            "sessions_per_fit": size,
                            "metric": metric,
                            "contrast": (
                                "factorial_main" if order == 1
                                else "interaction"
                            ),
                            "factors": "+".join(names),
                            "value": float(marginal),
                        })
                        if order == 1:
                            records.append({
                                "seed": seed,
                                "sessions_per_fit": size,
                                "metric": metric,
                                "contrast": "original_context",
                                "factors": names[0],
                                "value": contrast_value(
                                    lookup, seed, size, metric,
                                    bits, 0,
                                ),
                            })
    result = pd.DataFrame(records)
    if not np.isfinite(result.value).all():
        raise ValueError("Non-finite sharing contrasts.")
    return result


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize three seed-level contrasts without treating fits as seeds.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-seed factorial contrasts.

    Returns
    -------
    pandas.DataFrame
        Mean, standard deviation, count, and SEM by contrast.
    """
    keys = [
        "sessions_per_fit", "metric", "contrast", "factors",
    ]
    result = table.groupby(keys)["value"].agg(
        ["mean", "std", "count"],
    ).reset_index()
    if (result["count"] != SEED_COUNT).any():
        raise ValueError("Expected three seeds per contrast.")
    result["sem"] = result["std"] / np.sqrt(SEED_COUNT)
    return result


def score_curves(
    per_fit: pd.DataFrame, metric: str, path: Path,
) -> None:
    """Plot all-shared versus each single-private policy.

    Parameters
    ----------
    per_fit : pandas.DataFrame
        Fixed-reference results for all factorial cells.
    metric : str
        test_score or test_loss.
    path : pathlib.Path
        Absolute PDF output path.
    """
    if metric not in METRIC_YLABELS:
        raise ValueError(f"Unknown comparison metric: {metric}.")
    ylabel = METRIC_YLABELS[metric]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, (name, (bit, _)) in zip(
        axes, FACTOR_BITS.items(),
    ):
        for mask, label, color, style in (
            (0, "All shared", SHARED_COLOR, "-"),
            (bit, "Panel component private", PRIVATE_COLOR, "--"),
        ):
            rows = per_fit.loc[
                per_fit.private_mask == mask
            ].sort_values("sessions_per_fit")
            summary = rows.groupby(
                "sessions_per_fit"
            )[metric].agg(["mean", "std", "count"]).loc[list(SIZES)]
            if (summary["count"] != SEED_COUNT).any():
                raise ValueError(
                    f"Missing seed for {name}, {mask}, {metric}."
                )
            values = summary["mean"].to_numpy(dtype=float)
            sem = (
                summary["std"].to_numpy(dtype=float)
                / np.sqrt(SEED_COUNT)
            )
            sns.lineplot(
                x=SIZES, y=values, ax=ax, color=color,
                linestyle=style, linewidth=3,
                marker="o", markersize=8, estimator=None,
                label=label,
            )
            ax.errorbar(
                SIZES, values, yerr=sem, fmt="none",
                ecolor=color, capsize=4,
            )
        ax.set_title(
            FACTOR_TITLES[name], fontsize=PANEL_FONT, pad=9
        )
        ax.set_xticks(SIZES)
        ax.set_xlim(2, 19)
        ax.set_xlabel("Training sessions", fontsize=LABEL_FONT)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT)
        ax.tick_params(labelsize=TICK_FONT)
        ax.legend().remove()
        sns.despine(ax=ax)
    handles, labels = axes[0].get_legend_handles_labels()
    save_diagnostic(
        fig, path,
        "Fixed-reference Ahrens test: POCO sharing policies\n"
        "Three-seed mean ± SEM",
        handles, labels,
    )


def policy_metric_grid(
    per_fit: pd.DataFrame, metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate the complete eight-policy grid within three seeds.

    Parameters
    ----------
    per_fit : pandas.DataFrame
        One fixed-reference metric per fit, policy, size, and seed.
    metric : str
        Saved test_score or test_loss column.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Mean and SEM grids, each with shape (8 policies, 4 sizes).
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown policy metric: {metric}.")
    expected = pd.MultiIndex.from_product(
        (POLICY_ORDER, SIZES),
        names=("private_mask", "sessions_per_fit"),
    )
    grouped = per_fit.groupby(
        ["private_mask", "sessions_per_fit"]
    )[metric].agg(["mean", "std", "count"])
    if (set(grouped.index) != set(expected) or
            (grouped["count"] != SEED_COUNT).any()):
        raise ValueError(
            f"Expected three seeds in every policy-size "
            f"cell for {metric}."
        )
    grouped = grouped.reindex(expected)
    means = grouped["mean"].unstack(
        "sessions_per_fit"
    ).loc[list(POLICY_ORDER), list(SIZES)]
    sem = (
        grouped["std"] / np.sqrt(SEED_COUNT)
    ).unstack(
        "sessions_per_fit"
    ).loc[list(POLICY_ORDER), list(SIZES)]
    if not np.isfinite(
        (means.to_numpy(), sem.to_numpy())
    ).all():
        raise ValueError(
            f"Non-finite policy means or SEMs for {metric}."
        )
    return means, sem


def policy_heatmap(
    per_fit: pd.DataFrame, metric: str, path: Path,
) -> None:
    """Plot absolute test performance for all eight sharing policies.

    Parameters
    ----------
    per_fit : pandas.DataFrame
        Complete fixed-reference result table with 96 fit rows.
    metric : str
        Saved test_score or test_loss column.
    path : pathlib.Path
        Absolute PDF output path.
    """
    if not path.is_absolute():
        raise ValueError(f"Expected absolute figure path: {path}.")
    means, sem = policy_metric_grid(per_fit, metric)
    mean_digits = HEATMAP_MEAN_DIGITS[metric]
    sem_digits = HEATMAP_SEM_DIGITS[metric]
    annotations = np.empty(means.shape, dtype=object)
    for row in range(means.shape[0]):
        for column in range(means.shape[1]):
            value = means.iat[row, column]
            error = sem.iat[row, column]
            annotations[row, column] = (
                f"{value:.{mean_digits}f}\n"
                f"±{error:.{sem_digits}f}"
            )
    fig, ax = plt.subplots(figsize=HEATMAP_SIZE)
    sns.heatmap(
        means, annot=annotations, fmt="",
        cmap=HEATMAP_PALETTES[metric],
        linewidths=1.2, linecolor="white",
        cbar_kws={"label": METRIC_YLABELS[metric]},
        ax=ax,
    )
    ax.set_xticklabels(
        [str(size) for size in SIZES], rotation=0
    )
    ax.set_yticklabels(
        [POLICY_LABELS[mask] for mask in POLICY_ORDER],
        rotation=0,
    )
    ax.set_xlabel("Training sessions", fontsize=LABEL_FONT)
    ax.set_ylabel("Private parameters", fontsize=LABEL_FONT)
    ax.tick_params(labelsize=TICK_FONT)
    mesh = ax.collections[0]
    for artist, value in zip(
        ax.texts, means.to_numpy().ravel(),
    ):
        rgb = mesh.cmap(mesh.norm(value))[:3]
        luminance = np.dot(rgb, LUMINANCE_WEIGHTS)
        artist.set_color(
            "black" if luminance > LUMINANCE_THRESHOLD
            else "white"
        )
        artist.set_fontsize(TICK_FONT)
    colorbar = mesh.colorbar
    colorbar.ax.tick_params(labelsize=TICK_FONT)
    colorbar.set_label(
        METRIC_YLABELS[metric], fontsize=LABEL_FONT
    )
    save_diagnostic(
        fig, path,
        "Fixed-reference Ahrens test: all sharing policies\n"
        "Cell text: three-seed mean ± SEM; brighter = better",
    )


def contrast_figure(
    raw: pd.DataFrame, summary: pd.DataFrame,
    path: Path, kind: str = "original_context",
) -> None:
    """Plot sharing effects or higher-order scaling interactions.

    Parameters
    ----------
    raw : pandas.DataFrame
        Seed-level factorial contrasts.
    summary : pandas.DataFrame
        Three-seed means and SEMs.
    path : pathlib.Path
        Absolute PDF output path.
    kind : str, optional
        original_context or interaction, default original_context.
    """
    if kind == "original_context":
        entries = FACTOR_TITLES
        ylabel = CONTRAST_YLABELS
        title = (
            "Fixed-reference scaling: shared minus private gain\n"
            "Mean ± SEM over three seeds; dots show seeds"
        )
        zero_label = ZERO_LABEL
    elif kind == "interaction":
        entries = INTERACTION_TITLES
        ylabel = INTERACTION_YLABELS
        title = (
            "Fixed-reference interactions in added-session gain\n"
            "Mean ± SEM over three seeds; dots show seeds"
        )
        zero_label = ZERO_INTERACTION_LABEL
    else:
        raise ValueError(
            f"Expected original_context or interaction, got {kind}."
        )
    fig, axes = plt.subplots(
        len(METRICS), len(entries),
        figsize=(5 * len(entries), 9.5),
        squeeze=False,
    )
    for column, (name, panel_title) in enumerate(
        entries.items()
    ):
        triple = kind == "interaction" and name.count("+") == 2
        color = (
            TRIPLE_INTERACTION_COLOR if triple
            else CONTRAST_COLOR
        )
        style = "--" if triple else "-"
        for row, metric in enumerate(METRICS):
            ax = axes[row, column]
            selected = summary.loc[
                (summary.metric == metric) &
                (summary.contrast == kind) &
                (summary.factors == name)
            ]
            if (len(selected) != len(SIZES[1:]) or
                    set(selected.sessions_per_fit) !=
                    set(SIZES[1:])):
                raise ValueError(
                    f"Missing {kind} summary for {name}, {metric}."
                )
            selected = selected.set_index(
                "sessions_per_fit"
            ).loc[list(SIZES[1:])]
            x = selected.index.to_numpy(dtype=float)
            mean = selected["mean"].to_numpy(dtype=float)
            sem = selected["sem"].to_numpy(dtype=float)
            if (not np.isfinite((mean, sem)).all() or
                    np.any(sem < 0)):
                raise ValueError(
                    f"Invalid {kind} values for {name}, {metric}."
                )
            ax.errorbar(
                x, mean, yerr=sem, color=color,
                linestyle=style, linewidth=2.8,
                marker="o", markersize=8, capsize=5,
            )
            points = raw.loc[
                (raw.metric == metric) &
                (raw.contrast == kind) &
                (raw.factors == name)
            ]
            if len(points) != SEED_COUNT * len(SIZES[1:]):
                raise ValueError(
                    f"Expected nine seed-level {kind} values "
                    f"for {name}, {metric}."
                )
            for seed, offset in zip(
                range(SEED_COUNT), JITTER,
            ):
                chosen = points.loc[points.seed == seed]
                ax.scatter(
                    chosen.sessions_per_fit + offset,
                    chosen.value, s=35, alpha=0.6,
                    color=color,
                )
            zero_line = ax.axhline(
                0, color=ZERO_COLOR, linestyle=":",
                linewidth=ZERO_LINE_WIDTH,
            )
            ax.set_title(
                panel_title, fontsize=PANEL_FONT, pad=9
            )
            ax.set_xticks(SIZES[1:])
            ax.set_xlim(5, 19)
            ax.set_xlabel(
                "Training sessions", fontsize=LABEL_FONT
            )
            ax.set_ylabel(
                ylabel[metric], fontsize=LABEL_FONT
            )
            ax.tick_params(labelsize=TICK_FONT)
            sns.despine(ax=ax)
    save_diagnostic(
        fig, path, title,
        [zero_line], [zero_label],
    )


def write_table(table: pd.DataFrame, path: Path) -> None:
    """Atomically replace one nonempty derived result table.

    Parameters
    ----------
    table : pandas.DataFrame
        Finite scientific results.
    path : pathlib.Path
        Absolute CSV destination.
    """
    if table.empty or not np.isfinite(
        table.select_dtypes(include="number").to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(f"Invalid output table: {path}.")
    pending = path.with_name(f".{path.name}.pending")
    table.to_csv(pending, index=False)
    os.replace(pending, path)


def analyze_ahrens_head_sharing_factorial() -> Path:
    """Produce complete fixed-reference tables and sharing figures.

    Returns
    -------
    pathlib.Path
        Absolute stable figure directory.
    """
    per_fit, per_subject, exposure = read_fits()
    contrasts = factorial_table(per_fit)
    summary = summarize(contrasts)
    output = Path(FIG_DIR) / "ahrens_head_sharing_factorial"
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="ticks", context="talk")
    figures = {
        "comparison_score.pdf": lambda path: score_curves(
            per_fit, "test_score", path,
        ),
        "comparison_loss.pdf": lambda path: score_curves(
            per_fit, "test_loss", path,
        ),
        "sharing_contrasts.pdf": lambda path: contrast_figure(
            contrasts, summary, path,
        ),
        "policy_score_heatmap.pdf": lambda path: policy_heatmap(
            per_fit, "test_score", path,
        ),
        "policy_mse_heatmap.pdf": lambda path: policy_heatmap(
            per_fit, "test_loss", path,
        ),
        "factorial_interactions.pdf": lambda path: contrast_figure(
            contrasts, summary, path, kind="interaction",
        ),
    }
    pending = {}
    try:
        for name, render in figures.items():
            path = output / f".{name}.pending.pdf"
            render(path)
            pending[name] = path
        for name, table in (
            ("per_fit.csv", per_fit),
            ("per_subject.csv", per_subject),
            ("exposure_by_subject.csv", exposure),
            ("factorial_contrasts.csv", contrasts),
            ("factorial_summary.csv", summary),
        ):
            write_table(table, output / name)
        for name, path in pending.items():
            os.replace(path, output / name)
    finally:
        for path in pending.values():
            path.unlink(missing_ok=True)
    manifest = {
        "complete_fit_count": EXPECTED_FITS,
        "seed_count": SEED_COUNT,
        "sizes": SIZES,
        "reference_subjects": REFERENCE,
        "factor_bits": {
            name: bit for name, (bit, _) in FACTOR_BITS.items()
        },
        "score": "Equal subject mean of persistence-normalized scores.",
        "loss": "Pooled prediction-element MSE on fixed subjects.",
        "selection": "Pooled group validation MSE.",
        "contrasts": (
            "Within seed score gain or MSE reduction from size 3; "
            "factorial inclusion-exclusion, then three-seed summary."
        ),
        "legacy_exposure": (
            "Eight reused baselines have total presentations only; "
            "new fits have subject-level counts."
        ),
    }
    pending_manifest = output / ".manifest.json.pending"
    pending_manifest.write_text(json.dumps(manifest, indent=2))
    os.replace(pending_manifest, output / "manifest.json")
    print(f"Saved complete sharing report: {output.resolve()}")
    return output
