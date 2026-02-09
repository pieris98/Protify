#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


MODEL_NAMES = {
    'Random': 'Random vectors',
    'Random-ESM2-8': r'$Random ESM2_{8M}$',
    'Random-ESM2-35': r'$Random ESM2_{35M}$',
    'Random-ESM2-150': r'$Random ESM2_{150M}$',
    'Random-ESM2-650': r'$Random ESM2_{650M}$',
    'Random-Transformer': 'Random Transformer',
    'ESM2-8': r'$ESM2_{8M}$',
    'ESM2-35': r'$ESM2_{35M}$',
    'ESM2-150': r'$ESM2_{150M}$',
    'ESM2-650': r'$ESM2_{650M}$',
    'ESM2-3B': r'$ESM2_{3B}$',
    'ESM2-diff-150': r'$ESM2_{diff-150M}$',
    'ESM2-diffAV-150': r'$ESM2_{diffAV-150M}$',
    'ESMC-300': r'$ESMC_{300M}$',
    'ESMC-600': r'$ESMC_{600M}$',
    'E1-150': r'$E1_{150M}$',
    'E1-300': r'$E1_{300M}$',
    'E1-600': r'$E1_{600M}$',
    'ProtBert': r'$ProtBert_{420M}$',
    'ProtBert-BFD': r'$ProtBert_{BFD}$',
    'ProtT5': r'ProtT5-enc$_{3B}$',
    'ProtT5-XL-UniRef50-full-prec': r'ProtT5-XL$_{UniRef50}$',
    'ProtT5-XXL-UniRef50': r'ProtT5-XXL$_{UniRef50}$',
    'ProtT5-XL-BFD': r'ProtT5-XL$_{BFD}$',
    'ProtT5-XXL-BFD': r'ProtT5-XXL$_{BFD}$',
    'ANKH-Base': r'ANKH-Base$_{400M}$',
    'ANKH-Large': r'ANKH-Large$_{1.2B}$',
    'ANKH2-Large': r'ANKH2-Large$_{1.2B}$',
    'DSM-150': r'$DSM_{150M}$',
    'DSM-650': r'$DSM_{650M}$',
    'DSM-PPI': r'$DSM_{PPI}$',
    'GLM2-150': r'$GLM2_{150M}$',
    'GLM2-650': r'$GLM2_{650M}$',
    'GLM2-GAIA': r'$GLM2_{GAIA}$',
    'DPLM-150': r'$DPLM_{150M}$',
    'DPLM-650': r'$DPLM_{650M}$',
    'DPLM-3B': r'$DPLM_{3B}$',
    'ProtCLM-1b': r'$ProtCLM_{1B}$',
    'OneHot-Protein': 'OneHot Protein',
    'OneHot-DNA': 'OneHot DNA',
    'OneHot-RNA': 'OneHot RNA',
    'OneHot-Codon': 'OneHot Codon',
    'AMPLIFY-120': r'$AMPLIFY_{120M}$',
    'AMPLIFY-350': r'$AMPLIFY_{350M}$',
}

DATASET_NAMES = {
    # Gene Ontology and Enzyme Commission
    'EC_reg': 'EC',
    'CC_reg': r'$GO_{CC}$',
    'BP_reg': r'$GO_{BP}$',
    'MF_reg': r'$GO_{MF}$',
    
    # Basic protein properties
    'MB_reg': 'MB',
    'DL2_reg': r'$DL_{2}$',
    'DL10_reg': r'$DL_{10}$',
    'SL_13': 'Subcellular',
    'enzyme_kcat': r'$k_{cat}$',
    'solubility_prediction': 'solubility',
    'localization_prediction': 'localization',
    'temperature_stability': 'temperature stability',
    'optimal_temperature': 'optimal temperature',
    'optimal_ph': 'optimal pH',
    'material_production': 'material production',
    'fold_prediction': 'folds',
    'cloning_clf': 'cloning-clf',
    'stability_prediction': 'stability',
    'ec_active': r'$EC_{singlelabel}$',
    'soluprot': 'soluprot',
    
    # Protein-protein interactions
    'human-ppi-saprot': r'$Human-PPI_{saprot}$',
    'human-ppi-pinui': r'$Human-PPI_{PiNUI}$',
    'yeast-ppi-pinui': r'$Yeast-PPI_{PiNUI}$',
    'peptide-HLA-MHC-affinity': 'peptide HLA MHC affinity',
    'shs27-ppi-raw': r'$SHS_{27k-raw}-ppi$',
    'shs148-ppi-raw': r'$SHS_{148k-raw}-ppi$',
    'shs27-ppi-random': r'$SHS_{27k-random}-ppi$',
    'shs148-ppi-random': r'$SHS_{148k-random}-ppi$',
    'shs27-ppi-dfs': r'$SHS_{27k-dfs}-ppi$',
    'shs148-ppi-dfs': r'$SHS_{148k-dfs}-ppi$',
    'shs27-ppi-bfs': r'$SHS_{27k-bfs}-ppi$',
    'shs148-ppi-bfs': r'$SHS_{148k-bfs}-ppi$',
    'string-ppi-random': r'$STRING_{random-ppi}$',
    'string-ppi-dfs': r'$STRING_{dfs-ppi}$',
    'string-ppi-bfs': r'$STRING_{bfs-ppi}$',
    'ppi_SHS148k_bfs_2025': r'$SHS_{148k-bfs}-ppi$',
    'ppi_SHS148k_dfs_2025': r'$SHS_{148k-dfs}-ppi$',
    'ppi_SHS27k_bfs_2025': r'$SHS_{27k-bfs}-ppi$',
    'ppi_SHS27k_dfs_2025': r'$SHS_{27k-dfs}-ppi$',
    'ppi_SHS27k_random_2025': r'$SHS_{27k-random}-ppi$',
    'ppi_SHS148k_random_2025': r'$SHS_{148k-random}-ppi$',
    'ppi_STRING_random_2025': r'$STRING_{random}-ppi$',
    'ppi_STRING_dfs_2025': r'$STRING_{dfs}-ppi$',
    'ppi_STRING_bfs_2025': r'$STRING_{bfs}-ppi$',
    'gold-ppi': r'$Human PPI_{bernett}$',
    'plm-interact': r'$PLM-Interact_{human / cross}$',
    'plm_interact_human_train_cross_ppi': r'$PLM-Interact_{human / cross}$',
    'ppi-mutation-effect': r'$PPI_{mutation effect}$',
    'PPA-ppi': r'$PPA_{PPI}$',
    'bernett_gold_ppi': r'$Human PPI_{bernett}$',
    'ProteinProteinAffinity': 'PPI binding affinity',

    # Secondary structure
    'SS3': r'$SS_{3}$',
    'SS8': r'$SS_{8}$',
    
    # "Fitness"
    'fluorescence_prediction': 'fluorescence',
    'fitness_prediction': 'fitness',
    'millionfull_round_1_oct_2025': r'$AtOMT1_{millionfull}$',
    'million_full': r'$AtOMT1_{millionfull}$',

    # Special datasets
    'plastic': r'$plastic degradation_{benchmark}$',
    'foldseek-fold': 'foldseek fold',
    'foldseek-inverse': 'foldseek inverse',
    'ec-active': 'EC singlelabel',
    'bernett_processed': r'$Bernett_{processed}$',
    
    # ProteinGym datasets
    'proteingym_zs': r'$ProteinGym_{zero-shot}$',
    'proteingym_supervised': r'$ProteinGym_{supervised}$',
    
    # Taxonomic datasets (alternative naming with full prefix)
    'taxonomy_domain': r'$taxonomy_{domain}$',
    'taxonomy_kingdom': r'$taxonomy_{kingdom}$', 
    'taxonomy_phylum': r'$taxonomy_{phylum}$',
    'taxonomy_class': r'$taxonomy_{class}$',
    'taxonomy_order': r'$taxonomy_{order}$',
    'taxonomy_family': r'$taxonomy_{family}$',
    'taxonomy_genus': r'$taxonomy_{genus}$',
    'taxonomy_species': r'$taxonomy_{species}$',
    'diff_phylogeny': r'$taxonomy_{different}$',
    'diff_phylo': r'$taxonomy_{different}$',
    'taxonomy_domain_0.4_clusters': r'$taxonomy_{domain}$',
    'taxonomy_kingdom_0.4_clusters': r'$taxonomy_{kingdom}$',
    'taxonomy_phylum_0.4_clusters': r'$taxonomy_{phylum}$',
    'taxonomy_class_0.4_clusters': r'$taxonomy_{class}$',
    'taxonomy_order_0.4_clusters': r'$taxonomy_{order}$',
    'taxonomy_family_0.4_clusters': r'$taxonomy_{family}$',
    'taxonomy_genus_0.4_clusters': r'$taxonomy_{genus}$',
    'taxonomy_species_0.4_clusters': r'$taxonomy_{species}$',
    'taxon_domain': r'$taxonomy_{domain}$',
    'taxon_kingdom': r'$taxonomy_{kingdom}$',
    'taxon_phylum': r'$taxonomy_{phylum}$',
    'taxon_class': r'$taxonomy_{class}$',
    'taxon_order': r'$taxonomy_{order}$',
    'taxon_family': r'$taxonomy_{family}$',
    'taxon_genus': r'$taxonomy_{genus}$',
    'taxon_species': r'$taxonomy_{species}$',
    
    # Other datasets
    'plddt': r'$pLDDT_{AlphaFold2}$',
    'realness': r'$Realness_{dataset}$',
}


CLS_PREFS: List[Tuple[str, str]] = [
    #("f1",        "F1"),
    ("mcc",       "MCC"),
    ("accuracy",  "Accuracy"),
]
REG_PREFS: List[Tuple[str, str]] = [
    ("spearman_rho",  "Spearman rho"),
    ("r_squared", "R²"),
    ("pearson",   "Pearson r"),
]


def is_regression(metrics: Dict[str, float]) -> bool:
    """Heuristic based on key names."""
    reg = ("spearman", "pearson", "r_squared", "rmse", "mse")
    cls = ("accuracy", "f1", "mcc", "auc", "precision", "recall")
    # Filter out time-related metrics
    filtered_metrics = {k: v for k, v in metrics.items() 
                       if 'training_time' not in k.lower() and 'time_seconds' not in k.lower()}
    keys = {k.lower() for k in filtered_metrics}
    if any(k for k in keys if any(r in k for r in reg)):
        return True
    if any(k for k in keys if any(c in k for c in cls)):
        return False
    return False  # default to classification


def pick_metric(metrics: Dict[str, float], prefs: List[Tuple[str, str]]) -> Tuple[str, str]:
    """Return (key, pretty_name) for the first preference present in metrics."""
    for k, nice in prefs:
        for mk in metrics:
            # Skip time-related metrics
            if 'training_time' in mk.lower() or 'time_seconds' in mk.lower():
                continue
            if mk.lower().endswith(k):
                return k, nice
    raise KeyError("No preferred metric found.")


def parse_metric_value(value) -> Tuple[float, float]:
    """
    Parse a metric value that may be in 'mean±std' format or a plain number.
    Returns (mean, std) where std is 0.0 if not present.
    """
    if isinstance(value, str) and '±' in value:
        parts = value.split('±')
        try:
            mean_val = float(parts[0])
            std_val = float(parts[1]) if len(parts) > 1 else 0.0
            return mean_val, std_val
        except ValueError:
            return math.nan, 0.0
    elif isinstance(value, (int, float)):
        return float(value), 0.0
    return math.nan, 0.0


def get_metric_value(metrics: Dict[str, float], key_suffix: str) -> float:
    """Fetch metric value case-/prefix-insensitively; NaN if absent.
    For mean±std format, returns only the mean value."""
    for k, v in metrics.items():
        # Skip time-related metrics and _mean/_std suffixed keys
        if 'training_time' in k.lower() or 'time_seconds' in k.lower():
            continue
        if k.lower().endswith('_mean') or k.lower().endswith('_std'):
            continue
        if k.lower().endswith(key_suffix):
            mean_val, _ = parse_metric_value(v)
            return mean_val
    return math.nan


def get_metric_value_with_std(metrics: Dict[str, float], key_suffix: str) -> Tuple[float, float, str]:
    """
    Fetch metric value with std case-/prefix-insensitively.
    Returns (mean, std, display_string) where display_string is formatted for heatmap display.
    """
    for k, v in metrics.items():
        # Skip time-related metrics and _mean/_std suffixed keys
        if 'training_time' in k.lower() or 'time_seconds' in k.lower():
            continue
        if k.lower().endswith('_mean') or k.lower().endswith('_std'):
            continue
        if k.lower().endswith(key_suffix):
            mean_val, std_val = parse_metric_value(v)
            if std_val > 0:
                display_str = f"{mean_val:.2f}±{std_val:.2f}"
            else:
                display_str = f"{mean_val:.2f}"
            return mean_val, std_val, display_str
    return math.nan, 0.0, ""


def radar_factory(n_axes: int):
    theta = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    return fig, ax, theta


def plot_radar(*,
               categories: List[str],
               models: List[str],
               data: List[List[float]],
               title: str,
               outfile: Path,
               normalize: bool = False):
    # Use pretty names for categories (datasets) and models
    pretty_categories = [DATASET_NAMES.get(cat, cat) for cat in categories]
    pretty_models = [MODEL_NAMES.get(m, m) for m in models]

    if normalize:
        arr = np.asarray(data)
        rng = np.where(np.ptp(arr, axis=0) == 0, 1, np.ptp(arr, axis=0))
        data = (arr - arr.min(0)) / rng
        # Convert back to list of lists for consistency
        data = data.tolist()

    # append mean column (do this after normalization if normalize=True)
    pretty_categories = pretty_categories + ["Avg"]
    data = [row + [np.nanmean(row)] for row in data]

    fig, ax, theta = radar_factory(len(pretty_categories))
    ax.set_thetagrids(np.degrees(theta), pretty_categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 11))

    palette = [plt.cm.tab20(i / len(pretty_models)) for i in range(len(pretty_models))]
    for i, (m, vals) in enumerate(zip(pretty_models, data)):
        ang = np.concatenate([theta, [theta[0]]])
        val = np.concatenate([vals,  [vals[0]]])
        ax.plot(ang, val, lw=2, label=m, color=palette[i])
        ax.fill(ang, val, alpha=.25, color=palette[i])

    ax.grid(True)
    plt.title(title, pad=20)
    plt.legend(bbox_to_anchor=(1.25, 1.05))
    plt.tight_layout()
    plt.savefig(outfile, dpi=450, bbox_inches="tight")
    plt.close(fig)


def bar_plot(datasets: List[str],
             models: List[str],
             data: List[List[float]],
             metric_name: str,
             outfile: Path):
    rows = [
        {"Dataset": DATASET_NAMES.get(d, d), "Model": MODEL_NAMES.get(m, m), "Score": s}
        for m, col in zip(models, data)
        for d, s in zip(datasets, col)
    ]
    dfp = pd.DataFrame(rows)
    plt.figure(figsize=(max(12, .8 * len(datasets)), 8))
    sns.barplot(dfp, x="Dataset", y="Score", hue="Model")
    plt.title(f"{metric_name} across datasets (Cls→F1, Reg→Spearman)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=450, bbox_inches="tight")
    plt.close()


def normalize_per_dataset(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array per dataset (row-wise).
    
    Args:
        arr: Array of shape (num_datasets, num_models)
    
    Returns:
        Normalized array of same shape, with each row normalized to [0, 1]
    """
    normalized_data = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        lowest_performance = np.nanmin(arr[i, :])
        best_performance = np.nanmax(arr[i, :])
        denom = best_performance - lowest_performance
        denom = 1 if denom == 0 else denom
        normalized_data[i, :] = (arr[i, :] - lowest_performance) / denom
    return normalized_data

def heatmap_plot(datasets: List[str],
                 models: List[str],
                 data: List[List[float]],
                 metric_name: str,
                 outfile: Path,
                 normalize: bool = False,
                 display_strings: List[List[str]] = None,
                 no_std: bool = False):
    """
    Create a heatmap plot.
    
    Args:
        datasets: List of dataset names
        models: List of model names
        data: List of lists of mean values (for coloring)
        metric_name: Name of the metric being plotted
        outfile: Output file path
        normalize: Whether to normalize display values
        display_strings: Optional list of lists of display strings (e.g., "0.85±0.01").
                        If provided, these are used for annotations instead of raw values.
    """
    arr = np.array(data).T  # shape: (num_datasets, num_models)
    # Compute average row (mean across datasets for each model)
    avg_row = np.nanmean(arr, axis=0, keepdims=True)
    arr_with_avg = np.vstack([arr, avg_row])
    datasets_plus_avg = datasets + ['Average']

    # Clean display names
    clean_model_names = [MODEL_NAMES.get(m, m) for m in models]
    clean_dataset_names = [DATASET_NAMES.get(d, d) for d in datasets_plus_avg]
    print(clean_dataset_names)
    print(datasets_plus_avg)

    # Build display string matrix if provided
    if display_strings is not None and not no_std:
        # Transpose to match arr shape: (num_datasets, num_models)
        display_arr = np.array(display_strings).T.tolist()
        # Add average row display strings
        avg_display = []
        for j in range(len(models)):
            model_vals = [arr[i, j] for i in range(arr.shape[0]) if not math.isnan(arr[i, j])]
            if model_vals:
                avg_display.append(f"{np.mean(model_vals):.4f}")
            else:
                avg_display.append("")
        display_arr.append(avg_display)
    else:
        display_arr = None

    # For annotations: use normalized or original values based on normalize parameter
    if normalize:
        # Normalize values for display in annotations
        normalized_data = normalize_per_dataset(arr)
        
        # Add average row to normalized data
        avg_row_norm = np.nanmean(normalized_data, axis=0, keepdims=True)
        annot_arr = np.vstack([normalized_data, avg_row_norm])
        annot_label = 'Normalized Performance (0-1)'
        # Don't use display_strings for normalized view
        display_arr = None
    else:
        annot_arr = arr_with_avg
        annot_label = metric_name

    # Always normalize colors per row (dataset) for visualization
    # This creates a color array where each row is scaled 0-1
    color_arr = np.zeros_like(arr_with_avg)
    for i in range(arr_with_avg.shape[0]):
        row_min = np.nanmin(arr_with_avg[i, :])
        row_max = np.nanmax(arr_with_avg[i, :])
        denom = row_max - row_min
        if denom == 0 or np.isnan(denom):
            color_arr[i, :] = 0.5  # neutral color if all values are the same
        else:
            color_arr[i, :] = (arr_with_avg[i, :] - row_min) / denom

    # Calculate figure size based on content
    # Increase cell width if we have mean±std strings or normalized values
    has_std = display_arr is not None
    #if normalize:
    #    cell_width = 1.1
    #elif has_std:
    #    cell_width = 1.1
    #else:
    #    cell_width = 0.85  # standard width for .2f format
    cell_width = 1.3
    cell_height = 0.7  # height per cell in inches
    
    fig_width = max(8, cell_width * len(clean_model_names))
    fig_height = max(6, cell_height * len(clean_dataset_names) + 1)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create heatmap with per-row normalized colors (blue = bad, orange = good)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#3498db', '#85c1e9', '#FFD700']  # Blue -> Light Blue -> Yellow
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('blue_yellow', colors, N=n_bins)
    
    im = ax.imshow(color_arr, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar with "Worst to Best" label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Worst to Best', fontsize=16)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Worst', 'Mid', 'Best'], fontsize=11)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(clean_model_names)))
    ax.set_yticks(np.arange(len(clean_dataset_names)))
    ax.set_xticklabels(clean_model_names, rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(clean_dataset_names, rotation=0, fontsize=16)
    
    # Add value annotations
    # Use smaller font if displaying mean±std
    font_size = 10 if has_std else 16
    for i in range(annot_arr.shape[0]):
        for j in range(annot_arr.shape[1]):
            if display_arr is not None and i < len(display_arr) and j < len(display_arr[i]):
                text_str = display_arr[i][j]
            else:
                if i == annot_arr.shape[0] - 1:  # Average row
                    text_str = f'{annot_arr[i, j]:.4f}'
                else:
                    text_str = f'{annot_arr[i, j]:.2f}'
            text = ax.text(j, i, text_str,
                          ha="center", va="center", color="black", fontsize=font_size)
    
    # Add black boxes around best performers in each row
    for i in range(color_arr.shape[0]):
        if not np.all(np.isnan(color_arr[i, :])):
            best_idx = np.nanargmax(color_arr[i, :])
            ax.add_patch(plt.Rectangle((best_idx - 0.5, i - 0.5), 1, 1, 
                                       fill=False, edgecolor='black', lw=3))
    
    # Set appropriate title
    if normalize:
        title = f'{annot_label} Heatmap (Cls→F1, Reg→Spearman)\nColors normalized per dataset'
    else:
        title = f'{annot_label} Heatmap (Cls→F1, Reg→Spearman)\nColors normalized per dataset'
        
    plt.title(title, pad=20, fontsize=21)
    plt.ylabel('Dataset', fontsize=17)
    plt.xlabel('Model', fontsize=17)
    plt.tight_layout()
    plt.savefig(outfile, dpi=450, bbox_inches='tight')
    plt.close()


def load_tsv(tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    for c in df.columns:
        if c != "dataset":
            df[c] = df[c].apply(json.loads)
    return df


def create_plots(tsv: str, outdir: str, no_std: bool = False):
    tsv, outdir = Path(tsv), Path(outdir)
    df = load_tsv(tsv)
    models = [c for c in df.columns if c != "dataset"]

    # Resolve metric per-dataset (MCC or R², w/ fallbacks).
    datasets, scores_by_model = [], {m: [] for m in models}
    display_by_model = {m: [] for m in models}  # For mean±std display strings
    dataset_types = []  # Track which type each dataset is

    for _, row in df.iterrows():
        name = row["dataset"]
        metrics0 = row[models[0]]
        task = "regression" if is_regression(metrics0) else "classification"
        dataset_types.append(task)
        prefs = REG_PREFS if task == "regression" else CLS_PREFS

        try:
            suffix, pretty = pick_metric(metrics0, prefs)
        except KeyError:
            print(f"[WARN] {name}: no suitable metric – skipped.")
            continue

        datasets.append(name)
        for m in models:
            if no_std:
                mean_val, std_val, display_str = get_metric_value_with_std(row[m], suffix)
                display_str = f"{mean_val:.2f}"
            else:
                mean_val, std_val, display_str = get_metric_value_with_std(row[m], suffix)
            scores_by_model[m].append(mean_val)
            display_by_model[m].append(display_str)

    if not datasets:
        raise RuntimeError("No plottable datasets found.")

    # Check if we have only one type of dataset
    only_classification = all(t == "classification" for t in dataset_types)
    only_regression = all(t == "regression" for t in dataset_types)

    # Order datasets according to DATASET_NAMES keys
    ordered_datasets = []
    ordered_scores = {m: [] for m in models}
    ordered_display = {m: [] for m in models}  # For mean±std display strings
    ordered_types = []  # Keep track of ordered dataset types
    
    # First add datasets that are in DATASET_NAMES in their defined order
    for ds in DATASET_NAMES.keys():
        if ds in datasets:
            idx = datasets.index(ds)
            ordered_datasets.append(ds)
            ordered_types.append(dataset_types[idx])
            for m in models:
                ordered_scores[m].append(scores_by_model[m][idx])
                ordered_display[m].append(display_by_model[m][idx])
    
    # Then add any remaining datasets that weren't in DATASET_NAMES
    for ds in datasets:
        if ds not in ordered_datasets:
            ordered_datasets.append(ds)
            idx = datasets.index(ds)
            ordered_types.append(dataset_types[idx])
            for m in models:
                ordered_scores[m].append(scores_by_model[m][idx])
                ordered_display[m].append(display_by_model[m][idx])
    
    # Replace original lists with ordered ones
    datasets = ordered_datasets
    scores_by_model = ordered_scores
    display_by_model = ordered_display
    dataset_types = ordered_types

    # assemble lists in model order
    plot_matrix = [scores_by_model[m] for m in models]
    display_matrix = [display_by_model[m] for m in models]

    # Sort models by average score (ascending: worst to best)
    model_avgs = [np.nanmean(scores) for scores in plot_matrix]
    sorted_indices = np.argsort(model_avgs)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_plot_matrix = [plot_matrix[i] for i in sorted_indices]
    sorted_display_matrix = [display_matrix[i] for i in sorted_indices]

    # For normalized heatmap, sort based on normalized averages
    arr_for_norm = np.array(plot_matrix).T  # shape: (num_datasets, num_models)
    normalized_data = normalize_per_dataset(arr_for_norm)
    # Calculate normalized averages for each model (column-wise mean)
    normalized_model_avgs = np.nanmean(normalized_data, axis=0)
    sorted_indices_norm = np.argsort(normalized_model_avgs)
    sorted_models_norm = [models[i] for i in sorted_indices_norm]
    sorted_plot_matrix_norm = [plot_matrix[i] for i in sorted_indices_norm]
    sorted_display_matrix_norm = [display_matrix[i] for i in sorted_indices_norm]

    fig_tag = tsv.stem
    outdir = outdir / fig_tag
    outdir.mkdir(parents=True, exist_ok=True)

    # File paths for all plot types
    radar_path = outdir / f"{fig_tag}_radar_all.png"
    radar_path_norm = outdir / f"{fig_tag}_radar_all_normalized.png"
    bar_path = outdir / f"{fig_tag}_bar_all.png"
    bar_path_norm = outdir / f"{fig_tag}_bar_all_normalized.png"
    heatmap_path = outdir / f"{fig_tag}_heatmap_all.png"
    heatmap_path_norm = outdir / f"{fig_tag}_heatmap_all_normalized.png"

    # Set subtitle and metric name based on dataset types
    if only_classification:
        subtitle = "Classification datasets plot F1"
        metric_name = "F1"
    elif only_regression:
        subtitle = "Regression datasets plot Spearman rho"
        metric_name = "Spearman rho"
    else:
        subtitle = "Classification datasets plot F1; Regression datasets plot Spearman rho"
        metric_name = "F1 / Spearman rho"
    
    # Radar plot keeps original order
    plot_radar(categories=datasets,
               models=models,
               data=plot_matrix,
               title=subtitle,
               outfile=radar_path,
               normalize=False)
    plot_radar(categories=datasets,
               models=models,
               data=plot_matrix,
               title=subtitle + " (Normalized)",
               outfile=radar_path_norm,
               normalize=True)
    # Bar and heatmap use sorted order
    bar_plot(datasets, sorted_models, sorted_plot_matrix, metric_name, bar_path)
    # Normalized bar plot
    # For bar plot normalization, use min-max per dataset (column-wise normalization)
    arr = np.asarray(sorted_plot_matrix)
    rng = np.where(np.ptp(arr, axis=0) == 0, 1, np.ptp(arr, axis=0))
    arr_norm = (arr - arr.min(0)) / rng
    bar_plot(datasets, sorted_models, arr_norm.tolist(), metric_name + " (Normalized)", bar_path_norm)
    # Heatmap - pass display strings for mean±std annotation
    heatmap_plot(datasets, sorted_models, sorted_plot_matrix, metric_name, heatmap_path, 
                 normalize=False, display_strings=sorted_display_matrix, no_std=no_std)
    # Normalized heatmap uses sorting based on normalized averages
    heatmap_plot(datasets, sorted_models_norm, sorted_plot_matrix_norm, metric_name, heatmap_path_norm, 
                 normalize=True, display_strings=sorted_display_matrix_norm, no_std=no_std)
    print(f"Radar saved to {radar_path}")
    print(f"Radar (normalized) saved to {radar_path_norm}")
    print(f"Bar   saved to {bar_path}")
    print(f"Bar (normalized) saved to {bar_path_norm}")
    print(f"Heatmap saved to {heatmap_path}")
    print(f"Heatmap (normalized) saved to {heatmap_path_norm}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate radar, bar, and heatmap plots for all datasets. Always saves both normalized and unnormalized versions.")
    ap.add_argument("--input", required=True, help="TSV file with metrics")
    ap.add_argument("--output_dir", default="plots", help="Directory for plots")
    ap.add_argument("--no_std", action="store_true", help="Do not display standard deviation in heatmap plots")
    args = ap.parse_args()

    create_plots(Path(args.input), Path(args.output_dir), no_std=args.no_std)
    print("Finished.")


if __name__ == "__main__":
    # py -m visualization.plot_result
    
    # Check if input file is provided and run main to generate plots
    import sys
    if "--input" in sys.argv:
        main()
    else:
        # --- TESTS FOR PLOTTING FUNCTIONS ---
        print("\nRunning plot function tests...")
        from pathlib import Path
        tmpdir = Path("plots/test_plots")
        tmpdir.mkdir(parents=True, exist_ok=True)
        # Dummy data
        categories = ["A", "B", "C"]
        models = ["Model1", "Model2"]
        data = [
            [0.8, 0.6, 0.7],
            [0.5, 0.9, 0.4],
        ]
        # Radar plot
        radar_path = tmpdir / "test_radar.png"
        plot_radar(categories=categories, models=models, data=data, title="Test Radar", outfile=radar_path)
        assert radar_path.exists(), "Radar plot not created!"
        print(f"Radar plot test passed: {radar_path}")
        # Normalized radar plot
        radar_path_norm = tmpdir / "test_radar_normalized.png"
        plot_radar(categories=categories, models=models, data=data, title="Test Radar (Normalized)", outfile=radar_path_norm, normalize=True)
        assert radar_path_norm.exists(), "Normalized radar plot not created!"
        print(f"Normalized radar plot test passed: {radar_path_norm}")
        # Bar plot
        bar_path = tmpdir / "test_bar.png"
        bar_plot(categories, models, data, "Test Metric", bar_path)
        assert bar_path.exists(), "Bar plot not created!"
        print(f"Bar plot test passed: {bar_path}")
        # Normalized bar plot
        arr = np.asarray(data)
        rng = np.where(np.ptp(arr, axis=0) == 0, 1, np.ptp(arr, axis=0))
        arr_norm = (arr - arr.min(0)) / rng
        bar_path_norm = tmpdir / "test_bar_normalized.png"
        bar_plot(categories, models, arr_norm.tolist(), "Test Metric (Normalized)", bar_path_norm)
        assert bar_path_norm.exists(), "Normalized bar plot not created!"
        print(f"Normalized bar plot test passed: {bar_path_norm}")
        # Heatmap plot
        heatmap_path = tmpdir / "test_heatmap.png"
        heatmap_plot(categories, models, data, "Test Metric", heatmap_path)
        assert heatmap_path.exists(), "Heatmap plot not created!"
        print(f"Heatmap plot test passed: {heatmap_path}")
        # Normalized heatmap plot
        heatmap_path_norm = tmpdir / "test_heatmap_normalized.png"
        heatmap_plot(categories, models, data, "Test Metric", heatmap_path_norm, normalize=True)
        assert heatmap_path_norm.exists(), "Normalized heatmap plot not created!"
        print(f"Normalized heatmap plot test passed: {heatmap_path_norm}")
        # Heatmap plot with mean±std display strings
        display_strings = [
            ["0.80±0.02", "0.60±0.01", "0.70±0.03"],
            ["0.50±0.05", "0.90±0.02", "0.40±0.01"],
        ]
        heatmap_path_std = tmpdir / "test_heatmap_with_std.png"
        heatmap_plot(categories, models, data, "Test Metric", heatmap_path_std, display_strings=display_strings)
        assert heatmap_path_std.exists(), "Heatmap with std not created!"
        print(f"Heatmap with std test passed: {heatmap_path_std}")
        print("All plot function tests passed!\n")