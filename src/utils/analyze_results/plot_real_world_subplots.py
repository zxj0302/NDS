import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ================ CHANGE IF NEEDED ================
# Dataset abbreviations mapping
dataset_abbr = {
    'Krogan-Extended': 'KE',
    'Abortion': 'AB',
    'BitcoinOTC': 'BO',
    'CollecTRI': 'CT',
    'WikiElection': 'WE',
    'WikiRfa': 'WR',
    'Election': 'EL',
    'RedditHyperlinks': 'RH',
    'Slashdot081106': 'S8',
    'Slashdot090216': 'S9',
    'Epinions': 'EP',
    'WikiPolitics': 'WP',
    'WikiTalk': 'WT',
    'WikiData': 'WD',
    'Amazon': 'AM',
    'Stackoverflow': 'SO'

    # 'Biogrid': 'BG',
    # 'Brexit': 'BX',
    # 'Collins': 'CL',
    # 'Gavin': 'GV',
    # 'Gun': 'GN',
    # 'Krogan-Core': 'KC',
    # 'Partisanship': 'PA',
    # 'Referendum': 'RF'
}

# Keep the same dataset order used by the LaTeX table.
# desired_runtime_order = ['CL', 'GV', 'KC', 'RF', 'KE', 'AB', 'BG', 'GN', 'BX', 'PA', 'EL', 'EP', 'WP', 'WD', 'AM', 'SO']
desired_runtime_order = ['KE', 'AB', 'BO', 'CT', 'WE', 'WR', 'EL', 'RH', 'S8', 'S9', 'EP', 'WP', 'WT', 'WD', 'AM', 'SO']

# Method name abbreviations for display and legend labels
method_abbr = {
    'DCSGreedy': 'DCSGreedy',
    'NEG_DSD': 'NEG-DSD',
    'CEP': 'ECP',
    # 'CEP_MIP_B': 'EM',
    # 'CEP_QPBO_MIP_B': 'EM w/ Q',
    # 'DCS_GREEDY_PRUNING_QPBO_MIP_CONSTRAIN_B': 'BAR-ID',
    # 'NEG_DSD_PRUNING_QPBO_MIP_CONSTRAIN_B': 'BAR-IN',
    # 'CEP_PRUNING_QPBO_MIP_CONSTRAIN_B': 'BAR'
}

excluded_methods = {'CEP_PRUNING_QPBO_MIP_B', 'CEP_MIP_B', 'CEP_QPBO_MIP_B', 'DCS_GREEDY_PRUNING_QPBO_MIP_CONSTRAIN_B', 'NEG_DSD_PRUNING_QPBO_MIP_CONSTRAIN_B', 'CEP_PRUNING_QPBO_MIP_CONSTRAIN_B'}

# Output file
save_path = 'results/figure/rw_subplots_compare.pdf'

# Figure size control; reduce width to make dataset groups appear closer together.
figure_size = (12, 4)

# Compress the x positions to reduce the gap between adjacent dataset groups.
dataset_spacing = 0.75

# Plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8.5
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# ==================================================


def is_excluded_method(algo_name):
    display_name = method_abbr.get(algo_name, algo_name.replace('_', '\\_'))
    return algo_name in excluded_methods or display_name in excluded_methods


def read_algorithm_result(json_path):
    """Return (time, status) where time is in seconds."""
    if not os.path.exists(json_path):
        return None, 'No File'

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        info = data.get('config', {}).get('info', '')
        if 'Fail' in info and 'Terminate' not in info:
            return None, 'Failed'

        time = data.get('time', None)
        return time, 'Success'
    except Exception as exc:
        return None, f'Error: {str(exc)}'


def compare_algorithms(base_path='./output/real-world'):
    base_path = Path(base_path)
    datasets = sorted([d.name for d in base_path.iterdir() if d.is_dir()])

    all_algorithms = set()
    for dataset in datasets:
        dataset_path = base_path / dataset
        for json_file in dataset_path.glob('*.json'):
            all_algorithms.add(json_file.stem)

    algorithms = sorted(all_algorithms)
    algorithms = [algo for algo in algorithms if algo in method_abbr]

    results = []
    for dataset in datasets:
        dataset_path = base_path / dataset
        row = {'Dataset': dataset}

        for algo in algorithms:
            json_path = dataset_path / f'{algo}.json'
            time, status = read_algorithm_result(json_path)

            if status == 'Success':
                row[f'{algo}_Time'] = time
            else:
                row[f'{algo}_Time'] = np.nan

        results.append(row)

    df = pd.DataFrame(results)
    columns = ['Dataset']
    for algo in algorithms:
        columns.append(f'{algo}_Time')
    df = df[columns]
    return df, algorithms


def get_ordered_datasets(existing_datasets):
    abbr_to_dataset = {v: k for k, v in dataset_abbr.items()}
    ordered = [abbr_to_dataset[abbr] for abbr in desired_runtime_order if abbr in abbr_to_dataset]
    existing = set(existing_datasets)
    return [dataset for dataset in ordered if dataset in existing]


def build_chart_data(df, algorithms, ordered_datasets):
    chart_data = {}
    for dataset in ordered_datasets:
        row = df[df['Dataset'] == dataset]
        if row.empty:
            continue
        chart_data[dataset] = [row[algo].values[0] for algo in algorithms]
    return chart_data


def compute_log_floor(chart_data, minimum=1e-1, scale=0.5):
    positive_values = []
    for dataset_values in chart_data.values():
        for value in dataset_values:
            if isinstance(value, (int, float)) and not pd.isna(value):
                value_ms = value * 1000.0
                if value_ms > 0:
                    positive_values.append(value_ms)

    if not positive_values:
        return minimum

    return max(min(positive_values) * scale, minimum)


def make_subplot_bars(ax, datasets, algorithms, chart_data, ylabel, is_time=False, legend=False, bar_bottom=0.0):
    x = np.arange(len(datasets)) * dataset_spacing
    num_methods = len(algorithms)
    bar_width = min(0.8 / max(num_methods, 1), 0.12)
    offsets = (np.arange(num_methods) - (num_methods - 1) / 2.0) * bar_width

    colors = ['darkred', 'black', 'gray']

    for i, algo in enumerate(algorithms):
        display_name = method_abbr.get(algo, algo.replace('_', '\\_'))
        values = []
        for dataset in datasets:
            raw_val = chart_data.get(dataset, [np.nan] * num_methods)[i]
            if isinstance(raw_val, (int, float)) and not pd.isna(raw_val):
                values.append(raw_val * 1000.0 if is_time else raw_val)
            else:
                values.append(np.nan)

        ax.bar(
            x + offsets[i],
            values,
            width=bar_width,
            bottom=bar_bottom if is_time else 0.0,
            label=display_name,
            color=colors[i % len(colors)],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.7
        )

    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.01)

    if is_time:
        ax.set_yscale('log')
        ax.set_ylim(bottom=bar_bottom)
        left_edge = x[0] + offsets.min() - bar_width * 1.5
        right_edge = x[-1] + offsets.max() + bar_width * 1.5
        ax.set_xlim(left_edge, right_edge)

    if legend:
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            ncol=1,
            framealpha=0.9,
            edgecolor='black',
            fontsize=11,
            labelspacing=0.5,
            handlelength=1.6,
            borderpad=0.8
        )


def main(base_path='./output/real-world'):
    df, algorithms = compare_algorithms(base_path)

    ordered_datasets = get_ordered_datasets(df['Dataset'])
    if not ordered_datasets:
        raise RuntimeError('No real-world datasets were found under output/real-world.')

    time_chart_data = build_chart_data(df.rename(columns={f'{algo}_Time': algo for algo in algorithms}), algorithms, ordered_datasets)
    time_floor = compute_log_floor(time_chart_data)

    fig, ax_time = plt.subplots(1, 1, figsize=figure_size, sharex=True)
    fig.patch.set_facecolor('white')
    ax_time.set_facecolor('white')

    make_subplot_bars(
        ax_time,
        ordered_datasets,
        algorithms,
        time_chart_data,
        ylabel='Time (ms)',
        is_time=True,
        legend=True,
        bar_bottom=time_floor
    )
    ax_time.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax_time.set_xticks(np.arange(len(ordered_datasets)) * dataset_spacing)
    ax_time.set_xticklabels([dataset_abbr.get(dataset, dataset) for dataset in ordered_datasets], rotation=0)

    for spine in ax_time.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

    print(f'Saved subplot figure to {output_path}')
    print(f'Datasets shown: {", ".join(dataset_abbr.get(dataset, dataset) for dataset in ordered_datasets)}')
    print(f'Methods shown: {", ".join(method_abbr.get(algo, algo) for algo in algorithms)}')


if __name__ == '__main__':
    main()