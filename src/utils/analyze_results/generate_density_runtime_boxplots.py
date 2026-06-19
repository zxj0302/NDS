import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


#================ CHANGE IF NEEDED ================
# Method name mapping (from internal names to display names)
METHOD_ABBR = {
    'DCSGreedy': 'DCSGreedy',
    'NEG_DSD': 'NEG-DSD',
    'CEP': 'ECP',
}

# Set publication-quality style
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

# Define which methods to show (in this order)
methods_to_show = ['DCSGreedy', 'NEG-DSD', 'ECP']
save_path = 'results/figure/density_runtime_boxplots.pdf'
#==================================================


def load_complete_tables(data_dir: str, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    display_to_internal = {v: k for k, v in METHOD_ABBR.items()}
    wanted_internal_methods = [display_to_internal[method] for method in methods if method in display_to_internal]
    columns_to_read = ['graph_name'] + [method for method in wanted_internal_methods if method != 'graph_name']

    df_density = pd.read_csv(f'{data_dir}/ER_density_table.csv', usecols=columns_to_read)
    df_time = pd.read_csv(f'{data_dir}/ER_time_table.csv', usecols=columns_to_read)

    df_density_complete = df_density.dropna(subset=wanted_internal_methods)
    df_time_complete = df_time.dropna(subset=wanted_internal_methods)

    counted_graphs = set(df_density_complete['graph_name']).intersection(set(df_time_complete['graph_name']))
    df_density_complete = df_density_complete[df_density_complete['graph_name'].isin(counted_graphs)]
    df_time_complete = df_time_complete[df_time_complete['graph_name'].isin(counted_graphs)]

    methods_present = [method for method in methods if method in METHOD_ABBR.values()]
    return df_density_complete, df_time_complete, methods_present, len(counted_graphs)


def prepare_long_table(df_complete: pd.DataFrame, value_name: str) -> pd.DataFrame:
    df_long = df_complete.melt(id_vars=['graph_name'], var_name='method', value_name=value_name)
    df_long['method_display'] = df_long['method'].map(METHOD_ABBR)
    return df_long[df_long['method_display'].notna()]


def build_boxplot_panel(ax, data_by_method, labels, ylabel, title, yscale='linear', reference_line=None):
    data_to_plot = [data_by_method[label] for label in labels]
    stats = []

    for values in data_to_plot:
        stats.append({
            'median': np.median(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75),
        })

    ax.boxplot(
        data_to_plot,
        patch_artist=True,
        showmeans=False,
        widths=0.5,
        medianprops=dict(color='darkred', linewidth=1.5),
        boxprops=dict(linewidth=1.2, facecolor='white', edgecolor='black'),
        whiskerprops=dict(linewidth=1.2, color='black'),
        capprops=dict(linewidth=1.2, color='black'),
        flierprops=dict(marker='o', markersize=4, markerfacecolor='gray', markeredgecolor='gray', alpha=0.5),
    )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    if reference_line is not None:
        ax.axhline(y=reference_line, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    if yscale == 'log':
        ax.set_yscale('log')

    for label, stat, pos in zip(labels, stats, range(1, len(stats) + 1)):
        ax.plot(
            pos,
            stat['mean'],
            marker='*',
            markersize=4,
            color='darkred',
            markeredgecolor='darkred',
            markeredgewidth=1.0,
            zorder=3,
        )

        y_range = stat['q3'] - stat['q1']
        if yscale == 'log':
            offset = max(stat['mean'] * 0.06, stat['mean'] * 0.02)
        elif y_range == 0:
            if reference_line is not None and abs(stat['mean'] - reference_line) < 1e-9:
                offset = 0.03 * max(1.0, abs(reference_line))
            else:
                offset = 0.05 * max(1.0, abs(stat['mean']))
        else:
            offset = y_range * 0.05 if y_range > 0 else 0.5

        ax.text(
            pos,
            stat['mean'] + offset,
            f"{stat['mean']:.2f}",
            ha='center',
            va='bottom',
            fontsize=9,
            color='darkred',
            fontweight='bold',
        )

    if yscale == 'log':
        positive_values = [value for values in data_to_plot for value in values if value > 0]
        if positive_values:
            y_min = min(positive_values)
            y_max = max(positive_values)
            ax.set_ylim([y_min / 1.5, y_max * 1.05])
    else:
        y_min = min(min(values) for values in data_to_plot)
        y_max = max(max(values) for values in data_to_plot)
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.5
        ax.set_ylim([y_min - padding, y_max + padding])

    legend_elements = [
        Line2D([0], [0], color='darkred', linewidth=1.5, label='Median'),
        Line2D([0], [0], marker='*', color='darkred', linestyle='None', markersize=8,
               markeredgecolor='darkred', markeredgewidth=1.0, label='Mean'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 1.0),
              ncol=1, framealpha=0.9, edgecolor='black')


def main():
    df_density_complete, df_time_complete, methods, counted_graphs = load_complete_tables(
        data_dir='results/synthetic/ER',
        methods=methods_to_show,
    )

    if not methods:
        raise RuntimeError('No requested methods were found in the ER result tables.')

    df_density_long = prepare_long_table(df_density_complete, 'density')
    df_time_long = prepare_long_table(df_time_complete, 'runtime')

    baseline_density = df_density_long[df_density_long['method_display'] == 'DCSGreedy'][
        ['graph_name', 'density']
    ].rename(columns={'density': 'baseline_density'})
    df_density_long = df_density_long.merge(baseline_density, on='graph_name', how='left')
    df_density_long['density_improvement_pct'] = (
        (df_density_long['density'] - df_density_long['baseline_density'])
        / df_density_long['baseline_density']
        * 100.0
    )
    df_density_long = df_density_long.replace([np.inf, -np.inf], np.nan).dropna(subset=['density_improvement_pct'])

    baseline_runtime = df_time_long[df_time_long['method_display'] == 'DCSGreedy'][
        ['graph_name', 'runtime']
    ].rename(columns={'runtime': 'baseline_runtime'})
    df_time_long = df_time_long.merge(baseline_runtime, on='graph_name', how='left')
    df_time_long['relative_runtime_ratio'] = (
        df_time_long['runtime'] / df_time_long['baseline_runtime']
    )
    df_time_long = df_time_long.replace([np.inf, -np.inf], np.nan).dropna(subset=['relative_runtime_ratio'])

    density_by_method = {method: df_density_long[df_density_long['method_display'] == method]['density_improvement_pct'].values for method in methods}
    runtime_by_method = {
        method: df_time_long[df_time_long['method_display'] == method]['relative_runtime_ratio'].values
        for method in methods
    }

    fig, (ax_runtime, ax_density) = plt.subplots(1, 2, figsize=(5.5, 3.5))

    build_boxplot_panel(
        ax_runtime,
        runtime_by_method,
        methods,
        ylabel='Relative Runtime over DCSGreedy (x)',
        title='(a) Runtime Ratio',
        yscale='linear',
        reference_line=1.0,
    )
    build_boxplot_panel(
        ax_density,
        density_by_method,
        methods,
        ylabel='Density Improvement over DCSGreedy (%)',
        title='(b) Density Improvement',
        yscale='linear',
        reference_line=0.0,
    )

    for ax in (ax_runtime, ax_density):
        ax.set_xlabel('Method', fontsize=10, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

    print(f"Boxplots saved as '{save_path}'")
    print(f'Counted graphs: {counted_graphs}')

    print('\nRuntime Summary Statistics:')
    print('-' * 80)
    for method in methods:
        values = runtime_by_method[method]
        print(
            f"{method:15s} | Median: {np.median(values):8.4f}x | "
            f"Mean: {np.mean(values):8.4f}x | Std: {np.std(values):8.4f}x"
        )

    print('\nDensity Summary Statistics:')
    print('-' * 80)
    for method in methods:
        values = density_by_method[method]
        print(
            f"{method:15s} | Median: {np.median(values):8.4f}% | "
            f"Mean: {np.mean(values):8.4f}% | Std: {np.std(values):8.4f}%"
        )

    plt.show()


if __name__ == '__main__':
    main()