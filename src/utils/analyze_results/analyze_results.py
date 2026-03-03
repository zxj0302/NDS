"""
Analyze and visualize results from synthetic graph experiments.
Compares runtime, density, and success rates across different methods and graph classes.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Method name abbreviations for display
METHOD_ABBR = {
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN': 'CQM-B',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_95': 'CQM-B-95',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_95_NB': 'CQM-D-95',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_90': 'CQM-B-90',
    'CEP_PRUNING_QPBO_CEP_MIP': 'CQM-B w/o Con w/ C1',
    'CEP_PRUNING_QPBO_MIP_NB': 'CQM-D w/o Con',
    'CEP_QPBO_MIP_NB': 'CQM-D w/o P+Con',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_NB': 'CQM-D',
    'CEP_PRUNING_QPBO_CEP_INIT_MIP_CONSTRAIN_CEP': 'CQM-B w/ C1+C2+I',
    'CEP_PRUNING_QPBO_CEP_MIP_CONSTRAIN_CEP': 'CQM-B w/ C1+C2',
    'CEP_PRUNING_QPBO_CEP_MIP_CONSTRAIN': 'CQM-B w/ C1',
    'CEP_MIP': 'MIQP-B',
    'CEP_MIP_NB': 'MIQP-D',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_99_NB': 'CQM-D-99',
    'CEP_PRUNING_QPBO_CEP_MIP_CONSTRAIN_NB': 'CQM-D w/ C1',
    'NEG_DSD': 'NEG-DSD',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_CEP': 'CQM-B w/ C2',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_999_NB': 'CQM-D-999',
    'CEP': 'CEP',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_99': 'CQM-B-99',
    'CEP_QPBO_MIP': 'CQM-B w/o P+Con',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_999': 'CQM-B-999',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_90_NB': 'CQM-D-90',
    'CEP_PRUNING_QPBO_MIP': 'CQM-B w/o Con',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_CEP_NB': 'CQM-D w/C2',
    'CEP_PRUNING_QPBO_CEP_MIP_CONSTRAIN_CEP_NB': 'CQM-D w/ C1+C2',
    'DCSGreedy': 'DCSGreedy',
    'CEP_L1': 'CEP-L1',
    'CEP_L5': 'CEP-L5',
    'CEP_L20': 'CEP-L20',
    'CEP_L50': 'CEP-L50',
    'CEP_K0': 'CEP-K0',
    'CEP_K50': 'CEP-K50',
    'CEP_K100': 'CEP-K100',
    'CEP_K500': 'CEP-K500'
}

def apply_method_abbr(df: pd.DataFrame) -> pd.DataFrame:
    """Apply method abbreviations to dataframe for display."""
    df = df.copy()
    if 'method' in df.columns:
        df['method'] = df['method'].map(METHOD_ABBR).fillna(df['method'])
    return df


def load_results(graph_class: str, results_dir: str = "results/synthetic") -> pd.DataFrame:
    """Load results for a specific graph class."""
    json_path = Path(results_dir) / graph_class / f"{graph_class}_results.json"
    
    if not json_path.exists():
        print(f"Warning: {json_path} does not exist")
        return pd.DataFrame()
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Verify we have complete data (all method-graph combinations)
    if not df.empty:
        n_graphs = df['graph_name'].nunique()
        n_methods = df['method'].nunique()
        expected_rows = n_graphs * n_methods
        if len(df) != expected_rows:
            print(f"Warning: {graph_class} data may be incomplete. "
                  f"Expected {expected_rows} rows, got {len(df)}")
    
    return df


def load_all_results(results_dir: str = "results/synthetic") -> pd.DataFrame:
    """Load results for all graph classes."""
    classes = ['BA', 'ER', 'RGG', 'SBM', 'WS']
    all_dfs = []
    
    for graph_class in classes:
        df = load_results(graph_class, results_dir)
        if not df.empty:
            df['class'] = graph_class
            all_dfs.append(df)
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def fill_cep_mip_from_longest_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill CEP_MIP data for graphs where all other methods succeeded but CEP_MIP failed or is missing.
    Uses the runtime and density from the method with the longest runtime.
    
    Args:
        df: DataFrame with experiment results
    
    Returns:
        DataFrame with CEP_MIP rows replaced/added where appropriate
    """
    df = df.copy()
    target_method = 'CEP_MIP'
    
    # Get all unique methods
    all_methods = set(df['method'].unique())
    
    # Check if CEP_MIP exists in the data
    if target_method not in all_methods:
        print(f"⚠️  {target_method} not found in methods, skipping fill")
        return df
    
    other_methods = all_methods - {target_method}
    replaced_count = 0
    
    # List to track indices to remove (failed CEP_MIP entries)
    indices_to_remove = []
    new_rows = []
    
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        
        for graph_name in class_df['graph_name'].unique():
            graph_df = class_df[class_df['graph_name'] == graph_name]
            
            # Check CEP_MIP status for this graph
            cep_mip_row = graph_df[graph_df['method'] == target_method]
            cep_mip_failed = len(cep_mip_row) > 0 and cep_mip_row['status'].iloc[0] != 'success'
            cep_mip_missing = len(cep_mip_row) == 0
            
            # Only process if CEP_MIP failed or is missing
            if cep_mip_failed or cep_mip_missing:
                # Get results for other methods
                other_methods_df = graph_df[graph_df['method'] != target_method]
                methods_present = set(other_methods_df['method'].unique())
                
                # Check if all other methods are present and successful
                if other_methods.issubset(methods_present):
                    successful_df = other_methods_df[other_methods_df['status'] == 'success']
                    
                    # Only fill if all other methods succeeded
                    if len(successful_df) == len(other_methods):
                        # Find the method with longest runtime
                        if 'time' in successful_df.columns and len(successful_df) > 0:
                            longest_runtime_row = successful_df.loc[successful_df['time'].idxmax()]
                            
                            # If CEP_MIP failed, mark its row for removal
                            if cep_mip_failed:
                                indices_to_remove.append(cep_mip_row.index[0])
                            
                            # Create new row for CEP_MIP with same runtime and density
                            new_row = longest_runtime_row.copy()
                            new_row['method'] = target_method
                            new_row['status'] = 'filled'  # Mark as filled data
                            
                            new_rows.append(new_row)
                            replaced_count += 1
    
    # Remove failed CEP_MIP entries
    if indices_to_remove:
        df = df.drop(indices_to_remove)
    
    # Add new filled entries
    if new_rows:
        new_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"✓ Filled {replaced_count} CEP_MIP entries from longest runtime methods")
        return new_df
    else:
        print("ℹ️  No CEP_MIP entries needed to be filled")
        return df


def get_complete_graphs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only include graphs where ALL methods have successful results.
    Only counts actual 'success' status - excludes 'filled' data.
    
    Returns:
        DataFrame containing only graphs with complete results from all methods
    """
    complete_graphs = []
    
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        total_methods = class_df['method'].nunique()
        
        for graph_name in class_df['graph_name'].unique():
            graph_df = class_df[class_df['graph_name'] == graph_name]
            # Only count actual success, not filled data
            successful_methods = (graph_df['status'] == 'success').sum()
            
            # Only include if all methods succeeded on this graph
            if successful_methods == total_methods:
                complete_graphs.append(graph_name)
    
    # Filter original dataframe to only these graphs
    return df[df['graph_name'].isin(complete_graphs)].copy()


def calculate_success_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate success rate for each method in each class.
    Note: Excludes 'filled' entries from success rate calculation.
    Filled entries are counted in total but not as successful (to show original performance).
    
    Returns:
        DataFrame with columns: class, method, total_graphs, successful, failed, success_rate
    """
    results = []
    
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        
        for method in class_df['method'].unique():
            method_df = class_df[class_df['method'] == method]
            
            # Count all entries including filled
            total = len(method_df)
            # Only count original successes
            successful = (method_df['status'] == 'success').sum()
            # Failed includes both failed and filled (non-original results)
            failed = total - successful
            success_rate = (successful / total * 100) if total > 0 else 0
            
            results.append({
                'class': class_name,
                'method': method,
                'total_graphs': total,
                'successful': successful,
                'failed': failed,
                'success_rate': success_rate
            })
    
    return pd.DataFrame(results)


def calculate_runtime_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate runtime statistics for successful runs.
    Note: Should be called with complete graphs only (where all methods succeeded).
    Only uses actual 'success' status - excludes 'filled' data.
    
    Returns:
        DataFrame with columns: class, method, avg_time, total_time, min_time, max_time, std_time
    """
    results = []
    
    # Filter only successful runs with valid time (excluding 'filled')
    success_df = df[(df['status'] == 'success') & (df['time'].notna())]
    
    for class_name in success_df['class'].unique():
        class_df = success_df[success_df['class'] == class_name]
        
        for method in class_df['method'].unique():
            method_df = class_df[class_df['method'] == method]
            
            if len(method_df) > 0:
                times = method_df['time']
                
                results.append({
                    'class': class_name,
                    'method': method,
                    'avg_time': times.mean(),
                    'total_time': times.sum(),
                    'min_time': times.min(),
                    'max_time': times.max(),
                    'std_time': times.std(),
                    'count': len(times)
                })
    
    return pd.DataFrame(results)


def calculate_runtime_speedup(df: pd.DataFrame, baseline_method: str = 'CEP_MIP') -> pd.DataFrame:
    """
    Calculate runtime speedup compared to baseline method (CEP_MIP).
    Only includes graphs where the baseline method succeeds.
    Only uses actual 'success' status - excludes 'filled' data.
    
    Returns:
        DataFrame with columns: class, method, avg_speedup, graphs_count
    """
    results = []
    
    # Filter only successful runs with valid time (excluding 'filled')
    success_df = df[(df['status'] == 'success') & (df['time'].notna())]
    
    for class_name in success_df['class'].unique():
        class_df = success_df[success_df['class'] == class_name]
        
        # Get graphs where baseline method succeeds
        baseline_df = class_df[class_df['method'] == baseline_method]
        valid_graphs = set(baseline_df['graph_name'].unique())
        
        if len(valid_graphs) == 0:
            continue
        
        # Filter to only these graphs
        class_df_filtered = class_df[class_df['graph_name'].isin(valid_graphs)]
        
        for method in class_df_filtered['method'].unique():
            method_df = class_df_filtered[class_df_filtered['method'] == method]
            
            if len(method_df) == 0:
                continue
            
            # Calculate speedup for each graph
            speedups = []
            for graph_name in valid_graphs:
                graph_method_df = method_df[method_df['graph_name'] == graph_name]
                graph_baseline_df = baseline_df[baseline_df['graph_name'] == graph_name]
                
                if len(graph_method_df) > 0 and len(graph_baseline_df) > 0:
                    method_time = graph_method_df['time'].iloc[0]
                    baseline_time = graph_baseline_df['time'].iloc[0]
                    
                    if method_time > 0:
                        speedup = baseline_time / method_time
                        speedups.append(speedup)
            
            if len(speedups) > 0:
                results.append({
                    'class': class_name,
                    'method': method,
                    'avg_speedup': np.mean(speedups),
                    'median_speedup': np.median(speedups),
                    'min_speedup': np.min(speedups),
                    'max_speedup': np.max(speedups),
                    'graphs_count': len(speedups)
                })
    
    return pd.DataFrame(results)


def calculate_density_improvement(df: pd.DataFrame, baseline_method: str = 'CEP') -> pd.DataFrame:
    """
    Calculate density improvement compared to baseline method.
    Note: Should be called with complete graphs only (where all methods succeeded).
    Only uses actual 'success' status - excludes 'filled' data.
    
    Returns:
        DataFrame with density comparison and improvement percentage
    """
    results = []
    
    # Filter only successful runs with valid density (excluding 'filled')
    success_df = df[(df['status'] == 'success') & (df['density'].notna())]
    
    for class_name in success_df['class'].unique():
        class_df = success_df[success_df['class'] == class_name]
        
        for graph_name in class_df['graph_name'].unique():
            graph_df = class_df[class_df['graph_name'] == graph_name]
            
            # Get baseline density
            baseline_df = graph_df[graph_df['method'] == baseline_method]
            if len(baseline_df) == 0:
                continue
            
            baseline_density = baseline_df['density'].iloc[0]
            
            # Calculate improvement for each method
            for _, row in graph_df.iterrows():
                method = row['method']
                density = row['density']
                
                # Calculate improvement (assuming higher density is better)
                if baseline_density != 0:
                    improvement = ((density - baseline_density) / abs(baseline_density)) * 100
                else:
                    improvement = 0 if density == 0 else float('inf')
                
                results.append({
                    'class': class_name,
                    'graph_name': graph_name,
                    'method': method,
                    'density': density,
                    'baseline_density': baseline_density,
                    'improvement_pct': improvement
                })
    
    return pd.DataFrame(results)


def calculate_density_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average density rank for each method.
    Note: Should be called with complete graphs only (where all methods succeeded).
    Only uses actual 'success' status - excludes 'filled' data.
    Rank 1 = highest density (best), higher rank = lower density.
    
    Returns:
        DataFrame with columns: class, method, avg_rank, std_rank
    """
    results = []
    
    # Filter only successful runs with valid density (excluding 'filled')
    success_df = df[(df['status'] == 'success') & (df['density'].notna())]
    
    for class_name in success_df['class'].unique():
        class_df = success_df[success_df['class'] == class_name]
        
        # Calculate ranks for each graph
        method_ranks = {}
        
        for graph_name in class_df['graph_name'].unique():
            graph_df = class_df[class_df['graph_name'] == graph_name]
            
            # Assign ranks using pandas rank method (handles ties by giving same rank)
            # method='min' gives tied values the minimum rank
            # ascending=False because higher density is better
            graph_df = graph_df.copy()
            graph_df['rank'] = graph_df['density'].rank(ascending=False, method='min')
            
            # Store ranks for each method
            for _, row in graph_df.iterrows():
                method = row['method']
                rank = row['rank']
                
                if method not in method_ranks:
                    method_ranks[method] = []
                method_ranks[method].append(rank)
        
        # Calculate average rank for each method
        for method, ranks in method_ranks.items():
            results.append({
                'class': class_name,
                'method': method,
                'avg_rank': np.mean(ranks),
                'median_rank': np.median(ranks),
                'std_rank': np.std(ranks),
                'best_rank_count': sum(1 for r in ranks if r == 1),
                'graphs_count': len(ranks)
            })
    
    return pd.DataFrame(results)


def export_success_rate_table(success_df: pd.DataFrame, save_dir: str = "results/synthetic"):
    """Export success rate as formatted tables."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create pivot table with success rates
    pivot_rate = success_df.pivot(index='method', columns='class', values='success_rate')
    pivot_rate = pivot_rate.round(2)
    
    # Save as CSV
    csv_path = f"{save_dir}/success_rate_table.csv"
    pivot_rate.to_csv(csv_path)
    
    # Create formatted text table
    txt_path = f"{save_dir}/success_rate_table.txt"
    with open(txt_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("SUCCESS RATE TABLE (%)\n")
        f.write("="*100 + "\n\n")
        
        # Write header
        classes = sorted(pivot_rate.columns)
        f.write(f"{'Method':<50s}")
        for c in classes:
            f.write(f"{c:>12s}")
        f.write("\n")
        f.write("-"*100 + "\n")
        
        # Write data
        for method in pivot_rate.index:
            f.write(f"{method:<50s}")
            for c in classes:
                value = pivot_rate.loc[method, c]
                if pd.notna(value):
                    f.write(f"{value:>12.2f}")
                else:
                    f.write(f"{'N/A':>12s}")
            f.write("\n")
        
        f.write("\n")
        
        # Add summary with counts
        f.write("\n" + "="*100 + "\n")
        f.write("SUCCESS COUNTS (successful/total)\n")
        f.write("="*100 + "\n\n")
        
        pivot_counts = success_df.pivot(index='method', columns='class', values='successful')
        pivot_total = success_df.pivot(index='method', columns='class', values='total_graphs')
        
        f.write(f"{'Method':<50s}")
        for c in classes:
            f.write(f"{c:>12s}")
        f.write("\n")
        f.write("-"*100 + "\n")
        
        for method in pivot_counts.index:
            f.write(f"{method:<50s}")
            for c in classes:
                if pd.notna(pivot_counts.loc[method, c]):
                    succ = int(float(pivot_counts.loc[method, c]))
                    total = int(float(pivot_total.loc[method, c]))
                    f.write(f"{succ:>5d}/{total:<5d}")
                else:
                    f.write(f"{'N/A':>12s}")
            f.write("\n")
    
    print(f"✓ Success rate tables saved to {save_dir}")
    print(f"  - {csv_path}")
    print(f"  - {txt_path}")


def export_speedup_table(speedup_df: pd.DataFrame, baseline_method: str = 'CEP_MIP',
                        save_dir: str = "results/synthetic"):
    """Export runtime speedup table."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if speedup_df.empty:
        print("⚠️  No speedup data to export")
        return
    
    # Filter out baseline method
    speedup_df = speedup_df[speedup_df['method'] != baseline_method].copy()
    
    # Create pivot table
    pivot_speedup = speedup_df.pivot(index='method', columns='class', values='avg_speedup')
    pivot_speedup = pivot_speedup.round(2)
    
    # Save as CSV
    csv_path = f"{save_dir}/speedup_table.csv"
    pivot_speedup.to_csv(csv_path)
    
    # Create formatted text table
    txt_path = f"{save_dir}/speedup_table.txt"
    with open(txt_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"RUNTIME SPEEDUP vs {baseline_method} (x faster)\n")
        f.write("Note: Calculated only on graphs where CEP_MIP succeeds\n")
        f.write("="*100 + "\n\n")
        
        # Write header
        classes = sorted(pivot_speedup.columns)
        f.write(f"{'Method':<50s}")
        for c in classes:
            f.write(f"{c:>12s}")
        f.write("\n")
        f.write("-"*100 + "\n")
        
        # Write data sorted by average speedup
        for method in pivot_speedup.index:
            f.write(f"{method:<50s}")
            for c in classes:
                value = pivot_speedup.loc[method, c]
                if pd.notna(value):
                    f.write(f"{value:>12.2f}")
                else:
                    f.write(f"{'N/A':>12s}")
            f.write("\n")
    
    print(f"✓ Speedup tables saved to {save_dir}")
    print(f"  - {csv_path}")
    print(f"  - {txt_path}")


def export_rank_table(rank_df: pd.DataFrame, save_dir: str = "results/synthetic"):
    """Export density rank table."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if rank_df.empty:
        print("⚠️  No rank data to export")
        return
    
    # Create pivot table for average rank
    pivot_rank = rank_df.pivot(index='method', columns='class', values='avg_rank')
    pivot_rank = pivot_rank.round(2)
    
    # Save as CSV
    csv_path = f"{save_dir}/density_rank_table.csv"
    pivot_rank.to_csv(csv_path)
    
    # Create formatted text table
    txt_path = f"{save_dir}/density_rank_table.txt"
    with open(txt_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("AVERAGE DENSITY RANK (1=best, lower is better)\n")
        f.write("Note: Calculated only on graphs where ALL methods succeed\n")
        f.write("="*100 + "\n\n")
        
        # Write header
        classes = sorted(pivot_rank.columns)
        f.write(f"{'Method':<50s}")
        for c in classes:
            f.write(f"{c:>12s}")
        f.write("\n")
        f.write("-"*100 + "\n")
        
        # Write data sorted by average rank
        for method in pivot_rank.index:
            f.write(f"{method:<50s}")
            for c in classes:
                value = pivot_rank.loc[method, c]
                if pd.notna(value):
                    f.write(f"{value:>12.2f}")
                else:
                    f.write(f"{'N/A':>12s}")
            f.write("\n")
        
        # Add section with #1 rank counts
        f.write("\n\n" + "="*100 + "\n")
        f.write("NUMBER OF TIMES RANKED #1 (highest density)\n")
        f.write("="*100 + "\n\n")
        
        pivot_best = rank_df.pivot(index='method', columns='class', values='best_rank_count')
        
        f.write(f"{'Method':<50s}")
        for c in classes:
            f.write(f"{c:>12s}")
        f.write("\n")
        f.write("-"*100 + "\n")
        
        for method in pivot_best.index:
            f.write(f"{method:<50s}")
            for c in classes:
                value = pivot_best.loc[method, c]
                if pd.notna(value):
                    f.write(f"{int(float(value)):>12d}")
                else:
                    f.write(f"{'N/A':>12s}")
            f.write("\n")
    
    print(f"✓ Density rank tables saved to {save_dir}")
    print(f"  - {csv_path}")
    print(f"  - {txt_path}")


def plot_success_rates(success_df: pd.DataFrame, save_dir: str = "results/synthetic/figures"):
    """Plot success rates for each method across classes."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create pivot table for heatmap
    pivot = success_df.pivot(index='method', columns='class', values='success_rate')
    
    # Plot 1: Heatmap of success rates
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                cbar_kws={'label': 'Success Rate (%)'})
    plt.title('Success Rate by Method and Graph Class (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Graph Class', fontsize=12)
    plt.ylabel('Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/success_rates_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    classes = success_df['class'].unique()
    methods = success_df['method'].unique()
    x = np.arange(len(classes))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        method_data = success_df[success_df['method'] == method]
        rates = [method_data[method_data['class'] == c]['success_rate'].values[0] 
                 if len(method_data[method_data['class'] == c]) > 0 else 0
                 for c in classes]
        ax.bar(x + i * width, rates, width, label=method)
    
    ax.set_xlabel('Graph Class', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Comparison Across Graph Classes', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/success_rates_bars.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Success rate plots saved to {save_dir}")


def plot_runtime_comparison(runtime_df: pd.DataFrame, save_dir: str = "results/synthetic/figures"):
    """Plot runtime comparison across methods and classes."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    classes = sorted(runtime_df['class'].unique())
    n_classes = len(classes)
    
    # Plot 1: Average runtime comparison - separate subplot for each class
    fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5), sharey=False)
    if n_classes == 1:
        axes = [axes]
    
    for idx, class_name in enumerate(classes):
        class_data = runtime_df[runtime_df['class'] == class_name].sort_values('avg_time')
        
        colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(class_data)))
        axes[idx].bar(range(len(class_data)), class_data['avg_time'], color=colors)
        axes[idx].set_xticks(range(len(class_data)))
        axes[idx].set_xticklabels(class_data['method'], rotation=45, ha='right')
        axes[idx].set_ylabel('Average Runtime (seconds)', fontsize=11)
        axes[idx].set_title(f'{class_name}', fontsize=13, fontweight='bold')
        axes[idx].set_yscale('log')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (_, row) in enumerate(class_data.iterrows()):
            height = row['avg_time']
            axes[idx].text(i, height, f'{height:.2f}', 
                          ha='center', va='bottom', fontsize=8)
    
    fig.suptitle('Average Runtime Comparison by Class (log scale)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/runtime_avg_by_class.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Total runtime comparison - separate subplot for each class
    fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5), sharey=False)
    if n_classes == 1:
        axes = [axes]
    
    for idx, class_name in enumerate(classes):
        class_data = runtime_df[runtime_df['class'] == class_name].sort_values('total_time')
        
        colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(class_data)))
        axes[idx].bar(range(len(class_data)), class_data['total_time'], color=colors)
        axes[idx].set_xticks(range(len(class_data)))
        axes[idx].set_xticklabels(class_data['method'], rotation=45, ha='right')
        axes[idx].set_ylabel('Total Runtime (seconds)', fontsize=11)
        axes[idx].set_title(f'{class_name}', fontsize=13, fontweight='bold')
        axes[idx].set_yscale('log')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (_, row) in enumerate(class_data.iterrows()):
            height = row['total_time']
            axes[idx].text(i, height, f'{height:.1f}', 
                          ha='center', va='bottom', fontsize=8)
    
    fig.suptitle('Total Runtime Comparison by Class (log scale)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/runtime_total_by_class.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Heatmap of average runtime for overview
    pivot_avg = runtime_df.pivot(index='method', columns='class', values='avg_time')
    plt.figure(figsize=(10, 8))
    pivot_avg_log = np.log10(pivot_avg + 1)  # Log scale for heatmap
    pivot_avg_formatted = pivot_avg.map(lambda x: f'{x:.2f}')
    sns.heatmap(pivot_avg_log, annot=pivot_avg_formatted, 
                fmt='', cmap='YlOrRd', cbar_kws={'label': 'log10(Average Runtime + 1)'})
    plt.title('Average Runtime by Method and Class (log scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Graph Class', fontsize=12)
    plt.ylabel('Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/runtime_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Runtime ranking within each class
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ranking_data = []
    for class_name in classes:
        class_data = runtime_df[runtime_df['class'] == class_name].sort_values('avg_time')
        for rank, (_, row) in enumerate(class_data.iterrows(), 1):
            ranking_data.append({
                'class': class_name,
                'method': row['method'],
                'rank': rank
            })
    
    ranking_df = pd.DataFrame(ranking_data)
    ranking_pivot = ranking_df.pivot(index='method', columns='class', values='rank')
    
    sns.heatmap(ranking_pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Rank (1=fastest)'}, vmin=1)
    plt.title('Runtime Ranking by Method and Class (1=fastest)', fontsize=14, fontweight='bold')
    plt.xlabel('Graph Class', fontsize=12)
    plt.ylabel('Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/runtime_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Runtime plots saved to {save_dir}")


def plot_density_improvement(improvement_df: pd.DataFrame, baseline_method: str = 'CEP',
                            save_dir: str = "results/synthetic/figures"):
    """Plot density improvement compared to baseline."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate average improvement per method per class
    avg_improvement = improvement_df[improvement_df['method'] != baseline_method].groupby(
        ['class', 'method']
    )['improvement_pct'].agg(['mean', 'std', 'count']).reset_index()
    
    # Plot 1: Average improvement by method and class
    fig, ax = plt.subplots(figsize=(14, 6))
    
    classes = avg_improvement['class'].unique()
    methods = [m for m in avg_improvement['method'].unique() if m != baseline_method]
    x = np.arange(len(classes))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        method_data = avg_improvement[avg_improvement['method'] == method]
        improvements = [method_data[method_data['class'] == c]['mean'].values[0] 
                       if len(method_data[method_data['class'] == c]) > 0 else 0
                       for c in classes]
        ax.bar(x + i * width, improvements, width, label=method)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Graph Class', fontsize=12)
    ax.set_ylabel(f'Density Improvement vs {baseline_method} (%)', fontsize=12)
    ax.set_title(f'Average Density Improvement Compared to {baseline_method}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/density_improvement_bars.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Box plot of improvement distribution (separate figures for each class)
    # Define professional color palette
    color_positive = '#2ecc71'  # Green for positive improvement
    color_negative = '#e74c3c'  # Red for negative improvement
    color_avg_line = '#3498db'  # Blue for average line
    color_baseline = '#34495e'  # Dark gray for baseline
    
    for class_name in classes:
        class_data = improvement_df[
            (improvement_df['class'] == class_name) & 
            (improvement_df['method'] != baseline_method)
        ].copy()
        
        if len(class_data) == 0:
            continue
        
        # Calculate average improvement per method and sort
        method_avg = class_data.groupby('method')['improvement_pct'].mean().sort_values()
        methods = method_avg.index.tolist()
        
        # Create publication-quality figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        # Prepare data for boxplot
        boxplot_data = []
        avg_improvements = []
        
        for method in methods:
            method_data = class_data[class_data['method'] == method]['improvement_pct']
            boxplot_data.append(method_data.values)
            avg_improvements.append(method_data.mean())
        
        # Create boxplot with enhanced styling
        bp = ax.boxplot(boxplot_data, patch_artist=True, vert=False,
                       widths=0.6,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(color='#2c3e50', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='gray', 
                                     markersize=5, alpha=0.5))
        
        ax.set_yticklabels(methods, fontsize=11)
        
        # Color boxes based on whether improvement is positive
        for i, (patch, avg) in enumerate(zip(bp['boxes'], avg_improvements)):
            if avg > 0:
                color = color_positive
                alpha = 0.6
            else:
                color = color_negative
                alpha = 0.6
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor('#2c3e50')
        
        # Add average improvement curve with secondary axis
        ax2 = ax.twiny()
        y_positions = range(1, len(methods) + 1)
        
        # Plot the average line with enhanced styling
        line = ax2.plot(avg_improvements, y_positions, 
                       color=color_avg_line, linewidth=2.5, 
                       marker='D', markersize=8, 
                       label='Average Improvement',
                       markerfacecolor=color_avg_line,
                       markeredgecolor='white',
                       markeredgewidth=1.5,
                       alpha=0.9, zorder=10)
        
        ax2.set_xlabel('Average Improvement (%)', fontsize=13, fontweight='bold', 
                      color=color_avg_line, labelpad=10)
        ax2.tick_params(axis='x', labelcolor=color_avg_line, labelsize=11, width=1.5)
        ax2.set_ylim(ax.get_ylim())
        ax2.spines['top'].set_color(color_avg_line)
        ax2.spines['top'].set_linewidth(2)
        
        # Style the baseline reference line
        ax.axvline(x=0, color=color_baseline, linestyle='--', 
                  linewidth=2, alpha=0.7, label='Baseline (0%)', zorder=1)
        
        # Enhanced axis labels and title
        ax.set_xlabel('Density Improvement (%)', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel('Method', fontsize=13, fontweight='bold', labelpad=10)
        # ax.set_title(f'{class_name} Graph Class', fontsize=16, fontweight='bold', pad=20)
        
        # Improve grid
        ax.grid(axis='x', alpha=0.25, linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        
        # Style the spines
        for spine in ['bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(1.5)
            ax.spines[spine].set_color('#2c3e50')
        ax.spines['right'].set_visible(False)
        
        # Improve tick styling
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1, length=3)
        
        # Add legend with better positioning and styling
        legend1 = ax.legend(loc='lower right', frameon=True, fancybox=True, 
                          shadow=True, fontsize=11, framealpha=0.95)
        legend1.get_frame().set_facecolor('white')
        legend1.get_frame().set_edgecolor('#2c3e50')
        legend1.get_frame().set_linewidth(1.5)
        
        # Add a subtle text note
        note_text = f'Baseline: {baseline_method}'
        ax.text(0.02, 0.98, note_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/density_improvement_vs_{baseline_method}_{class_name}.pdf", 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    # Plot 3: Heatmap of average improvement
    pivot_improvement = avg_improvement.pivot(index='method', columns='class', values='mean')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_improvement, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Improvement (%)'})
    plt.title(f'Average Density Improvement vs {baseline_method} (%)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Graph Class', fontsize=12)
    plt.ylabel('Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/density_improvement_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Density improvement plots saved to {save_dir}")


def plot_runtime_speedup(speedup_df: pd.DataFrame, baseline_method: str = 'CEP_MIP',
                        save_dir: str = "results/synthetic/figures"):
    """Plot runtime speedup compared to baseline method."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if speedup_df.empty:
        print("⚠️  No speedup data to plot")
        return
    
    classes = sorted(speedup_df['class'].unique())
    
    # Plot 1: Boxplot for each class (separate figures)
    for class_name in classes:
        class_data = speedup_df[speedup_df['class'] == class_name].copy()
        
        # Filter out baseline (speedup = 1.0)
        class_data = class_data[class_data['method'] != baseline_method]
        
        if len(class_data) == 0:
            continue
        
        # Sort methods by average speedup for better visualization
        class_data = class_data.sort_values('avg_speedup')
        
        # Create figure with publication-quality settings
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        # Prepare data for boxplot
        methods = class_data['method'].tolist()
        boxplot_data = []
        avg_speedups = []
        
        for method in methods:
            method_data = class_data[class_data['method'] == method]
            if len(method_data) > 0:
                # Collect all speedup values for this method
                speedups = [method_data['avg_speedup'].iloc[0]]
                if 'median_speedup' in method_data.columns:
                    speedups.append(method_data['median_speedup'].iloc[0])
                if 'min_speedup' in method_data.columns:
                    speedups.append(method_data['min_speedup'].iloc[0])
                if 'max_speedup' in method_data.columns:
                    speedups.append(method_data['max_speedup'].iloc[0])
                
                boxplot_data.append(speedups)
                avg_speedups.append(method_data['avg_speedup'].iloc[0])
        
        # Define professional color palette
        color_faster = '#2ecc71'  # Green for faster (speedup > 1)
        color_slower = '#e74c3c'  # Red for slower (speedup < 1)
        color_avg_line = '#3498db'  # Blue for average line
        color_baseline = '#34495e'  # Dark gray for baseline
        
        # Create boxplot with enhanced styling
        bp = ax.boxplot(boxplot_data, patch_artist=True, vert=False,
                       widths=0.6,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(color='#2c3e50', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='gray', 
                                     markersize=5, alpha=0.5))
        
        ax.set_yticklabels(methods, fontsize=11)
        
        # Color boxes based on whether speedup > 1 with gradient effect
        for i, (patch, avg) in enumerate(zip(bp['boxes'], avg_speedups)):
            if avg > 1:
                color = color_faster
                alpha = 0.6
            else:
                color = color_slower
                alpha = 0.6
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor('#2c3e50')
        
        # Add average speedup curve with secondary axis
        ax2 = ax.twiny()
        y_positions = range(1, len(methods) + 1)
        
        # Plot the average line with enhanced styling
        line = ax2.plot(avg_speedups, y_positions, 
                       color=color_avg_line, linewidth=2.5, 
                       marker='D', markersize=8, 
                       label='Average Speedup',
                       markerfacecolor=color_avg_line,
                       markeredgecolor='white',
                       markeredgewidth=1.5,
                       alpha=0.9, zorder=10)
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Average Speedup Factor', fontsize=13, fontweight='bold', 
                      color=color_avg_line, labelpad=10)
        ax2.tick_params(axis='x', labelcolor=color_avg_line, labelsize=11, width=1.5)
        ax2.set_ylim(ax.get_ylim())
        ax2.spines['top'].set_color(color_avg_line)
        ax2.spines['top'].set_linewidth(2)
        
        # Style the baseline reference line
        ax.axvline(x=1, color=color_baseline, linestyle='--', 
                  linewidth=2, alpha=0.7, label='Baseline (1×)', zorder=1)
        
        # Enhanced axis labels and title
        ax.set_xlabel('Speedup Factor (log scale)', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel('Method', fontsize=13, fontweight='bold', labelpad=10)
        # ax.set_title(f'{class_name} Graph Class', fontsize=16, fontweight='bold', pad=20)
        
        # Set log scale and improve grid
        ax.set_xscale('log')
        ax.grid(axis='x', alpha=0.25, linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        
        # Style the spines
        for spine in ['bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(1.5)
            ax.spines[spine].set_color('#2c3e50')
        ax.spines['right'].set_visible(False)
        
        # Improve tick styling
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1, length=3)
        
        # Add legend with better positioning and styling
        legend1 = ax.legend(loc='lower right', frameon=True, fancybox=True, 
                          shadow=True, fontsize=11, framealpha=0.95)
        legend1.get_frame().set_facecolor('white')
        legend1.get_frame().set_edgecolor('#2c3e50')
        legend1.get_frame().set_linewidth(1.5)
        
        # Add a subtle text note
        note_text = f'Baseline: {baseline_method}'
        ax.text(0.02, 0.98, note_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/runtime_speedup_vs_{baseline_method}_{class_name}.pdf", 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    # Plot 2: Heatmap of speedup
    pivot_speedup = speedup_df.pivot(index='method', columns='class', values='avg_speedup')
    # Remove baseline method
    if baseline_method in pivot_speedup.index:
        pivot_speedup = pivot_speedup.drop(baseline_method)
    
    plt.figure(figsize=(10, 8))
    # Use log scale for better visualization
    pivot_speedup_log = np.log2(pivot_speedup)
    annot_data = pivot_speedup.map(lambda x: f'{x:.2f}x' if pd.notna(x) else '')
    
    sns.heatmap(pivot_speedup_log, annot=annot_data, fmt='', cmap='RdYlGn', 
                center=0, cbar_kws={'label': 'log2(Speedup)'})
    plt.title(f'Runtime Speedup vs {baseline_method}', fontsize=14, fontweight='bold')
    plt.xlabel('Graph Class', fontsize=12)
    plt.ylabel('Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/runtime_speedup_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Runtime speedup plots saved to {save_dir}")


def plot_density_ranks(rank_df: pd.DataFrame, save_dir: str = "results/synthetic/figures"):
    """Plot average density ranks for each method."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if rank_df.empty:
        print("⚠️  No rank data to plot")
        return
    
    classes = sorted(rank_df['class'].unique())
    n_classes = len(classes)
    
    # Plot 1: Average rank bar chart for each class
    fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5), sharey=True)
    if n_classes == 1:
        axes = [axes]
    
    for idx, class_name in enumerate(classes):
        class_data = rank_df[rank_df['class'] == class_name].sort_values('avg_rank')
        
        colors = plt.cm.get_cmap('RdYlGn_r')(np.linspace(0.2, 0.8, len(class_data)))
        axes[idx].barh(range(len(class_data)), class_data['avg_rank'], color=colors)
        axes[idx].set_yticks(range(len(class_data)))
        axes[idx].set_yticklabels(class_data['method'])
        axes[idx].set_xlabel('Average Rank', fontsize=11)
        axes[idx].set_title(f'{class_name}', fontsize=13, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
        axes[idx].invert_xaxis()  # Lower rank (better) on the right
        
        # Add value labels
        for i, (_, row) in enumerate(class_data.iterrows()):
            rank = row['avg_rank']
            best_count = row['best_rank_count']
            label = f'{rank:.2f} ({best_count}×1st)'
            axes[idx].text(rank, i, label, ha='right', va='center', fontsize=8)
    
    fig.suptitle('Average Density Rank (1=best, lower is better)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/density_ranks.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Heatmap of ranks
    pivot_rank = rank_df.pivot(index='method', columns='class', values='avg_rank')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_rank, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Average Rank'}, vmin=1)
    plt.title('Average Density Rank by Method and Class (1=best)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Graph Class', fontsize=12)
    plt.ylabel('Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/density_ranks_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Density rank plots saved to {save_dir}")


def generate_summary_report(df: pd.DataFrame, success_df: pd.DataFrame, 
                           runtime_df: pd.DataFrame, improvement_df: pd.DataFrame,
                           save_dir: str = "results/synthetic"):
    """Generate a comprehensive summary report."""
    report_path = Path(save_dir) / "analysis_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SYNTHETIC GRAPH EXPERIMENTS - ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Graph classes: {', '.join(df['class'].unique())}\n")
        f.write(f"Methods tested: {', '.join(df['method'].unique())}\n")
        f.write(f"Unique graphs: {df['graph_name'].nunique()}\n\n")
        
        # Success rates by class
        f.write("SUCCESS RATES BY CLASS\n")
        f.write("-"*80 + "\n")
        for class_name in sorted(success_df['class'].unique()):
            class_success = success_df[success_df['class'] == class_name]
            f.write(f"\n{class_name}:\n")
            for _, row in class_success.sort_values('success_rate', ascending=False).iterrows():
                f.write(f"  {row['method']:20s}: {row['success_rate']:6.2f}% "
                       f"({row['successful']}/{row['total_graphs']})\n")
        
        # Runtime statistics
        # Count complete graphs
        if len(runtime_df) > 0:
            complete_graphs_per_class = {}
            for class_name in runtime_df['class'].unique():
                # Count unique graphs that appear in all methods for this class
                class_runtime = runtime_df[runtime_df['class'] == class_name]
                complete_graphs_per_class[class_name] = class_runtime['count'].iloc[0] if len(class_runtime) > 0 else 0
        
        f.write("\n\nRUNTIME STATISTICS (seconds)\n")
        f.write("Note: Calculated on graphs where ALL methods have results\n")
        f.write("-"*80 + "\n")
        for class_name in sorted(runtime_df['class'].unique()):
            class_runtime = runtime_df[runtime_df['class'] == class_name]
            f.write(f"\n{class_name}:\n")
            f.write(f"  {'Method':20s} {'Avg':>12s} {'Total':>12s} {'Min':>12s} {'Max':>12s}\n")
            for _, row in class_runtime.sort_values('avg_time').iterrows():
                f.write(f"  {row['method']:20s} {row['avg_time']:12.4f} "
                       f"{row['total_time']:12.4f} {row['min_time']:12.4f} "
                       f"{row['max_time']:12.4f}\n")
        
        # Density improvement
        if len(improvement_df) > 0:
            baseline = improvement_df['method'].mode()[0] if 'CEP' not in improvement_df['method'].values else 'CEP'
            f.write(f"\n\nDENSITY IMPROVEMENT VS {baseline} (%)\n")
            f.write("Note: Calculated on graphs where ALL methods have results\n")
            f.write("-"*80 + "\n")
            
            avg_improvement = improvement_df[improvement_df['method'] != baseline].groupby(
                ['class', 'method']
            )['improvement_pct'].agg(['mean', 'std', 'min', 'max']).reset_index()
            
            for class_name in sorted(avg_improvement['class'].unique()):
                class_imp = avg_improvement[avg_improvement['class'] == class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  {'Method':20s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}\n")
                for _, row in class_imp.sort_values('mean', ascending=False).iterrows():
                    f.write(f"  {row['method']:20s} {row['mean']:12.4f} "
                           f"{row['std']:12.4f} {row['min']:12.4f} "
                           f"{row['max']:12.4f}\n")
        
        # Best performers
        f.write("\n\nBEST PERFORMERS\n")
        f.write("-"*80 + "\n")
        
        # Highest success rate
        best_success = success_df.loc[success_df['success_rate'].idxmax()]
        f.write(f"\nHighest Success Rate:\n")
        f.write(f"  {best_success['method']} on {best_success['class']}: "
               f"{best_success['success_rate']:.2f}%\n")
        
        # Fastest method
        fastest = runtime_df.loc[runtime_df['avg_time'].idxmin()]
        f.write(f"\nFastest Average Runtime:\n")
        f.write(f"  {fastest['method']} on {fastest['class']}: "
               f"{fastest['avg_time']:.4f} seconds\n")
        
        # Best density improvement
        if len(improvement_df) > 0 and 'method' in improvement_df.columns:
            baseline_method = improvement_df['method'].mode()[0] if 'CEP' not in improvement_df['method'].values else 'CEP'
            best_candidates = improvement_df[improvement_df['method'] != baseline_method]
            if len(best_candidates) > 0:
                best_imp = best_candidates.loc[best_candidates['improvement_pct'].idxmax()]
                f.write(f"\nHighest Density Improvement:\n")
                f.write(f"  {best_imp['method']} on {best_imp['graph_name']}: "
                       f"{best_imp['improvement_pct']:.2f}%\n")
    
    print(f"✓ Summary report saved to {report_path}")
    
    # Also print to console
    with open(report_path, 'r') as f:
        print("\n" + f.read())


def analyze():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze synthetic graph experiment results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/synthetic',
        help='Directory containing result files'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='DCSGreedy',
        help='Baseline method for density comparison'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        default=False,
        help='Skip generating plots'
    )
    parser.add_argument(
        '--fill-cep-mip',
        action='store_true',
        default=False,
        help='Fill CEP_MIP data from method with longest runtime when all other methods have results'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("ANALYZING SYNTHETIC GRAPH EXPERIMENT RESULTS")
    print("="*80)
    
    # Load data
    print("\n📊 Loading results...")
    df = load_all_results(args.results_dir)
    
    if df.empty:
        print("❌ No results found!")
        return
    
    print(f"✓ Loaded {len(df)} results from {df['class'].nunique()} classes")
    
    # Fill CEP_MIP data if requested
    if args.fill_cep_mip:
        print("\n🔧 Filling CEP_MIP data from longest runtime method...")
        df = fill_cep_mip_from_longest_runtime(df)
    
    # Calculate metrics
    print("\n📈 Calculating metrics...")
    success_df = calculate_success_rates(df)
    print(f"✓ Success rates calculated")
    
    # Filter to only complete graphs for fair comparison
    print("\n🔍 Filtering to graphs with complete results...")
    complete_df = get_complete_graphs(df)
    n_complete = complete_df['graph_name'].nunique()
    n_total = df['graph_name'].nunique()
    print(f"✓ Found {n_complete}/{n_total} graphs with results from all methods")
    
    if complete_df.empty:
        print("⚠️  No graphs with complete results from all methods!")
        print("   Runtime and density analysis will be skipped.")
        runtime_df = pd.DataFrame()
        speedup_df = pd.DataFrame()
        rank_df = pd.DataFrame()
        improvement_df = pd.DataFrame()
    else:
        runtime_df = calculate_runtime_stats(complete_df)
        print(f"✓ Runtime statistics calculated on {n_complete} complete graphs")
        
        # Calculate runtime speedup vs CEP_MIP (using filled data where CEP_MIP succeeds/filled)
        speedup_df = calculate_runtime_speedup(df, 'CEP_MIP')
        if not speedup_df.empty:
            print(f"✓ Runtime speedup calculated vs CEP_MIP")
        
        # Calculate density ranks
        rank_df = calculate_density_ranks(complete_df)
        print(f"✓ Density ranks calculated on {n_complete} complete graphs")
        
        improvement_df = calculate_density_improvement(complete_df, args.baseline)
        print(f"✓ Density improvements calculated on {n_complete} complete graphs vs {args.baseline}")
    
    # Apply method abbreviations for display
    print("\n🏷️  Applying method abbreviations...")
    success_df = apply_method_abbr(success_df)
    runtime_df = apply_method_abbr(runtime_df)
    speedup_df = apply_method_abbr(speedup_df)
    rank_df = apply_method_abbr(rank_df)
    improvement_df = apply_method_abbr(improvement_df)
    
    # Export success rate table
    print("\n📋 Exporting tables...")
    export_success_rate_table(success_df, args.results_dir)
    if not speedup_df.empty:
        export_speedup_table(speedup_df, METHOD_ABBR.get('CEP_MIP', 'CEP_MIP'), args.results_dir)
    if not rank_df.empty:
        export_rank_table(rank_df, args.results_dir)
    
    # Generate visualizations
    if not args.no_plots:
        print("\n🎨 Generating visualizations...")
        save_dir = f"{args.results_dir}/figures"
        
        plot_success_rates(success_df, save_dir)
        
        if not runtime_df.empty:
            plot_runtime_comparison(runtime_df, save_dir)
        else:
            print("⚠️  Skipping runtime plots (no complete graphs)")
        
        if not speedup_df.empty:
            plot_runtime_speedup(speedup_df, METHOD_ABBR.get('CEP_MIP', 'CEP_MIP'), save_dir)
        else:
            print("⚠️  Skipping speedup plots (no CEP_MIP results)")
        
        if not rank_df.empty:
            plot_density_ranks(rank_df, save_dir)
        else:
            print("⚠️  Skipping density rank plots (no complete graphs)")
        
        if not improvement_df.empty:
            baseline_abbr = METHOD_ABBR.get(args.baseline, args.baseline) or args.baseline
            plot_density_improvement(improvement_df, baseline_abbr, save_dir)
        else:
            print("⚠️  Skipping density improvement plots (no complete graphs)")
    
    # Generate summary report
    print("\n📝 Generating summary report...")
    df_abbr = apply_method_abbr(df)
    generate_summary_report(df_abbr, success_df, runtime_df, improvement_df, args.results_dir)
    
    # Save detailed DataFrames
    print("\n💾 Saving detailed results...")
    success_df.to_csv(f"{args.results_dir}/success_rates.csv", index=False)
    runtime_df.to_csv(f"{args.results_dir}/runtime_stats.csv", index=False)
    if not speedup_df.empty:
        speedup_df.to_csv(f"{args.results_dir}/runtime_speedup.csv", index=False)
    if not rank_df.empty:
        rank_df.to_csv(f"{args.results_dir}/density_ranks.csv", index=False)
    improvement_df.to_csv(f"{args.results_dir}/density_improvements.csv", index=False)
    print(f"✓ Detailed CSVs saved to {args.results_dir}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    analyze()
