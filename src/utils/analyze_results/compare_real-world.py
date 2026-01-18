import json
import os
import pandas as pd
from pathlib import Path

def read_algorithm_result(json_path):
    """
    Read algorithm result from JSON file.
    Returns (density, time, status) where status is 'Success' or 'Failed'
    """
    if not os.path.exists(json_path):
        return None, None, 'No File'
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check if the algorithm failed
        info = data.get('config', {}).get('info', '')
        if 'Fail' in info and 'Terminate' not in info:
            return None, None, 'Failed'
        
        density = data.get('density', None)
        time = data.get('time', None)
        
        return density, time, 'Success'
    except Exception as e:
        return None, None, f'Error: {str(e)}'

def compare_algorithms(base_path='./output/real-world'):
    """
    Compare algorithms across all datasets.
    """
    base_path = Path(base_path)
    
    # Get all dataset folders
    datasets = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
    
    # Collect all unique algorithm names
    all_algorithms = set()
    for dataset in datasets:
        dataset_path = base_path / dataset
        if dataset_path.is_dir():
            for json_file in dataset_path.glob('*.json'):
                algo_name = json_file.stem  # filename without extension
                all_algorithms.add(algo_name)
    
    algorithms = sorted(all_algorithms)
    
    # Create comparison data
    results = []
    
    for dataset in datasets:
        dataset_path = base_path / dataset
        row = {'Dataset': dataset}
        
        for algo in algorithms:
            json_path = dataset_path / f'{algo}.json'
            density, time, status = read_algorithm_result(json_path)
            
            if status == 'Success':
                row[f'{algo}_Density'] = density
                row[f'{algo}_Time'] = time
            else:
                row[f'{algo}_Density'] = status
                row[f'{algo}_Time'] = status
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns: Dataset, then for each algorithm: Density, Time
    columns = ['Dataset']
    for algo in algorithms:
        columns.extend([f'{algo}_Density', f'{algo}_Time'])
    
    df = df[columns]
    
    return df, algorithms

def create_separate_tables(df, algorithms):
    """
    Create separate tables for density and time comparisons.
    """
    # Density table
    density_columns = ['Dataset'] + [f'{algo}_Density' for algo in algorithms]
    density_df = df[density_columns].copy()
    density_df.columns = ['Dataset'] + algorithms
    
    # Time table
    time_columns = ['Dataset'] + [f'{algo}_Time' for algo in algorithms]
    time_df = df[time_columns].copy()
    time_df.columns = ['Dataset'] + algorithms
    
    return density_df, time_df

def create_latex_tables(density_df, time_df, algorithms):
    """
    Create separate LaTeX tables for runtime and density.
    Runtime is shown in milliseconds.
    """
    datasets = list(density_df['Dataset'])
    
    # Dataset abbreviations mapping
    dataset_abbr = {
        'Abortion': 'AB',
        'Biogrid': 'BG',
        'Brexit': 'BX',
        'Collins': 'CL',
        'Election': 'EL',
        'Gavin': 'GV',
        'Gun': 'GN',
        'Krogan-Core': 'KC',
        'Krogan-Extended': 'KE',
        'Partisanship': 'PA',
        'Referendum': 'RF'
    }
    
    # Method name abbreviations for LaTeX
    method_abbr = {
        'CEP_MIP': 'MIQP-B',
        'CEP_MIP_NB': 'MIQP-D',
        'CEP_PRUNING_QPBO_MIP_CONSTRAIN': 'CQM-B',
        'CEP_PRUNING_QPBO_MIP_CONSTRAIN_NB': 'CQM-D'
    }
    
    # Format numeric values
    def format_value(val, is_time=False):
        if isinstance(val, (int, float)):
            # Convert seconds to milliseconds for time values
            if is_time:
                val = val * 1000
                # Format time with 3 significant figures
                if val >= 100:
                    # Use scientific notation for large values
                    return f"{val:.2e}"
                elif val >= 10:
                    return f"{val:.1f}"
                elif val >= 1:
                    return f"{val:.2f}"
                elif val >= 0.1:
                    return f"{val:.3f}"
                else:
                    # Use scientific notation for very small values
                    return f"{val:.2e}"
            else:
                # Density: keep 3 decimal places
                return f"{val:.3f}"
        else:
            return "-"  # For failed or missing results
    
    # Prepare data for both tables
    time_data = {'Method': algorithms}
    density_data = {'Method': algorithms}
    
    for dataset in datasets:
        time_vals = []
        density_vals = []
        
        for algo in algorithms:
            density_val = density_df[density_df['Dataset'] == dataset][algo].values[0]
            time_val = time_df[time_df['Dataset'] == dataset][algo].values[0]
            
            density_vals.append(format_value(density_val, is_time=False))
            time_vals.append(format_value(time_val, is_time=True))
        
        time_data[dataset] = time_vals
        density_data[dataset] = density_vals
    
    time_combined_df = pd.DataFrame(time_data)
    density_combined_df = pd.DataFrame(density_data)
    
    # Column format: l for method name, then c for each dataset
    col_format = '@{}l' + 'c' * len(datasets) + '@{}'
    
    # Header row: Method and dataset names
    header = "Method"
    for dataset in datasets:
        # Use abbreviation if available, otherwise use full name
        abbr = dataset_abbr.get(dataset, dataset)
        header += f" & {abbr}"
    header += " \\\\"
    
    # ===== TIME TABLE =====
    time_latex_lines = []
    time_latex_lines.append("\\begin{table*}[t]")
    time_latex_lines.append("\\centering")
    time_latex_lines.append("\\footnotesize")
    time_latex_lines.append("\\setlength{\\tabcolsep}{4pt}")
    time_latex_lines.append("\\caption{Algorithm Runtime Comparison (milliseconds)}")
    time_latex_lines.append("\\label{tab:algorithm_runtime}")
    time_latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    time_latex_lines.append("\\hline")
    time_latex_lines.append(header)
    time_latex_lines.append("\\hline")
    
    # Data rows for time
    for i, algo in enumerate(algorithms):
        # Use abbreviated name if available, otherwise use full name with escaped underscores
        display_name = method_abbr.get(algo, algo.replace('_', '\\_'))
        row = display_name
        for dataset in datasets:
            time_val = time_data[dataset][i]
            row += f" & {time_val}"
        row += " \\\\"
        time_latex_lines.append(row)
    
    time_latex_lines.append("\\hline")
    time_latex_lines.append("\\end{tabular}")
    time_latex_lines.append("\\end{table*}")
    
    time_latex = '\n'.join(time_latex_lines)
    
    # ===== DENSITY TABLE =====
    density_latex_lines = []
    density_latex_lines.append("\\begin{table*}[t]")
    density_latex_lines.append("\\centering")
    density_latex_lines.append("\\footnotesize")
    density_latex_lines.append("\\setlength{\\tabcolsep}{4pt}")
    density_latex_lines.append("\\caption{Algorithm Density Comparison}")
    density_latex_lines.append("\\label{tab:algorithm_density}")
    density_latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    density_latex_lines.append("\\hline")
    density_latex_lines.append(header)
    density_latex_lines.append("\\hline")
    
    # Data rows for density
    for i, algo in enumerate(algorithms):
        # Use abbreviated name if available, otherwise use full name with escaped underscores
        display_name = method_abbr.get(algo, algo.replace('_', '\\_'))
        row = display_name
        for dataset in datasets:
            density_val = density_data[dataset][i]
            row += f" & {density_val}"
        row += " \\\\"
        density_latex_lines.append(row)
    
    density_latex_lines.append("\\hline")
    density_latex_lines.append("\\end{tabular}")
    density_latex_lines.append("\\end{table*}")
    
    density_latex = '\n'.join(density_latex_lines)
    
    return time_latex, density_latex, time_combined_df, density_combined_df

def main():
    print("=" * 100)
    print("Comparing Algorithm Performance on Real-World Datasets")
    print("=" * 100)
    print()
    
    # Get comparison data
    df, algorithms = compare_algorithms()
    
    # Create separate tables
    density_df, time_df = create_separate_tables(df, algorithms)
    
    # Create LaTeX tables
    time_latex, density_latex, time_combined_df, density_combined_df = create_latex_tables(density_df, time_df, algorithms)
    
    # Display results
    print("RUNTIME COMPARISON (milliseconds)")
    print("-" * 100)
    print(time_combined_df.to_string(index=False))
    print()
    print()
    
    print("DENSITY COMPARISON")
    print("-" * 100)
    print(density_combined_df.to_string(index=False))
    print()
    
    # Save LaTeX tables to separate files
    with open('runtime_table.tex', 'w') as f:
        f.write("% Algorithm Runtime Comparison Table (milliseconds)\n")
        f.write("% Methods as rows, datasets as columns\n\n")
        f.write(time_latex)
    
    with open('density_table.tex', 'w') as f:
        f.write("% Algorithm Density Comparison Table\n")
        f.write("% Methods as rows, datasets as columns\n\n")
        f.write(density_latex)
    
    print()
    print("=" * 100)
    print("LaTeX tables saved to:")
    print("  - runtime_table.tex")
    print("  - density_table.tex")
    print("=" * 100)
    
    # Print LaTeX code to console
    print()
    print("RUNTIME TABLE (LaTeX)")
    print("-" * 100)
    print(time_latex)
    print()
    print()
    
    print("DENSITY TABLE (LaTeX)")
    print("-" * 100)
    print(density_latex)
    print()
    
    # Print summary statistics
    print()
    print("SUMMARY STATISTICS")
    print("-" * 100)
    
    for algo in algorithms:
        time_col = f'{algo}_Time'
        density_col = f'{algo}_Density'
        
        # Count successes and failures
        time_data = df[time_col]
        success_count = sum(isinstance(x, (int, float)) for x in time_data)
        total_count = len(time_data)
        
        print(f"{algo}:")
        print(f"  Success: {success_count}/{total_count}")
        
        if success_count > 0:
            # Calculate average time and density for successful runs
            valid_times = [x for x in time_data if isinstance(x, (int, float))]
            valid_densities = [df.loc[i, density_col] for i, x in enumerate(time_data) if isinstance(x, (int, float))]
            
            avg_time = sum(valid_times) / len(valid_times)
            avg_density = sum(float(d) for d in valid_densities) / len(valid_densities)
            
            print(f"  Average Time: {avg_time:.6f} seconds")
            print(f"  Average Density: {avg_density:.6f}")
        print()

if __name__ == '__main__':
    main()
