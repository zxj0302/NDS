import pandas as pd
import numpy as np
from pathlib import Path
import argparse


#================ CHANGE IF NEEDED ================
# Method name mapping (from internal names to display names)
METHOD_ABBR = {
    'DCSGreedy': 'DCSGreedy',
    'NEG_DSD': 'NEG-DSD',
    'CEP_L1': 'ECP-L1',
    'CEP_L5': 'ECP-L5',
    'CEP_L20': 'ECP-L20',
    'CEP_L50': 'ECP-L50',
    'CEP_L100': 'ECP-L100',
    'CEP_L200': 'ECP-L200',
    'CEP_L300': 'ECP-L300',
    'CEP_K0': 'ECP-K0',
    'CEP_K50': 'ECP-K50',
    'CEP_K100': 'ECP-K100',
    'CEP_K200': 'ECP-K200',
    'CEP_K500': 'ECP-K500',
    'CEP': 'ECP',
    'CEP_MIP_B': 'EM',
    'CEP_QPBO_MIP_B': 'EM w/ Q',
    'CEP_PRUNING_QPBO_MIP_B': 'EM w/ Q+P',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_B': 'BAR',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_80_B': 'BAR-80',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_90_B': 'BAR-90',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_95_B': 'BAR-95',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_99_B': 'BAR-99',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_999_B': 'BAR-999',
}

# Define which methods to show in the table (in this order)
methods_to_show = ['DCSGreedy', 'NEG-DSD', 'ECP', 'BAR-80', 'BAR-90', 'BAR-95', 'BAR-99', 'BAR', 'EM']
baseline_alg = 'EM'
default_save_path = Path('results/tex/syn_speedup_table.tex')
#==================================================


def build_latex_table(methods, summary, failed_counts_by_method, baseline_name, transpose=False):
    latex_lines = []
    latex_lines.append('\\begin{table*}[htbp]')
    latex_lines.append('\\centering')
    latex_lines.append('\\small')
    latex_lines.append(f'\\caption{{Runtime Speedup over {baseline_name} on Syn84}}')
    latex_lines.append('\\label{tab:speedup}')
    latex_lines.append('\\setlength{\\tabcolsep}{3.5pt}')

    metrics = [
        ('Avg', 'avg'),
        ('Median', 'median'),
        ('Min', 'min'),
        ('Max', 'max'),
    ]

    if not transpose:
        latex_lines.append('\\begin{tabular}{l' + 'r' * len(methods) + '}')
        latex_lines.append('\\toprule')
        latex_lines.append('Metric & ' + ' & '.join(methods) + ' \\\\')
        latex_lines.append('\\midrule')

        for metric_name, metric_key in metrics:
            row = metric_name
            for method in methods:
                row += f" & {format_number(summary[method][metric_key])}"
            row += ' \\\\'
            latex_lines.append(row)

        failed_row = 'Failed Graphs'
        for method in methods:
            failed_row += f' & {failed_counts_by_method[method]}'
        failed_row += ' \\\\'
        latex_lines.append(failed_row)
    else:
        latex_lines.append('\\begin{tabular}{l' + 'r' * (len(metrics) + 1) + '}')
        latex_lines.append('\\toprule')
        latex_lines.append('Method & ' + ' & '.join(metric_name for metric_name, _ in metrics) + ' & Failed Graphs \\\\')
        latex_lines.append('\\midrule')

        for method in methods:
            row = method
            for _, metric_key in metrics:
                row += f" & {format_number(summary[method][metric_key])}"
            row += f" & {failed_counts_by_method[method]}"
            row += ' \\\\'
            latex_lines.append(row)

    latex_lines.append('\\bottomrule')
    latex_lines.append('\\end{tabular}')
    latex_lines.append('\\end{table*}')
    return latex_lines


def format_number(num, max_digits=4):
    """
    Format a number to use at most 4 total digits.
    If the number is too large, use scientific notation.
    """
    if pd.isna(num):
        return '-'
    
    # Handle the case where the number is very small or very large
    abs_num = abs(num)
    
    if abs_num == 0:
        return '0'
    
    # Calculate number of digits before decimal point
    if abs_num >= 1:
        digits_before = len(str(int(abs_num)))
    else:
        digits_before = 0
    
    # If the integer part alone exceeds max_digits, use scientific notation
    if digits_before > max_digits:
        # Use scientific notation with appropriate precision
        exp = int(np.floor(np.log10(abs_num)))
        mantissa = num / (10 ** exp)
        # Format mantissa with max_digits-1 total digits
        precision = max(0, max_digits - 2)  # -2 for the leading digit and potential sign
        return f"{mantissa:.{precision}f}e{exp:+d}"
    
    # Otherwise, use fixed-point notation with appropriate decimal places
    if digits_before >= max_digits:
        # No decimal places
        return f"{int(round(num))}"
    else:
        # Allow some decimal places
        decimal_places = max_digits - digits_before
        formatted = f"{num:.{decimal_places}f}"
        # Remove trailing zeros and decimal point if not needed
        formatted = formatted.rstrip('0').rstrip('.')
        return formatted

# Read only the required columns from the ER time table.
display_to_internal = {v: k for k, v in METHOD_ABBR.items()}
wanted_internal_methods = [display_to_internal[m] for m in methods_to_show if m in display_to_internal]
columns_to_read = ['graph_name'] + [c for c in wanted_internal_methods if c != 'graph_name']
df_time = pd.read_csv('results/synthetic/ER/ER_time_table.csv', usecols=columns_to_read)

# Count per-method failures on the raw table before any complete-case filtering.
failed_counts_by_method = {
    METHOD_ABBR[method]: int(df_time[method].isna().sum())
    for method in wanted_internal_methods
}

# Keep only graphs where all wanted methods have valid times.
df_time_complete = df_time.dropna(subset=wanted_internal_methods)
counted_graphs = df_time_complete['graph_name'].nunique()
failed_graphs = df_time['graph_name'].nunique() - counted_graphs

# Long format with display names.
df_long = df_time_complete.melt(id_vars=['graph_name'], var_name='method', value_name='time')
df_long['method_display'] = df_long['method'].map(METHOD_ABBR)
df_long = df_long[df_long['method_display'].notna()]

# Baseline runtime is DCSGreedy on the same counted graphs.
baseline_times = df_long[df_long['method_display'] == baseline_alg][['graph_name', 'time']].rename(columns={'time': 'baseline_time'})
df_long = df_long.merge(baseline_times, on='graph_name', how='left')
df_long['speedup'] = df_long['baseline_time'] / df_long['time']

# Methods that are present after complete-case filtering.
methods = [m for m in methods_to_show if m in df_long['method_display'].values]

# Compute summary statistics directly from the counted graphs.
summary = {}
for method in methods:
    method_values = df_long[df_long['method_display'] == method]['speedup'].values
    summary[method] = {
        'avg': float(np.mean(method_values)),
        'median': float(np.median(method_values)),
        'min': float(np.min(method_values)),
        'max': float(np.max(method_values)),
    }

def main():
    parser = argparse.ArgumentParser(description='Generate runtime speedup LaTeX table for ER graphs.')
    parser.add_argument('--transpose', action='store_true', help='Transpose the table layout.')
    parser.add_argument('--output', type=str, default=None, help='Output LaTeX file path.')
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else default_save_path
    if args.transpose and args.output is None:
        output_path = output_path.with_name(output_path.stem + '_transposed' + output_path.suffix)

    latex_lines = build_latex_table(
        methods=methods,
        summary=summary,
        failed_counts_by_method=failed_counts_by_method,
        baseline_name=baseline_alg,
        transpose=args.transpose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(latex_lines) + '\n')

    print(f'Saved LaTeX table to {output_path}')
    print(f'Counted graphs: {counted_graphs}')
    print(f'Failed graphs: {failed_graphs}')


if __name__ == '__main__':
    main()
