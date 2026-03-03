import pandas as pd
import numpy as np

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

# Read the CSV file
df = pd.read_csv('results/synthetic/runtime_speedup.csv')

# Define which methods to show in the table (in this order)
# methods_to_show = ['DCSGreedy','NEG-DSD', 'CEP', 'MIQP-D', 'CQM-B-90', 'CQM-B-95', 'CQM-B-99', 'CQM-B', 'CQM-D-90', 'CQM-D-95', 'CQM-D-99', 'CQM-D']
methods_to_show = ['MIQP-B', 'CQM-B w/o P+Con', 'CQM-B w/o Con', 'CQM-B']

# Filter to only include methods that exist in the data and are in our list
methods = [m for m in methods_to_show if m in df['method'].values]

# Create the LaTeX table
print("\\begin{table*}[htbp]")
print("\\centering")
print("\\small")
print("\\caption{Runtime Speedup over MIQP-B on Syn175}")
print("\\label{tab:speedup}")
print("\\setlength{\\tabcolsep}{3.5pt}")  # Adjust column spacing (default is 6pt)

# Create table header
print("\\begin{tabular}{l" + "r" * len(methods) + "}")
print("\\toprule")

# Method names as column headers
header = "Metric & " + " & ".join(methods) + " \\\\"
print(header)
print("\\midrule")

# Metrics to display
metrics = [
    ('avg_speedup', 'Avg'),
    ('median_speedup', 'Median'),
    ('min_speedup', 'Min'),
    ('max_speedup', 'Max')
]

# Generate rows
for metric_col, metric_name in metrics:
    row = metric_name
    for method in methods:
        # Get the value for this method and metric
        value = df[df['method'] == method][metric_col].values
        if len(value) > 0:
            formatted_value = format_number(value[0])
        else:
            formatted_value = '-'
        row += f" & {formatted_value}"
    row += " \\\\"
    print(row)

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table*}")
