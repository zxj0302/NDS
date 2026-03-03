import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('results/synthetic/ER/ER_time_table.csv')

# Extract node count, edge count, and weight mode from graph_name
# Format: ER_nx_my_settingz_inst0
def parse_graph_name(name):
    parts = name.split('_')
    n = int(parts[1][1:])  # Remove 'n' prefix
    m = int(parts[2][1:])  # Remove 'm' prefix
    setting = int(parts[3].replace('setting', ''))
    weight_mode = setting % 5
    return n, m, weight_mode

# Parse graph names
df['node_count'] = df['graph_name'].apply(lambda x: parse_graph_name(x)[0])
df['edge_count'] = df['graph_name'].apply(lambda x: parse_graph_name(x)[1])
df['weight_mode'] = df['graph_name'].apply(lambda x: parse_graph_name(x)[2])
df['density'] = df['edge_count'] / df['node_count']

# Filter for CEP_PRUNING_QPBO_MIP_CONSTRAIN method
method_col = 'CEP_PRUNING_QPBO_MIP_CONSTRAIN'

# Create a success indicator (1 if not null, 0 if null)
df['success'] = df[method_col].notna().astype(int)

# Define distribution labels
distribution_labels = {
    0: r'$\mathcal{U}(-1, 1)$',
    1: r'$\mathcal{N}(0, 0.3)$',
    2: r'$\mathcal{N}(0, 0.5)$',
    3: r'$\text{Beta}(2.0, 5.0)$',
    4: r'$\text{Beta}(5.0, 2.0)$'
}

# Map weight mode to distribution labels
df['distribution'] = df['weight_mode'].map(distribution_labels)

# Table 1: Success count by node count and distribution
table1 = df.pivot_table(
    values='success',
    index='node_count',
    columns='distribution',
    aggfunc='sum',
    fill_value=0
).astype(int)

# Reorder columns to match the order of distributions
column_order = [distribution_labels[i] for i in range(5)]
table1 = table1[column_order]

# Table 2: Success count by density and distribution
# Round density to avoid floating point issues
df['density_rounded'] = df['density'].round(1)

table2 = df.pivot_table(
    values='success',
    index='density_rounded',
    columns='distribution',
    aggfunc='sum',
    fill_value=0
).astype(int)

# Reorder columns to match the order of distributions
table2 = table2[column_order]

# Generate LaTeX for Table 1
print("=" * 80)
print("Table 1: Success Count by Node Count and Distribution")
print("=" * 80)
latex1 = table1.to_latex(
    escape=False,
    column_format='l' + 'c' * len(table1.columns),
    caption='Success count by node count and weight distribution',
    label='tab:success_node_count'
)
print(latex1)
print()

# Generate LaTeX for Table 2
print("=" * 80)
print("Table 2: Success Count by Density and Distribution")
print("=" * 80)
latex2 = table2.to_latex(
    escape=False,
    column_format='l' + 'c' * len(table2.columns),
    caption='Success count by density and weight distribution',
    label='tab:success_density'
)
print(latex2)
print()

# Save to files
with open('success_table_node_count.tex', 'w') as f:
    f.write(latex1)
print("Saved Table 1 to: success_table_node_count.tex")

with open('success_table_density.tex', 'w') as f:
    f.write(latex2)
print("Saved Table 2 to: success_table_density.tex")

# Print summary statistics
print("\n" + "=" * 80)
print("Summary Statistics")
print("=" * 80)
print(f"Total graphs: {len(df)}")
print(f"Total successful runs: {df['success'].sum()}")
print(f"Success rate: {df['success'].mean():.2%}")
print(f"\nNode counts: {sorted(df['node_count'].unique())}")
print(f"Densities: {sorted(df['density'].unique())}")
