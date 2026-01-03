"""
Collect and organize results from synthetic graph experiments.
For a given graph class (BA, ER, RGG, SBM, WS), this script reads all results
from different methods and organizes density and time metrics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import natsort
import pandas as pd


def read_result_json(json_path: str) -> Optional[Dict]:
    """Read a result JSON file and extract relevant metrics."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None


def is_result_successful(data: Optional[Dict]) -> bool:
    """Check if a result is truly successful.
    
    A result is successful if:
    1. Data exists (not None)
    2. The json["config"]["info"] field does not contain "Fail" substring
    """
    if data is None:
        return False
    
    try:
        info = data.get('config', {}).get('info', '')
        if 'Fail' in info:
            return False
        return True
    except Exception:
        # If there's any issue accessing config.info, consider it successful
        # (backwards compatibility)
        return True


def collect_class_results(graph_class: str, output_base: str = "output/synthetic") -> pd.DataFrame:
    """
    Collect results for all graphs in a given class.
    
    Args:
        graph_class: Name of the graph class (BA, ER, RGG, SBM, WS)
        output_base: Base directory containing output results
        
    Returns:
        DataFrame with columns: graph_name, method, density, time, status
    """
    class_dir = Path(output_base) / graph_class
    
    if not class_dir.exists():
        print(f"Directory {class_dir} does not exist")
        return pd.DataFrame()
    
    results = []
    
    # Iterate through all graph folders
    graph_folders = natsort.natsorted([d for d in class_dir.iterdir() if d.is_dir()])
    
    for graph_folder in graph_folders:
        graph_name = graph_folder.name
        
        # Get all JSON files (each represents a different method)
        json_files = list(graph_folder.glob("*.json"))
        
        # Track which methods have results
        for json_file in json_files:
            method_name = json_file.stem  # filename without .json extension
            
            data = read_result_json(str(json_file))
            
            if is_result_successful(data):
                # Extract density and time
                density = data.get('density', None)
                time = data.get('time', None)
                
                results.append({
                    'graph_name': graph_name,
                    'method': method_name,
                    'density': density,
                    'time': time,
                    'status': 'success'
                })
            else:
                # Failed due to reading error or "Fail" in config.info
                results.append({
                    'graph_name': graph_name,
                    'method': method_name,
                    'density': None,
                    'time': None,
                    'status': 'failed'
                })
    
    df = pd.DataFrame(results)
    return df


def get_available_methods(graph_class: str, output_base: str = "output/synthetic") -> set:
    """Get the set of all methods that appear in any graph of this class."""
    class_dir = Path(output_base) / graph_class
    methods = set()
    
    if class_dir.exists():
        for graph_folder in class_dir.iterdir():
            if graph_folder.is_dir():
                for json_file in graph_folder.glob("*.json"):
                    methods.add(json_file.stem)
    
    return methods


def create_complete_results(graph_class: str, output_base: str = "output/synthetic") -> pd.DataFrame:
    """
    Create a complete results table including missing methods (marked as failed).
    
    Args:
        graph_class: Name of the graph class
        output_base: Base directory containing output results
        
    Returns:
        DataFrame with all graph-method combinations
    """
    class_dir = Path(output_base) / graph_class
    
    if not class_dir.exists():
        print(f"Directory {class_dir} does not exist")
        return pd.DataFrame()
    
    # Get all available methods across all graphs
    all_methods = get_available_methods(graph_class, output_base)
    
    # Get all graph names
    graph_folders = natsort.natsorted([d for d in class_dir.iterdir() if d.is_dir()])

    results = []
    
    for graph_folder in graph_folders:
        graph_name = graph_folder.name
        
        # Get methods that have results for this graph
        existing_methods = {}
        for json_file in graph_folder.glob("*.json"):
            method_name = json_file.stem
            data = read_result_json(str(json_file))
            existing_methods[method_name] = data
        
        # Add results for all methods
        for method in all_methods:
            if method in existing_methods:
                data = existing_methods[method]
                if is_result_successful(data):
                    results.append({
                        'graph_name': graph_name,
                        'method': method,
                        'density': data.get('density', None),
                        'time': data.get('time', None),
                        'status': 'success'
                    })
                else:
                    # Failed due to reading error or "Fail" in config.info
                    results.append({
                        'graph_name': graph_name,
                        'method': method,
                        'density': None,
                        'time': None,
                        'status': 'failed'
                    })
            else:
                # Method failed for this graph (no JSON file)
                results.append({
                    'graph_name': graph_name,
                    'method': method,
                    'density': None,
                    'time': None,
                    'status': 'failed'
                })
    
    df = pd.DataFrame(results)
    return df


def save_results(df: pd.DataFrame, graph_class: str, output_dir: str = "output/synthetic"):
    """
    Save results in multiple formats.
    
    Args:
        df: DataFrame containing results
        graph_class: Name of the graph class
        output_dir: Directory to save results
    """
    output_path = Path(output_dir) / graph_class
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_file = output_path / f"{graph_class}_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved CSV to {csv_file}")
    
    # Save as JSON
    json_file = output_path / f"{graph_class}_results.json"
    df.to_json(json_file, orient='records', indent=2)
    print(f"Saved JSON to {json_file}")
    
    # Create pivot tables for easier analysis
    if not df.empty and 'density' in df.columns:
        # Pivot table for density
        density_pivot = df.pivot_table(
            values='density',
            index='graph_name',
            columns='method',
            aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
        )
        # Sort index using natural sort
        density_pivot = density_pivot.reindex(natsort.natsorted(density_pivot.index))
        density_file = output_path / f"{graph_class}_density_table.csv"
        density_pivot.to_csv(density_file)
        print(f"Saved density pivot table to {density_file}")
        
        # Pivot table for time
        time_pivot = df.pivot_table(
            values='time',
            index='graph_name',
            columns='method',
            aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
        )
        # Sort index using natural sort
        time_pivot = time_pivot.reindex(natsort.natsorted(time_pivot.index))
        time_file = output_path / f"{graph_class}_time_table.csv"
        time_pivot.to_csv(time_file)
        print(f"Saved time pivot table to {time_file}")


def collect():
    """Main function to collect and organize results."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect results from synthetic graph experiments'
    )
    parser.add_argument(
        'graph_class',
        type=str,
        choices=['BA', 'ER', 'RGG', 'SBM', 'WS', 'all'],
        default='all',
        help='Graph class to process (or "all" for all classes)'
    )
    parser.add_argument(
        '--output-base',
        type=str,
        default='output/synthetic',
        help='Base directory containing output results'
    )
    parser.add_argument(
        '--save-base',
        type=str,
        default='results/synthetic',
        help='Base directory containing saved results'
    )
    parser.add_argument(
        '--complete',
        action='store_true',
        default=True,
        help='Include all method-graph combinations (mark missing as failed)'
    )
    
    args = parser.parse_args()
    
    # Determine which classes to process
    if args.graph_class == 'all':
        classes = ['BA', 'ER', 'RGG', 'SBM', 'WS']
    else:
        classes = [args.graph_class]
    
    # Process each class
    for graph_class in classes:
        print(f"\n{'='*60}")
        print(f"Processing class: {graph_class}")
        print(f"{'='*60}")
        
        if args.complete:
            df = create_complete_results(graph_class, args.output_base)
        else:
            df = collect_class_results(graph_class, args.output_base)
        
        if not df.empty:
            print(f"\nFound {len(df)} results")
            print(f"Graphs: {df['graph_name'].nunique()}")
            print(f"Methods: {df['method'].nunique()}")
            
            # Show summary statistics
            if 'status' in df.columns:
                print("\nStatus summary:")
                print(df['status'].value_counts())
            
            # Save results
            save_results(df, graph_class, args.save_base)
        else:
            print(f"No results found for {graph_class}")


if __name__ == "__main__":
    collect()
