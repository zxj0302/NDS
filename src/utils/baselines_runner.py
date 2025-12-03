"""
Runner script to load YAML config and run baselines.
This script can be used as a standalone entry point.
"""

from loguru import logger
import yaml
import sys
import os

# Add the parent directory to the path so we can import baselines
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import run

def load_yaml_config(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_from_yaml(yaml_path, experiment='real-world', single=True, skip=True):
    """
    Load configuration from YAML file and run experiments.
    
    Args:
        yaml_path: Path to YAML configuration file
        experiment: Which experiment to run ('real-world', 'synthetic', 'test')
        single: If True, run on individual datasets. If False, run on folders
        skip: If True, skip existing output files
    """
    config = load_yaml_config(yaml_path)
    
    # Get the specific experiment configuration
    if experiment not in config:
        logger.error(f"Experiment '{experiment}' not found in config file")
        return
    
    exp_config = config[experiment]
    run(exp_config, single=single, skip=skip)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run baseline algorithms from YAML config')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--experiment', '-e', type=str, default='real-world',
                       choices=['real-world', 'synthetic', 'test'],
                       help='Which experiment to run')
    parser.add_argument('--folder-mode', '-f', action='store_true',
                       help='Run in folder mode (for synthetic data)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Do not skip existing output files')
    
    args = parser.parse_args()
    
    run_from_yaml(args.config, 
                  experiment=args.experiment,
                  single=not args.folder_mode,
                  skip=not args.no_skip)
