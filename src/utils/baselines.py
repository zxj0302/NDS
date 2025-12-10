from loguru import logger
import os
import subprocess
import json
import natsort


def write_config_json(params, config_dir="./temp"):
    """Write algorithm parameters to a JSON config file in temp directory."""
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    # Create a unique filename based on params
    import hashlib
    config_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    config_path = os.path.join(config_dir, f"config_{config_hash}.json")
    
    with open(config_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    return config_path


def get_dataset_paths(config):
    """Extract all dataset paths from config based on input type."""
    # Handle single file inputs
    if 'input' in config:
        for item in config['input']:
            path = item.get('path') if isinstance(item, dict) else item
            enabled = item.get('toggle', True) if isinstance(item, dict) else True
            if enabled:
                yield path
    
    # Handle folder inputs (batch processing)
    elif 'input_folder' in config:
        for folder_config in config['input_folder']:
            if not folder_config.get('toggle', True):
                continue
            folder_path = folder_config['path']
            for filename in natsort.natsorted(os.listdir(folder_path)):
                if filename.endswith('.txt'):
                    yield os.path.join(folder_path, filename)


def get_output_path(dataset_path, comp_name, output_base, reverse):
    """Generate output path for a dataset and competitor."""
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    suffix = '_r' if reverse else ''
    return os.path.join(output_base, dataset_name, f'{dataset_name}_{comp_name}{suffix}.json')


def run_competitor(dataset_path, competitor, output_path, reverse):
    """Run a single competitor on a dataset."""
    comp_name = competitor['name']
    program = competitor['exe']
    params = competitor.get('params', {})
    
    # Build complete config
    full_config = {
        'input': dataset_path,
        'output': output_path,
        'reverse_weight': reverse,
        **params
    }
    
    # Write config and execute
    config_path = write_config_json(full_config)
    try:
        subprocess.run([program, config_path], check=True)
        result = json.load(open(output_path))
        logger.info(f'{comp_name:<45}: time: {result["time"]:.6f}s, density: {result["density"]:.6f}')
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)


def run(config, skip_existing=True):
    """Run all enabled competitors on all enabled datasets.
    
    Args:
        config: Configuration dict with 'input'/'input_folder', 'competitors', 'output', etc.
        skip_existing: Skip if output file already exists
    """
    output_base = config['output']
    reverse = config.get('weight_reverse', False)
    competitors = [c for c in config['competitors'] if c.get('toggle', True)]
    
    # Process each dataset
    for dataset_path in get_dataset_paths(config):
        logger.info(f'Processing dataset: {dataset_path}')
        
        for competitor in competitors:
            output_path = get_output_path(dataset_path, competitor['name'], output_base, reverse)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Skip if already exists
            if skip_existing and os.path.exists(output_path):
                logger.debug(f'Skipping existing: {output_path}')
                continue
            
            run_competitor(dataset_path, competitor, output_path, reverse)
    
    logger.success('All done!')
