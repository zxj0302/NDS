from loguru import logger
import os
import subprocess
import json
from tqdm import tqdm
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


def run(config, single=True, skip=True):
    if single:
        datasets = config.get('input')
        output_folder = config.get('output')
        reverse = config.get('weight_reverse', False)
        for dataset in natsort.natsorted(datasets):
            if dataset.get('toggle') is False:
                continue
            dataset_path = dataset.get('path') if isinstance(dataset, dict) else dataset
            logger.info(f'Running on dataset: {dataset_path}')
            dataset_name = dataset_path.split('/')[-1].split('.')[0]
            for competitor in config.get('competitors'):
                if not competitor.get('toggle', False):
                    continue
                comp_name = competitor.get('name')
                program = competitor.get('exe')
                params = competitor.get('params', {})
                
                suffix = '_r' if reverse else ''
                output = os.path.join(output_folder, f'{dataset_name}', f'{dataset_name}_{comp_name}{suffix}.json')
                if not os.path.exists(os.path.dirname(output)):
                    os.makedirs(os.path.dirname(output))
                if skip and os.path.exists(output):
                    continue
                
                # Build complete config with all parameters
                full_config = {
                    'input': dataset_path,
                    'output': output,
                    'reverse_weight': reverse,
                    **params
                }
                
                # Write config JSON
                config_path = write_config_json(full_config)
                
                try:
                    # All algorithms now use single config file parameter
                    cmd = [program, config_path]
                    subprocess.run(cmd, check=True)
                    result = json.load(open(output))
                    logger.info(f'{comp_name:<15}: time: {result["time"]:.6f}s, density: {result["density"]:.6f}')
                finally:
                    # Clean up config file
                    if os.path.exists(config_path):
                        os.remove(config_path)
    else:
        reverse = config.get('weight_reverse', False)
        for input_folder in config.get('input_folder'):
            if input_folder.get('toggle') is False:
                continue
            input_folder_path = input_folder.get('path')
            logger.info(f'Running on input folder: {input_folder_path}')
            for graph_file in tqdm(natsort.natsorted([f for f in os.listdir(input_folder_path) if f.endswith('.txt')])):
                logger.info(f'Processing graph file: {graph_file}')
                file_name = os.path.splitext(graph_file)[0]
                for competitor in config.get('competitors'):
                    if not competitor.get('toggle', False):
                        continue
                    comp_name = competitor.get('name')
                    program = competitor.get('exe')
                    params = competitor.get('params', {})
                    
                    dataset_path = os.path.join(input_folder_path, graph_file)
                    suffix = '_r' if reverse else ''
                    output = os.path.join(input_folder_path.replace('input', 'output'), f'{file_name}', f'{file_name}_{comp_name}{suffix}.json')
                    if not os.path.exists(os.path.dirname(output)):
                        os.makedirs(os.path.dirname(output))
                    if skip and os.path.exists(output):
                        continue
                    
                    # Build complete config with all parameters
                    full_config = {
                        'input': dataset_path,
                        'output': output,
                        'reverse_weight': reverse,
                        **params
                    }
                    
                    # Write config JSON
                    config_path = write_config_json(full_config)
                    
                    try:
                        # All algorithms now use single config file parameter
                        cmd = [program, config_path]
                        subprocess.run(cmd, check=True)
                        result = json.load(open(output))
                        logger.info(f'{comp_name:<15}: time: {result["time"]:.6f}s, density: {result["density"]:.6f}')
                    finally:
                        # Clean up config file
                        if os.path.exists(config_path):
                            os.remove(config_path)
    logger.success('All done!')
