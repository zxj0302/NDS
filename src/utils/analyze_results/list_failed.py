import json
import yaml
from pathlib import Path

def check_synthetic_results():
    """
    Iterate through output/synthetic folder and check CEP results.
    Categorizes subsubfolders into 'failed' and 'success' based on whether
    'Fail' appears in json['config']['info'].
    """
    synthetic_dir = Path('output/synthetic')
    failed = []
    success = []
    
    # Iterate through all subfolders (BA, ER, RGG, SBM, WS)
    for subfolder in synthetic_dir.iterdir():
        if not subfolder.is_dir():
            continue
            
        # Iterate through all subsubfolders
        for subsubfolder in subfolder.iterdir():
            if not subsubfolder.is_dir():
                continue
                
            # Construct path to the JSON file
            json_file = subsubfolder / 'CEP_PRUNING_QPBO_CEP_MIP_CONSTRAIN_CEP.json'
            
            # Check if file exists
            if not json_file.exists():
                print(f"Warning: {json_file} not found")
                continue
            
            # Read and check the JSON file
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check if 'Fail' substring exists in config.info
                info = data.get('config', {}).get('info', '')
                
                if 'Fail' in info:
                    failed.append(subsubfolder.name)
                else:
                    success.append(subsubfolder.name)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {json_file}: {e}")
    
    # Output results to YAML file
    results = {
        'failed': failed,
        'success': success,
        'summary': {
            'total_failed': len(failed),
            'total_success': len(success),
            'total': len(failed) + len(success)
        }
    }
    
    output_file = 'results_summary.yaml'
    with open(output_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Success: {len(success)}, Failed: {len(failed)}")
    
    return results

if __name__ == '__main__':
    results = check_synthetic_results()
