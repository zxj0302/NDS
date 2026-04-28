import json
import yaml
from pathlib import Path
from natsort import natsorted

#================ CHANGE IF NEEDED ================
goal = 'CEP_PRUNING_QPBO_MIP_CONSTRAIN_B.json'
output_file = f'results/skip/{goal.split(".")[0]}.yaml'
DEFAULT_TIME_BOUND_SECONDS = 3599.0
#==================================================

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
        # only process ER folder
        if subfolder.name != "ER":
            continue

        if not subfolder.is_dir():
            continue
            
        # Iterate through all subsubfolders
        for subsubfolder in natsorted(subfolder.iterdir()):
            if not subsubfolder.is_dir():
                continue
                
            # Construct path to the JSON file
            json_file = subsubfolder / goal
            
            # Check if file exists
            if not json_file.exists():
                print(f"Warning: {json_file} not found")
                failed.append(subsubfolder.name)
                continue
            
            # Read and check the JSON file
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check if 'Fail' substring exists in config.info
                info = data.get('config', {}).get('info', '')
                runtime = data.get('time', None)
                
                if ('Terminate' not in info) and ('Fail' in info):
                    failed.append(subsubfolder.name)
                elif runtime is not None and float(runtime) >= DEFAULT_TIME_BOUND_SECONDS:
                    failed.append(subsubfolder.name)
                else:
                    success.append(subsubfolder.name)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {json_file}: {e}")
    
    # Output results to YAML file
    results = {
        'summary': {
            'total_failed': len(failed),
            'total_success': len(success),
            'total': len(failed) + len(success)
        },
        'failed': failed,
        'success': success
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Success: {len(success)}, Failed: {len(failed)}")
    
    return results

if __name__ == '__main__':
    results = check_synthetic_results()
