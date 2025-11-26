from loguru import logger
import os.path
import subprocess
import json
from tqdm import tqdm
import natsort


def run(config, single=True, skip=True):
    if single:
        datasets = config.get('input')
        output_folder = config.get('output')
        reverse = config.get('weight_reverse', False)
        for dataset in natsort.natsorted(datasets):
            if dataset.get('toggle') is False:
                continue
            dataset = dataset.get('path')
            logger.info(f'Running on dataset: {dataset}')
            dataset_name = dataset.split('/')[-1].split('.')[0]
            for competitor in config.get('competitors'):
                if not competitor.get('toggle', False):
                    continue
                comp_name = competitor.get('name')
                competitor['params']['input'] = dataset
                output = os.path.join(output_folder, f'{dataset_name}', f'{dataset_name}_{comp_name}{'_r' if reverse else ''}.json')
                if not os.path.exists(os.path.dirname(output)):
                    os.makedirs(os.path.dirname(output))
                if skip and os.path.exists(output):
                    continue
                competitor['params']['output'] = output
                competitor['params']['reverse'] = reverse

                # comp_name is the function name
                func = globals().get(comp_name)
                result = func(competitor)  # type: ignore
                logger.info(f'{comp_name:<9}: time: {result[0]:.6f}s, density: {result[1]:.6f}')
    else:
        reverse = config.get('weight_reverse', False)
        for input_folder in config.get('input_folder'):
            if input_folder.get('toggle') is False:
                continue
            input_folder = input_folder.get('path')
            logger.info(f'Running on input folder: {input_folder}')
            for graph_file in tqdm(natsort.natsorted([f for f in os.listdir(input_folder) if f.endswith('.txt')])):
                logger.info(f'Processing graph file: {graph_file}')
                file_name = os.path.splitext(graph_file)[0]
                for competitor in config.get('competitors'):
                    if not competitor.get('toggle', False):
                        continue
                    comp_name = competitor.get('name')
                    competitor['params']['input'] = os.path.join(input_folder, graph_file)
                    output = os.path.join(input_folder.replace('input', 'output'), f'{file_name}', f'{file_name}_{comp_name}{'_r' if reverse else ''}.json')
                    if not os.path.exists(os.path.dirname(output)):
                        os.makedirs(os.path.dirname(output))
                    if skip and os.path.exists(output):
                        continue
                    competitor['params']['output'] = output
                    competitor['params']['reverse'] = reverse

                    # comp_name is the function name
                    func = globals().get(comp_name)
                    result = func(competitor)  # type: ignore
                    logger.info(f'{comp_name:<9}: time: {result[0]:.6f}s, density: {result[1]:.6f}')
    logger.success('All done!')


def NEG_DSD(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    C_values = params.get('C_values', "1.0")
    num_iter = params.get('num_iter', 1)
    subprocess.run([program, input, output, "1" if reverse else "0", str(C_values), str(num_iter)], check=True)
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def DCSGreedy(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    num_iter = params.get('num_iter', 1)
    subprocess.run([program, input, output, "1" if reverse else "0", str(num_iter)], check=True)
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def CEP(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    max_neg = params.get('max_neg')
    max_local_optima = params.get('max_local_optima')
    do_peeling = params.get('do_peeling', False)
    num_iter = params.get('num_iter', 1)
    subprocess.run([program, input, output, "1" if reverse else "0", str(max_neg), str(max_local_optima), "1" if do_peeling else "0", str(num_iter)], check=True)
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def CEP_MIP(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    max_neg = params.get('max_neg')
    max_local_optima = params.get('max_local_optima')
    do_peeling = params.get('do_peeling', False)
    dinkelbach_iterations = params.get('dinkelbach_iterations')
    epsilon = params.get('epsilon')
    mip_time_limit = params.get('mip_time_limit', 300.0)
    use_binary = params.get('use_binary', True)
    num_iter = params.get('num_iter', 1)
    subprocess.run([program, input, output, "1" if reverse else "0", str(max_neg), str(max_local_optima), "1" if do_peeling else "0", str(dinkelbach_iterations), str(epsilon), str(mip_time_limit), "1" if use_binary else "0", str(num_iter)], check=True)
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def CEP_QPBO(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    max_neg = params.get('max_neg')
    max_local_optima = params.get('max_local_optima')
    do_peeling = params.get('do_peeling', False)
    step_size = params.get('step_size')
    dinkelbach_iterations = params.get('dinkelbach_iterations')
    epsilon = params.get('epsilon')
    mip_time_limit = params.get('mip_time_limit', 300.0)
    num_iter = params.get('num_iter', 1)
    subprocess.run([program, input, output, "1" if reverse else "0", str(max_neg), str(max_local_optima), "1" if do_peeling else "0", str(step_size), str(dinkelbach_iterations), str(epsilon), str(mip_time_limit), str(num_iter)], check=True)
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']