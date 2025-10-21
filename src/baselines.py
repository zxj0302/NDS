from loguru import logger
import os.path
import subprocess
import json
from tqdm import tqdm
import natsort


def run(config, single=True):
    if single:
        datasets = config.get('input')
        output_folder = config.get('output')
        reverse = config.get('weight_reverse', False)
        for dataset in natsort.natsorted(datasets):
            logger.info(f'Running on dataset: {dataset}')

            dataset_name = dataset.split('/')[-1].split('.')[0]
            for competitor in config.get('competitors'):
                comp_name = competitor.get('name')
                competitor['params']['input'] = dataset
                output = os.path.join(output_folder, f'{dataset_name}', f'{dataset_name}_{comp_name}{'_r' if reverse else ''}.json')
                if not os.path.exists(os.path.dirname(output)):
                    os.makedirs(os.path.dirname(output))
                competitor['params']['output'] = output
                competitor['params']['reverse'] = reverse

                # comp_name is the function name
                func = globals().get(comp_name)
                result = func(competitor)

                logger.info(f'{comp_name:<9}: time: {result[0]:.6f}s, density: {result[1]:.6f}')
    else:
        reverse = config.get('weight_reverse', False)
        for input_folder in config.get('input_folder'):
            logger.info(f'Running on input folder: {input_folder}')
            for graph_file in tqdm(natsort.natsorted([f for f in os.listdir(input_folder) if f.endswith('.txt')])):
                print()
                logger.info(f'Processing graph file: {graph_file}')
                file_name = graph_file.split('.')[0]
                for competitor in config.get('competitors'):
                    comp_name = competitor.get('name')
                    competitor['params']['input'] = os.path.join(input_folder, graph_file)
                    output = os.path.join(input_folder.replace('input', 'output'), f'{file_name}', f'{file_name}_{comp_name}{'_r' if reverse else ''}.json')
                    if not os.path.exists(os.path.dirname(output)):
                        os.makedirs(os.path.dirname(output))
                    competitor['params']['output'] = output
                    competitor['params']['reverse'] = reverse

                    # comp_name is the function name
                    func = globals().get(comp_name)
                    result = func(competitor)

                    logger.info(f'{comp_name:<9}: time: {result[0]:.6f}s, density: {result[1]:.6f}')

    logger.success('All done!')


def LNDS(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    max_neg = params.get('max_neg')
    num_iter = params.get('num_iter', 1)
    subprocess.run([program, input, output, "1" if reverse else "0", str(max_neg), str(num_iter)], check=True)
    # readin the output file
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def LNDS_EP(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    max_neg = params.get('max_neg')
    num_iter = params.get('num_iter', 1)
    subprocess.run([program, input, output, "1" if reverse else "0", str(max_neg), str(num_iter)], check=True)
    # readin the output file
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def GNDS(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    max_neg = params.get('max_neg')
    max_local_optima = params.get('max_local_optima')
    num_iter = params.get('num_iter', 1)
    subprocess.run([program, input, output, "1" if reverse else "0", str(max_neg), str(max_local_optima), str(num_iter)], check=True)
    # readin the output file
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def QPBO_MIP(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    dinkelbach_iterations = params.get('dinkelbach_iterations')
    epsilon = params.get('epsilon')
    num_iter = params.get('num_iter', 1)
    subprocess.run([program, input, output, "1" if reverse else "0", str(dinkelbach_iterations), str(epsilon), str(num_iter)], check=True)
    # readin the output file
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def NEG_DSD(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    C = params.get('C')
    num_iter = params.get('num_iter', 1)
    subprocess.run([program, input, output, "1" if reverse else "0", str(C), str(num_iter)], check=True)
    # readin the output file
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
    # readin the output file
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']
