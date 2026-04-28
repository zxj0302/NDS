from src import *

cfg = yaml.safe_load(open('./config.yaml'))['test']
# skip_list = yaml.safe_load(open('./temp/skip.yaml'))['filter1']
skip_list = []
run(cfg, skip_list, skip_existing=False)