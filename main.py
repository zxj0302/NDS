from src import *

cfg = yaml.safe_load(open('./config.yaml'))['test']
# skip_list = yaml.safe_load(open('./skip_d.yaml'))['failed']
skip_list = []
run(cfg, skip_list, skip_existing=False)