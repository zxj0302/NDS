from src import *

cfg = yaml.safe_load(open('./config.yaml'))['synthetic']
skip_list = yaml.safe_load(open('./skip_list.yaml'))['failed']
run(cfg, skip_list, skip_existing=False)