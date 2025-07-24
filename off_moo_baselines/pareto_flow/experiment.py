import os 
import sys 
import wandb 
import torch 
import numpy as np 
import pandas as pd 
import datetime 
import json 
from copy import deepcopy
BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", ".."
)
sys.path.append(BASE_PATH)

import off_moo_bench as ob
from off_moo_baselines.pareto_flow.gfmo_args import parse_args
from utils import set_seed
from off_moo_bench.task_set import *
from off_moo_baselines.pareto_flow.gfmo_experiments import train_proxies, train_flow_matching, sampling

def run(config: dict, args):
    if config["task"] in ALLTASKSDICT.keys():
        config["task"] = ALLTASKSDICT[config["task"]]
    results_dir = os.path.join(config['results_dir'], 
                               f"{config['model']}-{config['train_mode']}-{config['task']}")
    config["results_dir"] = results_dir 
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f"-ts-{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"
    run_name = f"{config['model']}-{config['train_mode']}-seed{config['seed']}-{config['task']}"
    
    logging_dir = os.path.join(config['results_dir'], run_name + ts_name)
    config["results_dir"] = logging_dir
    os.makedirs(logging_dir, exist_ok=True)

    if config['use_wandb']:
        if 'wandb_api' in config.keys():
            wandb.login(key=config['wandb_api'])

        wandb.init(
            project="Offline-MOO",
            name=run_name + ts_name,
            config=config,
            group=f"{config['model']}-{config['train_mode']}",
            job_type=config['run_type'],
            mode="online",
            dir=os.path.join(config['results_dir'], '..')
        )
    
    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(config, f, indent=4)

    set_seed(config['seed'])

    task = ob.make(config['task'])
    
    model_save_dir = config['model_save_dir']
    os.makedirs(model_save_dir, exist_ok=True)

    train_proxies(config, args, task)
    train_flow_matching(config, args, task)
    sampling(config, args, task)
    
if __name__ == "__main__":
    from utils import process_args 
    config = process_args(return_dict=True)
    args = parse_args()
    save_dir = "./"
    results_dir = os.path.join(save_dir, "results")
    model_save_dir = os.path.join(save_dir, "model")
    
    config["results_dir"] = results_dir
    config["model_save_dir"] = model_save_dir
    run(config, args)