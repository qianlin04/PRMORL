from math import log
from lib.common_ptan import actions
import sys
import time

import torch
import torch.nn as nn

import copy
from types import SimpleNamespace
import wandb
import os

import lib.common_ptan as ptan
import argparse


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 args: SimpleNamespace):
        self.selector = selector
        self.args = copy.deepcopy(args)
        self.update(0)

    def update(self, time_step: int):
        eps = self.args.epsilon_start - time_step / (self.args.time_steps)
        self.selector.epsilon = max(self.args.epsilon_final, eps)
        

def save_model(net, args, name = '', ext=''):
        save_path = "Exps/{}/{}/".format(name, args.seed)
        model_name = "{}_{}_{}".format(args.scenario_name, args.name_model, ext)
        if not os.path.exists(save_path): 
            os.makedirs(save_path)
        torch.save(net.state_dict(), "{}{}.pkl".format(save_path, model_name))



class WandbWriter:
    def __init__(self, use_wandb, project, args, seed, group="default"):
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=project,
                config=args,
                name=f'{args.scenario_name}_{seed}',
                group=group,
            )
    
    def add_scalar(self, name, data, t=None):
        if self.use_wandb:
            wandb.log({name: data, 't': t})

    def log(self, *args, **kwargs):
        if self.use_wandb:
            wandb.log(*args, **kwargs)

    def log_table(self, table_name, **kwargs):
        if self.use_wandb:
            data = [list(v) for v in zip(*kwargs.values())]
            table = wandb.Table(data=data, columns = list(kwargs.keys()))
            wandb.log({table_name : table})
    
def make_config(parser, config):
    for key in vars(config):
        value = getattr(config, key)
        if type(value) == bool:
            #assert not value, "Default bool params should be set to false."
            if not value:
                parser.add_argument('--{}'.format(key), action='store_true')
            else:
                parser.add_argument('--{}'.format(key), action='store_false')
        else:
            parser.add_argument('--{}'.format(key),
                                type=type(value) if value is not None else str, default=value)
    return parser.parse_args()
