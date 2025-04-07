from traj_predictor.interaction_model import UQnet
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import torchvision.datasets as dataset
from torch.optim.lr_scheduler import StepLR
import datetime
from scipy.special import expit
from traj_predictor.utils import *
from cl_data_stream.seq_dataset import *
from traj_predictor.losses import *
from traj_predictor.evaluation import *
from utils.args_loading import *
from absl import logging
logging._warn_preinit_stderr = 0
logging.warning('Worrying Stuff')
import argparse
from utils.metrics import *
from experiments.testing_1_task import *
from utils.args_loading import scenario_info
    
def main():
    parser = argparse.ArgumentParser(description='Testing process of CL', allow_abbrev=False)
    parser.add_argument('--num_tasks', type=int, default=10, help='The number of continuous tasks to be tested.')
    parser.add_argument('--continual_learning',  type=bool, default=True, help='Whether the CL strategies are used in training.')
    parser.add_argument('--model', type=str, default='joint')
    parser.add_argument('--buffer_size', type=str, default='2000')
    parser.add_argument('--batch_size', type=int, default= 8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default= device)
    args = parser.parse_args()
    args.scenario_info = scenario_info

    print("args.num_tasks:", args.num_tasks)
    print("args.continual_learning:", args.continual_learning)
    print("args.model:", args.model)
    

    total_fde_lists = []
    total_mr_lists = []

    task_id = 9
    ret_fde_list, ret_mr_list = test_1_task(task_num = task_id+1, args = args)
    total_fde_lists.append(ret_fde_list)
    total_mr_lists.append(ret_mr_list)

    if task_id>0:
        fde_bwt = e_bwt(total_fde_lists, task_id+1)
        mr_bwt = e_bwt(total_mr_lists, task_id+1)
        with open(result_dir+'/'+str(task_id+1)+'_continual_tasks_'+args.method_name+'.txt', "a") as file:
            file.write("\n FDE backward transfer:"+str(fde_bwt))# smaller is better, it can be negative
            file.write("\n Missing Rate backward transfer: "+str(mr_bwt))
            file.close()


if __name__ == '__main__':
    main()
    