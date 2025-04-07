from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from traj_predictor.interaction_model import UQnet
from traj_predictor.losses import *
from utils.args_loading import *
from cl_data_stream.traj_dataset import InteractionDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
abs_dir = data_dir+'/'



def store_interaction_loaders(self, scenario_id):
    scenario_info = {1:'JT2', 2:'JT3', 3:'JT4', 4:'JT5', 5:'JT6', 6:'JT7', 7:'JT8'}
    scenario_index = scenario_id ## choose scenario
    scenario_name = scenario_info[scenario_index]
    print(f"Scenario Index: {scenario_index+1}, Scenario Name: {scenario_name}")
    trainset = InteractionDataset(['train'], scenario_name,'train', paralist, paralist['mode'], filters=False)
    train_loader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
    return train_loader



class Joint_INTERACTION():
    NAME = 'joint-interaction'
    SETTING = 'domain-il'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, args) -> None:
        super(Joint_INTERACTION, self).__init__()
        self.args = args
    
    
    def get_data_loaders(self, task_id) -> Tuple[DataLoader, DataLoader]: 
        train = store_interaction_loaders(self, task_id)
        return train

    @staticmethod
    def get_backbone():
        return UQnet(paralist, test=True, drivable=False).to(device)


    @staticmethod
    def get_loss():
        return OverAllLoss(paralist).to(device)
    