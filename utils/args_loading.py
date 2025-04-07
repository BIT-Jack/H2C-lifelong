import sys
import os
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')
from argparse import ArgumentParser
import torch
import numpy  # needed (don't change it)
import importlib


#file path
root_dir = ''#Please revised as your root file path
saved_dir = root_dir+'results/weights'
result_dir = root_dir+'results/logs'
data_dir = root_dir+'cl_dataset'


scenario_info = {0:'MA', 1:'FT', 2:'LN', 3:'ZS2', 4:'OF', 5:'EP0', 6:'GL', 7:'ZS0'}
#Training setting of UQnet:
paralist = dict(xmax = 23,#23,
                ymin = -12,#-12,
                ymax = 75,
                resolution = 0.5,
                nb_map_vectors = 5,
                nb_traj_vectors = 9,
                map_dim = 5,
                traj_dim = 8,
                nb_map_gnn = 5,
                nb_traj_gnn = 5, 
                nb_mlp_layers = 3,
                c_out_half = 32,
                c_mlp = 64,
                c_out = 96,
                encoder_nb_heads = 3,
                encoder_attention_size = 128,
                encoder_agg_mode = "cat",
                decoder_attention_size = 64,
                decoder_nb_heads = 3,
                decoder_agg_mode = "cat",
                decoder_masker = False,
                sigmax = 0.6,
                sigmay = 0.6,
                r_list = [2,4,8,16],
                kf = 1,
                model = 'densetnt',
                sample_range=1,
                use_masker=False, 
                lane2agent='lanegcn',
                #integrate which written in jupyter, lyl
                use_sem=False,
                mode='lanescore',
                prob_mode='ce',
                inference=False #when testing turn this into True to calculate uncertainty
                )



def args_loading():
    torch.set_num_threads(4)
    parser = ArgumentParser(description='CL for interactive behavior learning', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, default= 'seq-interaction')   # 'joint-interaction'    'seq-interaction'
    parser.add_argument('--model', type=str, 
                        default= 'der', 
                        help='Model name.')


    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=1, help='Run only a few forward steps per epoch')
    parser.add_argument('--train_task_num', type=int, default=8, help='The Number of Continual Tasks for Training')
    parser.add_argument('--buffer_size', type=int,default= 2000,
                                help='The size of the memory buffer.')

    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
    parser.add_argument('--lr', type=float, default= 0.001,
                        help='Learning rate.')
    parser.add_argument('--n_epochs', type=int, default= 1,
                        help='n_epochs.')
    parser.add_argument('--batch_size', type=int, default= 8,
                        help='Batch size.')
    parser.add_argument('--alpha', type=float, default= 1.0,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, default= 1.0,
                        help='Penalty weight.')
    parser.add_argument('--gamma', type=float, default= 0.5,
                        help='the added constant to solve QP in GEM')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default= device)
    
    #parameter for decide whether record the amount of replayed data in each task or not
    parser.add_argument('--replayed_rc', type=bool, default=False,
                        help='turn True for replayed data logging')
    
    parser.add_argument('--restart_training', type=bool, default=False,
                        help='True for restart')
    parser.add_argument('--restart_pre_task_num', type=int, default=1,
                        help='scenario index of restarted training')


    args = parser.parse_args()  # parameters for a specific method
    mod = importlib.import_module('cl_model.' + args.model)
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser(parser)
    args = parser.parse_args()

    return args
