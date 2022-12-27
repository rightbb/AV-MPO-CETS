import numpy as np
from rule_base import *
HPARAMS_REGISTRY = {}

class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


def parse_args_and_update_hparams(H, parser, s=None):
    args = parser.parse_args(s)
    valid_args = set(args.__dict__.keys())
    #print(args.hparam_sets)
    H.update(parser.parse_args(s).__dict__)

def add_arguments(parser):
    # utils
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--desc', type=str, default='test')
    parser.add_argument('--hparam_sets', '--hps', type=str)
    parser.add_argument('--gpu', type=str, default=None)

    # model
    parser.add_argument('--model', type=str, default='vmpo', help='{vmpo|ppo}')
    parser.add_argument('--state_rep', type=str, default='none' )#help='{none|lstm|trxl|gtrxl}')
    parser.add_argument('--n_latent_var', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--mem_len', type=int, default=10)

    # env
    parser.add_argument('--env_name', type=str, default='rooms_watermaze')
    parser.add_argument('--action_dim', type=int, default=9)
    parser.add_argument('--log_interval', type=int, default=40)
    parser.add_argument('--max_episodes', type=int, default=20000)
    parser.add_argument('--max_timesteps', type=int, default=30000)
    parser.add_argument('--update_timestep', type=int, default=100)#更新的时间
    parser.add_argument('--action_list', type=list, default=[])

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--gamma', type=float, default=0.96)
    parser.add_argument('--K_epochs', type=int, default=4)
    parser.add_argument('--eps_clip', type=float, default=0.3)

    parser.add_argument('--action_list_scheduling',type=list,default=rule_base)#['MaxDistance_InOrder','MinDistance_InOrder','MaxLoad_Mindistance_InOrder','MaxLoad_Maxdistance_InOrder','Minwaiting_InOrder','MinLoad_Mindistance_InOrder','MinLoad_Maxdistance_InOrder','MinLoadrate_Mindistance_InOrder'])
    #parser.add_argument('--action_list_scheduling',type=list,default=['Random','MinDistance_InOrder_dispart','MaxDistance_InOrder_dispart','MinLoadrate_Mindistance_InOrder','MinLoadrate_Mindistance_InOrder_part'])

    return parser
