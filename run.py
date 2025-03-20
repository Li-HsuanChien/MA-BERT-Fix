import cfgs.config as config
import argparse, yaml
import random
from easydict import EasyDict as edict


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Bilinear Args')

    parser.add_argument('--run', dest='run_mode',
                        choices=['train', 'val', 'test',''],
                        help='{train, val, test}',
                        type=str, required=True)

    parser.add_argument('--mode', dest='mode',
                        choices=['maa'],
                        help='{maa, ...}',
                        default='maa', type=str)

    parser.add_argument('--dataset', dest='dataset',
                        choices=['imdb', 'yelp_13', 'yelp_14', 'google'],
                        help='{imdb, yelp_13, yelp_14, google}',
                        default='imdb', type=str)

    parser.add_argument('--gpu', dest='gpu',
                        help="gpu select, eg.'0, 1, 2'",
                        type=str,
                        default="0,1")

    parser.add_argument('--seed', dest='seed',
                        help='fix random seed',
                        type=int,
                        default=random.randint(0, 99999999))

    parser.add_argument('--version', dest='version',
                        help='version control',
                        type=str,
                        default="default")
    parser.add_argument('--attributes', dest='attributes',
                        choices=['usr_prd', 'usr_ctgy', 'prd_ctgy', 'usr_prd_ctgy', 'kw'],
                        help='{usr_prd, usr_ctgy, prd_ctgy, usr_prd_ctgy, kw}',
                        default='usr_prd', type=str)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    __C = config.__C

    args = parse_args()
    args_dict = edict({**vars(args)})
    config.add_edit(args_dict, __C)
    config.proc(__C)
    print('Hyper Parameters:')
    config.config_print(__C)

    from common.trainer_maa import MAATrainer
    execution = MAATrainer(__C)
    execution.run(__C.run_mode)


