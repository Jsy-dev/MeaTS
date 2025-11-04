import argparse
import os
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main
from utils.str2bool import str2bool

parser = argparse.ArgumentParser(description='Metaformer')


# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')


# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='Mformer',
                    help='model name, options: [Mformer]')


# data loader

parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./all_six_datasets/weather/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')


parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='close', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# forecasting task
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio')


parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--n_feat', type=int, default=21, help='num of features')
parser.add_argument('--n_outfeat', type=int, default=21, help='output size')
parser.add_argument('--n_layer_enc', type=int, default=1, help='num of heads')
parser.add_argument('--n_layer_dec', type=int, default=0, help='num of heads')
parser.add_argument('--n_embd', type=int, default=64, help='num of heads')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--n_proj', type=int, default=3, help='num of heads')
parser.add_argument('--attn_pdrop', type=float, default=0, help='num of heads')
parser.add_argument('--resid_pdrop', type=float, default=0, help='num of heads')
parser.add_argument('--mlp_hidden_times', type=int, default=4, help='num of heads')
parser.add_argument('--block_activate', type=str, default='GELU', help='num of heads')
parser.add_argument('--max_len', type=int, default=2048, help='num of heads')
parser.add_argument('--patch_size', type=int, default=8, help='the patch size')
parser.add_argument('--patch_stride', type=int, default=4, help='the patch stride')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--inverse', type=int, default=0, help='head dropout')


parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate') #0.00001
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')

args = parser.parse_args()



# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print(torch.cuda.is_available())
# print('Args in experiment:')
# print(args)
if __name__ == '__main__':

    Exp = Exp_Main

    # setting record of experiments
    setting = f'{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_fl{args.pred_len}_dim{args.n_feat}_em{args.n_embd}_pl{args.patch_size}_ps{args.patch_stride}_ms{args.mask_rate}'

    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.test(setting)
    #
    # if args.do_predict:
    #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #     exp.predict(setting, True)


    torch.cuda.empty_cache()
