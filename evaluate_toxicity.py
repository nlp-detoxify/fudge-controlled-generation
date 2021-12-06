import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import namedtuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model, MarianTokenizer, MarianMTModel

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *
from predict_toxicity import predict_toxicity, predict_toxicity_gpt

def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    os.makedirs(args.save_dir, exist_ok=True)
    # combine GPT with toxic conditioning model
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_string)
    gpt_tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    gpt_pad_id = gpt_tokenizer.encode(PAD_TOKEN)[0]
    gpt_model = AutoModelWithLMHead.from_pretrained(args.model_string).to(args.device)
    gpt_model.eval()

    conditioning_model = None
    if not args.gpt:
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        model_args = checkpoint['args']
        conditioning_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
        conditioning_model.load_state_dict(checkpoint['state_dict'])
        conditioning_model = conditioning_model.to(args.device)
        conditioning_model.eval()

        if args.verbose:
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.ckpt, checkpoint['epoch']))
            print('num params', num_params(conditioning_model))

    inputs = []
    with open(args.in_file, 'r') as rf:
        for line in rf:
            inputs.append(line.strip())

    save_file = 'toxicity_results.txt'

    if args.split:
        save_file = f'toxicity_results_{args.group}of{args.num_groups}.txt'
        start = len(inputs)//args.num_groups * args.group
        end = len(inputs)//args.num_groups * (args.group+1) if args.group != args.num_groups-1 else len(inputs)
        inputs = inputs[start:end]

    with open(os.path.join(args.save_dir, save_file), 'a') as f:
        for inp in tqdm(inputs, total=len(inputs)):
            # skip empty input
            if len(inp) == 0:
                continue
            result = None
            if args.gpt:
                result = predict_toxicity_gpt(gpt_model, 
                        gpt_tokenizer, 
                        [inp], 
                        [], # condition_words
                        dataset_info, 
                        precondition_topk=args.precondition_topk,
                        postcondition_topk=args.precondition_topk,
                        # do_sample=args.do_sample,
                        length_cutoff=args.length_cutoff,
                        condition_lambda=args.condition_lambda,
                        device=args.device)
            else:
                result = predict_toxicity(gpt_model, 
                        gpt_tokenizer, 
                        conditioning_model, 
                        [inp], 
                        [], # condition_words
                        dataset_info, 
                        precondition_topk=args.precondition_topk,
                        postcondition_topk=args.precondition_topk,
                        # do_sample=args.do_sample,
                        length_cutoff=args.length_cutoff,
                        condition_lambda=args.condition_lambda,
                        device=args.device)
            result = result[0]
            result = result.replace('\n', ' ')
            # print(result)
            f.write(result + '\n')


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='gpt2-medium')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, required=True, help='save results')

    parser.add_argument('--in_file', type=str, default=None, required=True, help='file containing text to run pred on')
    # TODO used to be 200; doing 50 bc PPLM does 10
    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample or greedy; only greedy implemented')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=512, help='max length')

    parser.add_argument('--split', action='store_true', default=False, help='split input into groups')
    parser.add_argument('--num_groups', type=int, default=1, help='number of groups to split input into')
    parser.add_argument('--group', type=int, default=0, help='which input group: 0 through num_groups-1')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--gpt', action='store_true', default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)