import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import defaultdict
import string
import csv

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params, pad_mask
from constants import *


def nontox_accuracy(sentences, tokenizer, model, device='cuda'):
    with torch.no_grad():
        total_nontoxic = 0
        for sent in tqdm(sentences, total=len(sentences)):
            input = tokenizer.encode(sent).to(device)
            nontoxic = (model(input)[0] == 0).cpu().item()
            total_nontoxic += nontoxic
        return total_nontoxic / len(sentences) # avg accuracy


def perplexity(sentences, tokenizer, model, device='cuda'):
    # calculate perplexity 
    with torch.no_grad():
        ppl = []
        sos_token = tokenizer.decode([0])
        for sentence in tqdm(sentences, total=len(sentences)):
            full_tensor_input = tokenizer.encode(sos_token + sentence.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)
            full_loss = model(full_tensor_input, labels=full_tensor_input)[0].mean()
            ppl.append(torch.exp(full_loss).flatten().cpu().item())
    return np.mean(ppl), np.std(ppl)


def grammaticality(sentences, tokenizer, model, device='cuda'):
    with torch.no_grad():
        total_good = 0
        for sent in tqdm(sentences, total=len(sentences)):
            good_prob = F.softmax(model(tokenizer.encode(sent, return_tensors='pt').to(device))[0].flatten(), dim=0)[1].item()
            total_good += good_prob
        return total_good / len(sentences) # avg probability of grammaticality according to model


def distinctness(sentences):
    d1 = set()
    d2 = set()
    d3 = set()
    total_words = 0
    for sentence in sentences:
        o = sentence.split(' ')
        total_words += len(o)
        d1.update(o)
        for i in range(len(o) - 1):
            d2.add(o[i] + '_' + o[i+1])
        for i in range(len(o) - 2):
            d3.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
    return len(d1) / total_words, len(d2) / total_words, len(d3) / total_words


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--log_file', type=str, default="toxicity_data/toxicity_results.txt", help='where to load results from')
    parser.add_argument('--prompt_file', type=str, default='toxicity_data/rtp_prompts.txt', help='where to load prompts from')
    parser.add_argument('--batch_size', type=int, default=8, help='max samples at a time')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    tw_topic_match_c_total = 0
    category_totals_c = defaultdict(lambda:0)
    results = []
    with open(args.log_file, 'r') as rf:
        for line in rf:
            results.append(line.strip())

    # toxic_tokenizer = AutoTokenizer.from_pretrained('???')
    # toxic_model = AutoModelForSequenceClassification.from_pretrained('????').to(args.device)
    # toxic_model.eval()
    # print('non-toxicity:', nontox_accuracy(results, toxic_tokenizer, toxic_model, device=args.device))

    print('distinctness:', distinctness(results))

    grammar_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA')
    grammar_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA').to(args.device)
    grammar_model.eval()
    print('grammaticality:', grammaticality(results, grammar_tokenizer, grammar_model, device=args.device))

    eval_tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    eval_model = AutoModelWithLMHead.from_pretrained('openai-gpt').to(args.device)
    eval_model.eval()
    print('GPT perplexity:', perplexity(results, eval_tokenizer, eval_model, device=args.device))

    eval_tokenizer = AutoTokenizer.from_pretrained('transfo-xl-wt103')
    eval_model = AutoModelWithLMHead.from_pretrained('transfo-xl-wt103').to(args.device)
    eval_model.eval()
    print('TFXL perplexity:', perplexity(results, eval_tokenizer, eval_model, device=args.device))
