import json
import os
import sys

def tox_round(a):
    if a == None:
        return None
    if a > 1 or a < 0:
        raise ValueError("Value not between 0 and 1")
    if a >= 0.5:
        return 1
    return 0

with open('./realtoxicityprompts-data/generations/prompted/prompted_gens_gpt2_pplm.jsonl','r') as json_file:
    json_list = list(json_file)

generations = open('./pplm_generation.txt','w')
toxicity = open('./pplm_toxicity.txt','w')
toxicity01 = open('./pplm_toxicity01.txt','w')

for json_str in json_list:
    result = json.loads(json_str)
    result = result[u'generations'][0]
    # print(type(result[u'text']),result[u'text'])
    # print(type(result[u'text'].replace("\n"," ").encode('utf-8')),result[u'text'].replace("\n"," ").encode('utf-8'))
    generations.write(result[u'text'].replace("\n"," ").encode('utf-8')+'\n')
    toxicity.write(str(result[u'toxicity'])+'\n')
    toxicity01.write(str(tox_round(result[u'toxicity']))+'\n')

generations.close()
toxicity.close()
toxicity01.close()