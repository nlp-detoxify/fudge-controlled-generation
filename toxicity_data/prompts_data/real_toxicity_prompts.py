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

with open('./realtoxicityprompts-data/prompts.jsonl','r') as json_file:
    json_list = list(json_file)

prompts = open('./rtp_prompts.txt','w')
toxicity = open('./rtp_toxicity.txt','w')
toxicity01 = open('./rtp_toxicity01.txt','w')
continuation = open('./rtp_continuation.txt','w')
cont_tox = open('./rtp_continuationtoxicity.txt','w')
cont_tox01 = open('./rtp_continuationtoxicity01.txt','w')

for json_str in json_list:
    result = json.loads(json_str)
    # TODO replace newlines with spaces
    
    prompts.write(result[u'prompt'][u'text'].replace("\n"," ")+'\n')
    toxicity.write(str(result[u'prompt'][u'toxicity'])+'\n')
    toxicity01.write(str(tox_round(result[u'prompt'][u'toxicity']))+'\n')
    continuation.write(result[u'continuation'][u'text'].replace("\n"," ")+'\n')
    cont_tox.write(str(result[u'continuation'][u'toxicity'])+'\n')
    cont_tox01.write(str(tox_round(result[u'continuation'][u'toxicity']))+'\n')


prompts.close()
toxicity.close()
toxicity01.close()
continuation.close()
cont_tox.close()
cont_tox01.close()