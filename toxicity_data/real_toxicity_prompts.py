import json
import os
import sys


with open('./realtoxicityprompts-data/prompts.jsonl','r') as json_file:
    json_list = list(json_file)

prompts = open('./rtp_prompts.txt','w')
toxicity = open('./rtp_toxicity.txt','w')
continuation = open('./rtp_continuation.txt','w')
cont_tox = open('./rtp_continuationtoxicity.txt','w')

for json_str in json_list:
    result = json.loads(json_str)
    # TODO replace newlines with spaces
    prompts.write(result[u'prompt'][u'text'].encode('utf-8')+'\n')
    toxicity.write(str(result[u'prompt'][u'toxicity'])+'\n')
    continuation.write(result[u'continuation'][u'text'].encode('utf-8')+'\n')
    cont_tox.write(str(result[u'continuation'][u'toxicity'])+'\n')


prompts.close()
toxicity.close()
continuation.close()
cont_tox.close()


# One text file where each line is the prompt
# Another text file where each line is the toxicity
