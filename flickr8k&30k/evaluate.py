import torch
import torch.backends.cudnn as cudnn
import os
from torchvision import transforms
import torchvision.models as models
from dataset import CustomDataset
from helper import load_checkpoint, save_checkpoint
from torch import nn
from lstms import *
from modelcnn import *
import json
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
import nltk 

references = []
hypotheses = []
f = open('reference/reference.json',) 
references = json.load(f)
c = open('hypotesis/hypotheses.json',) 
hypotheses = json.load(c)

weights = (1.0/1.0, )
bleu1 = corpus_bleu(references, hypotheses, weights)
print("bleu1 score: ", bleu1)
with open('results/results.json', 'a') as f:
    json.dump(("bleu1 score:", bleu1), f)

weights=(1.0/2.0, 1.0/2.0,)
bleu2 = corpus_bleu(references, hypotheses, weights)
print("bleu2 score: ", bleu2)
with open('results/results.json', 'a') as f:
    json.dump(("bleu2 score:", bleu2), f)
            
weights=(1.0/3.0, 1.0/3.0, 1.0/3.0,)
bleu3 = corpus_bleu(references, hypotheses, weights)
print("bleu3 score: ", bleu3)
with open('results/results.json', 'a') as f:
    json.dump(("bleu3 score:", bleu3), f)

bleu4 = corpus_bleu(references, hypotheses)
print("bleu4 score: ", bleu4)
with open('results/results.json', 'a') as f:
    json.dump(("bleu4 score:", bleu4, "/n"), f)









listToStrR = ' '.join([str(elem) for elem in references]) 
listToStrH = ' '.join([str(elem) for elem in hypotheses]) 

meteor = round(meteor_score([listToStrH], listToStrR), 2) # list of references
print(meteor, "\n")
