import torch
import torchvision.models as models
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, attention_size, encoder_size, decoder_size):
        super(Attention, self).__init__()
        #attention_dim = 512  
        #decoder_dim = 512
        self.attention = nn.Linear(attention_size, 1)
        self.encoder = nn.Linear(encoder_size, attention_size)
        self.decoder = nn.Linear(decoder_size, attention_size)
        self.soft = nn.Softmax(1)

    def forward(self, einput, dinput):
        x1 = self.encoder(einput) #n * (h*w) * att_c
        x2 = self.decoder(dinput).unsqueeze(1) #n * 1 * att_c
        out = F.relu(x1+x2) # n* (h*w) * att_c
        out = self.attention(out).squeeze(2) #n * (h*w)
        out = self.soft(out) #n * (h*w)
        out1 = out.unsqueeze(2) # n * (h*w) *1
        # ([n*(h*w)*enc] * [n * (h*w) * 1]).sum(1) = n*enc
        weights = (einput*out1).sum(1)
        #weights dimension: n*enc
        #out(alpha) dimension: N * (H*W)
        return weights,out



        
