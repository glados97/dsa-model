import torch
import torchvision
from torch import nn
from attentionmodel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMs(nn.Module):
    def __init__(self, encoder_dim, attention_dim, embed_dim, decoder_dim, dic_size, dropout=0.5):
        super(LSTMs, self).__init__()

        # dimensions
        self.encoder_dim = encoder_dim # 512
        self.attention_dim = attention_dim  # 512
        self.embed_dim = embed_dim  # 512
        self.decoder_dim = decoder_dim  # 512
        self.dic_size = dic_size  # 1000+

        #attention
        self.attention = Attention(attention_dim, encoder_dim, decoder_dim)

        # embedding
        self.embedding = nn.Embedding(dic_size, embed_dim)

        # lstm
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        # intialization layer h and c for lstm module
        self.h = nn.Linear(encoder_dim, decoder_dim)
        self.c = nn.Linear(encoder_dim, decoder_dim)

        # fully connected layers
        self.fc_sig = nn.Linear(decoder_dim, encoder_dim)
        self.fc_dic = nn.Linear(decoder_dim, dic_size)

        # dropout and sigmoid
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        
        # weight initialization
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                nn.init.constant_(m.bias, 0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, encoder_out, encoder_cap, cap_len):

        # flatten image to size (batch_size, sumofpixcel, encoder_dim)
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(
            encoder_out.size(0), -1, encoder_out.size(-1)) # n*(h*w)*enc
        
        num_pixels = encoder_out.size(1) # h*w
        dic_size = self.dic_size # dic

        # sort data in length of caption (useful in the for loop of LSTMs to reduce time)
        sorted_cap_len, sorted_index = torch.sort(cap_len.squeeze(1), dim=0, descending=True) # n*1
        encoder_out = encoder_out[sorted_index]  # n*(h*w)*enc
        encoder_cap = encoder_cap[sorted_index]  # n*maxL

        # embedding
        cap_embedding = self.embedding(encoder_cap)  # n*maxL*emb

        # initializa LSTM Cell
        mean_encoder_out = encoder_out.mean(dim=1)  # n*enc
        h = self.h(mean_encoder_out)  # n*dec
        c = self.c(mean_encoder_out)  # n*dec

        # leave the last work <end>
        decoder_len = (sorted_cap_len - 1).tolist()  # n*1
        max_length = max(decoder_len)

        # initialize the output predictions and alpha
        # print(batch_size)
        # print(max_length)
        # print(dic_size)
        # print(dic_size)
        predictions = torch.zeros(
            batch_size, max_length, dic_size).to(device)  # n*maxL*dic
        alphas = torch.zeros(batch_size, max_length,
                             num_pixels).to(device)  # n*maxL*(h*w)

        # loop over the max length of caption
        for i in range(max_length):
            # the sub batch have length of caption greater than i
            # should be put into i^th LSTM
            subatch_index = sum([l > i for l in decoder_len]) # n'

            # get attention area and alpha
            # ========================================
            attention_area, alpha = self.attention(encoder_out[:subatch_index], h[:subatch_index])
            # attention_area n'*enc
            # alpha n'*(h*w)
            # ========================================


            # ========================================
            # gate scale
            mask = self.fc_sig(h[:subatch_index])  # n'*enc
            softmask = self.sigmoid(mask)  # n'*enc
            attentioned_out = softmask * attention_area  # n'*enc
            # ========================================

            # run LSTM
            # ========================================
            # concate the captions and attentioned area
            # n'*emb cat n'*enc = n' * (emb + enc)
            xt = torch.cat(
                [cap_embedding[:subatch_index, i, :], attentioned_out], dim=1)
            # run LSTMcell
            h, c = self.lstm(xt , (h[:subatch_index], c[:subatch_index]))
            preds = self.fc_dic(self.dropout(h))  # n'*dic
            # ========================================

            #append result
            predictions[:subatch_index, i, :] = preds  # n'*dic
            alphas[:subatch_index, i, :] = alpha  # n'*(h*w)

        return predictions, encoder_cap, decoder_len, alphas, sorted_index

    def predict(self, encoder_out, word_map):
        k = 3
        # flatten image to size (batch_size, sumofpixcel, encoder_dim)
        encoded_size = encoder_out.size(1)
        encoder_out = encoder_out.view(
            encoder_out.size(0), -1, encoder_out.size(-1))  # 1*(h*w)*enc

        num_pixels = encoder_out.size(1)  # h*w
        dic_size = len(word_map)  # dic

        # sort data in length of caption (useful in the for loop of LSTMs to reduce time)
        encoder_out = encoder_out.expand(
            k, num_pixels, encoder_out.size(2))  # k*(h*w)*enc

        # initialize first word for k same image is <start>
        prewords = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # k*1

        # top k captions
        topkcap = prewords  # k*1

        # top k score corresponding to top k captions
        topkscore = torch.zeros(k, 1).to(device)  # k*1

        # top k alpha corresponding to top k captions
        topkalpha = torch.ones(k, 1, encoded_size, encoded_size).to(device)

        # log for top k
        topkcap_all = list()
        topkscore_all = list()
        topkalpha_all = list()

        # the length of word we are going to predict
        step = 1

        # initialize the h and c based on encodered out
        h = self.h(encoder_out.mean(dim=1))
        c = self.c(encoder_out.mean(dim=1))

        # while we have not get k captions
        while True:
            # SAME as that in the forward
            # =========================================
            # embedding
            embedding = self.embedding(prewords)  # s*1*emb
            embedding = embedding.squeeze(1)  # s*emb
            attention_area, alpha = self.attention(encoder_out, h)
            alpha = alpha.view(-1, encoded_size, encoded_size)
            mask = self.fc_sig(h)
            softmask = self.sigmoid(mask)
            attentioned_out = softmask * attention_area
            xt = torch.cat([embedding, attentioned_out], dim=1)
            h, c = self.lstm(xt, (h, c))
            preds = self.fc_dic(h)  # s*dic
            #=========================================

            # soft max for the scores of predicts
            preds = F.log_softmax(preds, dim=1)  # s*dic

            # calculate all the scores that with previous predicted word
            # and the current word
            preds = topkscore.expand_as(preds) + preds  # s*dic

            # in the first, all the current words are the same("<begin>")
            # thus no need to look at all
            # otherwise, choose the top k in the map
            if step == 1:
                topkscore, topkword = preds[0].topk(k, 0, True, True)
            else:
                topkscore, topkword = preds.view(-1).topk(k, 0, True, True)

            # get the index among the k captions
            precapinx = topkword / dic_size
            # get the ibdex for the next word for the captions
            nexcapinx = topkword % dic_size

            # append the word and alpha(attentiob mask) to the captions
            topkcap = torch.cat(
                [topkcap[precapinx], nexcapinx.unsqueeze(1)], dim=1)  # s*(step + 1)
            topkalpha = torch.cat(
                [topkalpha[precapinx], alpha[precapinx].unsqueeze(1)], dim=1)  # s*(step + 1)*-1*-1
            h = h[precapinx]
            c = c[precapinx]
            encoder_out = encoder_out[precapinx]

            # calculate the index of the captions which is not endding
            nonendinx = []
            for idx, nexcap in enumerate(nexcapinx):
                if nexcap.cpu().numpy() != word_map['<end>']:
                    nonendinx.append(idx)

            # calculate the index of the captions which is endded
            nonendset = set(nonendinx)
            allset = set(range(len(nexcapinx)))
            nonendset = set(nonendinx)
            endinx = list(allset - nonendset)

            # appended the ended captions to the result and decrement k
            if len(endinx) > 0:
                topkcap_all.extend(topkcap[endinx].tolist())
                topkalpha_all.extend(topkalpha[endinx].tolist())
                topkscore_all.extend(topkscore[endinx])
                k -= len(endinx)

            # if already find k captions, break and to find the max
            if k == 0:
                break

            # update all the list, state and so on
            topkcap = topkcap[nonendinx]
            topkalpha = topkalpha[nonendinx]
            topkscore = topkscore[nonendinx].unsqueeze(1)
            h = h[nonendinx]
            c = c[nonendinx]
            encoder_out = encoder_out[nonendinx]
            prewords = nexcapinx[nonendinx].unsqueeze(1)

            # if length too long, break
            if step > 30:
                break
            step += 1

        # find the max index based on the scores
        # return the best captions and its corresponding alpha by
        # max index
        maxinx = topkscore_all.index(max(topkscore_all))
        bestcap = topkcap_all[maxinx]
        bestalpha = topkalpha_all[maxinx]
        return bestcap, bestalpha
