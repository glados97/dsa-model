import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
import skimage.transform
from PIL import Image
from dataset import *
from torch import nn
from helper import *
import numpy as np
import random
import imageio
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import torch.nn.functional as F

## Parameters ##
output_path = "./test_caption_out"
attention_output_path = "./test_attention_out"
dataset = "flickr8k"
num_imgs = 40
check_point_name = "best_checkpoint_flickr8k.pth.tar"
dictionary_json_path = os.path.join("./preprocess_out", 'DICTIONARY_WORDS_' + dataset + '.json')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load checkpoint ##
encoder, decoder, decoder_opt, last_epoch, best_bleu_score = load_checkpoint(check_point_name)

word_map = None
with open(dictionary_json_path, 'r') as file:
    word_map = json.load(file)
dict_len = len(word_map)
# word_dict.to(device)
reversed_word_dict = {}
for k, v in word_map.items():
    reversed_word_dict[v] = k


# reversed_word_dict.to(device)

def search_caption(img_tensor):
    encoder_out = encoder(img_tensor)  # 1*14*14*512
    k = 8
    # flatten image to size (batch_size, sumofpixcel, encoder_dim)
    encoded_size = encoder_out.size(1)
    encoder_out = encoder_out.view(
        encoder_out.size(0), -1, encoder_out.size(-1))  # 1*(h*w)*enc

    num_pixels = encoder_out.size(1)  # h*w
    dic_size = decoder.dic_size  # dic

    # sort data in length of caption (useful in the for loop of LSTMs to reduce time)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_out.size(2))  # k*(h*w)*enc

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
    h = decoder.h(encoder_out.mean(dim=1))
    c = decoder.c(encoder_out.mean(dim=1))

    # while we have not get k captions
    while True:
        # SAME as that in the forward
        # =========================================
        # embedding
        embedding = decoder.embedding(prewords)  # s*1*emb
        embedding = embedding.squeeze(1)  # s*emb
        attention_area, alpha = decoder.attention(encoder_out, h)
        alpha = alpha.view(-1, encoded_size, encoded_size)
        mask = decoder.fc_sig(h)
        softmask = decoder.sigmoid(mask)
        attentioned_out = softmask * attention_area
        xt = torch.cat([embedding, attentioned_out], dim=1)
        h, c = decoder.lstm(xt, (h, c))
        preds = decoder.fc_dic(h)  # s*dic
        # =========================================

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
        topkcap = torch.cat([topkcap[precapinx], nexcapinx.unsqueeze(1)], dim=1)  # s*(step + 1)
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
            #             print("k=0")
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
            #           print(">30")
            break
        step += 1

    # find the max index based on the scores
    # return the best captions and its corresponding alpha by
    # max index
    #     print(len(topkscore_all))
    maxinx = topkscore_all.index(max(topkscore_all))
    bestcap = topkcap_all[maxinx]
    bestalpha = topkalpha_all[maxinx]
    return bestcap, bestalpha


if __name__ == '__main__':
    #     print(word_map['<start>'])
    #     print(word_map['<end>'])
    random.seed(442)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # validate(normalize)
    test_loader = torch.utils.data.DataLoader(CustomDataset("./preprocess_out", "flickr8k", 'TEST',
                                                            transform=transforms.Compose([normalize])),
                                              batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    # Plot captions
    # ============================================================================================================
    f, axarr = plt.subplots(10, 3, figsize=(25, 50))
    f.suptitle('Results: Generated Captions')
    for i, (image, raw_img, caps, caplens, allcaps) in enumerate(test_loader):
        #         if i % 5 != 0:
        #             continue
        if i == num_imgs:
            break
        # already resized and permuted in dataset
        bestcap, bestalpha = search_caption(image.to(device))
        #         print(bestcap)
        decoded_sentence = []
        for encoded_word in bestcap:
            decoded_sentence.append(reversed_word_dict[encoded_word])
        caption = " ".join(decoded_sentence)

        # dimension of image to 1 x W x H x 3
        img_np = raw_img.numpy()[0].transpose(1, 2, 0)

        row = i // 3
        col = i % 3
        axarr[row, col].imshow(img_np)
        axarr[row, col].text(1, -15, caption, verticalalignment='center', fontsize=12, wrap=True,
                             bbox=dict(facecolor='red', alpha=0.4))
        #         axarr[i].imshow(img_np)
        #         axarr[i].text(1, -15, caption, verticalalignment='center', fontsize=14, bbox=dict(facecolor='red', alpha=0.4))

        #         plt.figure()
        #         plt.text(1, -15, caption, verticalalignment='center', fontsize=14, bbox=dict(facecolor='red', alpha=0.4))
        #         plt.imshow(img_np)
        #         # plt.axis('off')
        #         plt.savefig(output_path + "/{}_caption.jpg".format(i))
        imageio.imwrite(os.path.join(output_path, str(i) + "_raw.jpg"), img_np)
        imageio.imwrite(os.path.join(output_path, str(i) + ".jpg"), image.numpy()[0].transpose(1, 2, 0))
    # Write all decoded caps into a text file
    plt.savefig(output_path + "/generated_captions.jpg")
    # ============================================================================================================

    # Plot attention
    # ============================================================================================================
    f.suptitle('Results: Generated attention')
    for i, (image, raw_img, caps, caplens, allcaps) in enumerate(test_loader):
        if i == num_imgs:
            break
        # already resized and permuted in dataset
        bestcap, bestalpha = search_caption(image.to(device))
        
        # to numpy
        bestalpha = torch.FloatTensor(bestalpha)
        image = image[0].numpy().transpose(1, 2, 0)
        
        # get the caption sequence
        decoded_sentence = []
        for encoded_word in bestcap:
            decoded_sentence.append(reversed_word_dict[encoded_word])

        # for every word, plot masked image
        for w in range(len(decoded_sentence)):

            # print(w)
            # print(decoded_sentence[w])
            plt.subplot(np.ceil(len(decoded_sentence) / 5.), 5, w + 1)
            # add text
            plt.text(0, 1, '%s' % (decoded_sentence[w]), color='black',
                     backgroundcolor='white', fontsize=12)
            plt.imshow(raw_img[0].numpy().transpose(1, 2, 0))
            current_alpha = bestalpha[w, :]
            alpha = skimage.transform.resize(
                current_alpha.numpy(), [256, 256])
            plt.imshow(alpha, alpha=0)
            # for the beginning, the mask is none for the word "<begin>"
            if w == 0:
                plt.imshow(alpha, alpha=0)
            else:
                plt.imshow(alpha, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
        plt.savefig(attention_output_path + "/generated_attention_" + str(i) + ".jpg")
        plt.clf()
    # ============================================================================================================
