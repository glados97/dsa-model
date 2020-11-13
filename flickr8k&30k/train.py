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


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
encoder_dim = 512  # dimension of CNN
decoder_dim = 512  # dimension of LSTMs
emb_dim = 512  # dimension of embeddings
attention_dim = 512  # dimension of attention
dict_size = None
dropout = 0.5

best_bleu_score = 0.
decoder_lr = 4e-4  # learning rate for decoder
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention'
stepsize = 1
gamma = 0.99

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

def main():
    start_epoch = 0
    numepoch = 1

    # Load word map into memory
    word_map_path = "./preprocess_out"
    dataset = "flickr8k"
    word_map = None
    with open(os.path.join("./preprocess_out", 'DICTIONARY_WORDS_' + dataset + '.json'), 'r') as file:
        word_map = json.load(file)
        dict_size = len(word_map)

    # TODO: load train and validation data _ XIE
    # https://pytorch.org/docs/master/torchvision/models.html
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(CustomDataset("./preprocess_out", "flickr8k", 'TRAIN',
                                                             transform=transforms.Compose([normalize])),
                                               batch_size=48, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(CustomDataset("./preprocess_out", "flickr8k", 'VAL',
                                                           transform=transforms.Compose([normalize])),
                                             batch_size=48, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Change Load checkpoint Name
    # check_point_name = "best_checkpoint_flickr8k.pth.tar"
    # encoder, decoder, decoder_opt, last_epoch, best_bleu_score = load_checkpoint(check_point_name)
    # start_epoch = last_epoch + 1

    # move to device if possibble
    encoder = CNN().to(device)
    decoder = LSTMs(encoder_dim=encoder_dim,
                    attention_dim=attention_dim,
                    embed_dim=emb_dim,
                    decoder_dim=decoder_dim,
                    dic_size=dict_size,
                    dropout=dropout).to(device)
    decoder_opt = torch.optim.Adam(
        params=decoder.parameters(), lr=decoder_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        decoder_opt, step_size=stepsize, gamma=gamma)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(start_epoch, numepoch):
        ######################################
        # TODO: check convergence

        # begin train
        ######################################
        print("=========== Epoch: ", epoch, "=============")
        # encoder.train()
        decoder.train()

        scheduler.step()
        # Batches Train
        for i, (img, caption, cap_len) in enumerate(train_loader):
            print("Iteration: ", i)
            # use GPU if possible
            img = img.to(device)
            caption = caption.to(device)
            cap_len = cap_len.to(device)
            decoder_opt.zero_grad()

            # forward
            encoded = encoder(img)
            # print("img", img.shape)
            # print("encoded", encoded.shape)
            preds, sorted_caps, decoded_len, alphas, _ = decoder(encoded, caption, cap_len)

            # ignore the begin word
            trues = sorted_caps[:, 1:]

            # pack and pad
            preds, _ = pack_padded_sequence(preds, decoded_len, batch_first=True)
            trues, _ = pack_padded_sequence(trues, decoded_len, batch_first=True)

            # calculate loss
            loss = criterion(preds, trues)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            loss.backward()

            print("Training Loss: ", loss)
            # update weight
            decoder_opt.step()

            # TODO: print performance

        ######################################
        # end trian
     
        # validate and return score
        val_loss_all = 0
        references = []
        hypotheses = []
        #######################################
        # TODO: check if with torch.no_grad(): necessary
        decoder.eval()
        with torch.no_grad():
            for i, (img, caption, cap_len, all_captions) in enumerate(val_loader):
                # use GPU if possible
                img = img.to(device)
                caption = caption.to(device)
                cap_len = cap_len.to(device)

                # forward
                encoded = encoder(img)
                preds, sorted_caps, decoded_len, alphas, sorted_index = decoder(
                    encoded, caption, cap_len)

                # ignore the begin word
                trues = sorted_caps[:, 1:]
                preds2 = preds.clone()
                # pack and pad
                preds, _ = pack_padded_sequence(preds, decoded_len, batch_first=True)
                trues, _ = pack_padded_sequence(trues, decoded_len, batch_first=True)

                # calculate loss
                loss = criterion(preds, trues)
                loss += alpha_c * (1. - alphas.sum(dim=1) ** 2).mean()
                val_loss_all += loss

                # TODO: print performance
                all_captions = all_captions[sorted_index]
                for j in range(all_captions.shape[0]):
                    img_caps = all_captions[j].tolist()
                    img_captions = list(
                        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                            img_caps))  # remove <start> and pads
                    references.append(img_captions)
                _, predmax = torch.max(preds2, dim=2)
                predmax = predmax.tolist()
                temp_preds = list()
                for j, p in enumerate(predmax):
                    temp_preds.append(
                        predmax[j][:decoded_len[j]])  # remove pads
                predmax = temp_preds
                hypotheses.extend(predmax)
                assert len(references) == len(hypotheses)

            
            with open(('reference/reference%s.json'%(epoch)), 'a') as f:
                json.dump(references, f)

            
            with open(('hypotesis/hypotheses%s.json'%(epoch)), 'a') as f:
                json.dump(hypotheses, f)
            
            
            with open(('results/results%s.json'%(epoch)), 'a') as f:
                json.dump(("epoch: ", epoch), f)
            
            #bleu
            weights = (1.0/1.0, )
            bleu1 = corpus_bleu(references, hypotheses, weights)
            print("bleu1 score: ", bleu1)
            with open(('results/results%s.json'%(epoch)), 'a') as f:
                json.dump(("bleu1 score:", bleu1), f)

            weights=(1.0/2.0, 1.0/2.0,)
            bleu2 = corpus_bleu(references, hypotheses, weights)
            print("bleu2 score: ", bleu2)
            with open(('results/results%s.json'%(epoch)), 'a') as f:
                json.dump(("bleu2 score:", bleu2), f)
                        
            weights=(1.0/3.0, 1.0/3.0, 1.0/3.0,)
            bleu3 = corpus_bleu(references, hypotheses, weights)
            print("bleu3 score: ", bleu3)
            with open(('results/results%s.json'%(epoch)), 'a') as f:
                json.dump(("bleu3 score:", bleu3), f)

            bleu4 = corpus_bleu(references, hypotheses)
            print("bleu4 score: ", bleu4)
            with open(('results/results%s.json'%(epoch)), 'a') as f:
                json.dump(("bleu4 score:", bleu4), f)

            print("Validation Loss All: ", val_loss_all)
            with open(('results/results%s.json'%(epoch)), 'a') as f:
                json.dump(("Lost:", val_loss_all), f)

        # Save Checkpoint
        save_checkpoint(encoder, decoder, decoder_opt, dataset, epoch, 0, True)





if __name__== "__main__":
    main()
