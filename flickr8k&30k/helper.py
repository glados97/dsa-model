import torch
import os

def save_checkpoint(encoder, decoder, decoder_opt, dataset, epoch_num, bleu, is_highest_score):
    state = {
        'encoder': encoder,
        'decoder': decoder,
        # 'encoder_opt': encoder_opt,
        'decoder_opt': decoder_opt,
        'epoch': epoch_num,
        'bleu-score': bleu
    }

    filename = 'checkpoint_' + dataset + '.pth.tar'

    torch.save(state, os.path.join("./checkpoint", filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_highest_score:
        torch.save(state, os.path.join('./checkpoint', 'best_' + filename))


def load_checkpoint(checkpoint_name):
    checkpoint = torch.load(os.path.join("./checkpoint", checkpoint_name))
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    # encoder_opt = checkpoint['encoder_opt']
    decoder_opt = checkpoint['decoder_opt']
    epoch = checkpoint['epoch']
    bleu_score = checkpoint['bleu-score']
    return encoder, decoder, decoder_opt, epoch, bleu_score
