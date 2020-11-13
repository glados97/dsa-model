from lstms import *
import torch

if __name__ == "__main__":
    input = torch.randn((5, 32, 32, 64))

    # Model parameters
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5
    N = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    
    encoder_out = torch.rand((N, 32, 32, 512))
    encoder_cap = torch.randint(0,5,(N, 10))
    cap_len = torch.randint(0, 10,(N,1))
    decoder = LSTMs(encoder_dim=512, 
                attention_dim=attention_dim,
                embed_dim=emb_dim,
                decoder_dim=decoder_dim,
                dic_size=100,
                dropout=dropout)

    out1, out2 = decoder(encoder_out, encoder_cap, cap_len)
    print(out1)
    # x = torch.tensor([3])
    # embembedding = nn.Embedding(5, 3)
    # print(embembedding(encoder_cap))