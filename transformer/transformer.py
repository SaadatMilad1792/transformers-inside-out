########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import torch.nn as nn
from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder

########################################################################################################################
## -- the entire transformer architecture -- ###########################################################################
########################################################################################################################
class Transformer(nn.Module):
  def __init__(self, model_emb, hidden, num_heads, dropout_p, num_layers_enc, 
               num_layers_dec, max_sequence_length, src_stoi, tgt_stoi, 
               SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = 'cpu'):
    
    super(Transformer, self).__init__()
    self.SOS_TOKEN, self.PAD_TOKEN, self.EOS_TOKEN = SOS_TOKEN, PAD_TOKEN, EOS_TOKEN
    self.encoder = TransformerEncoder(model_emb, num_heads, hidden, dropout_p, num_layers_enc, src_stoi,
                                      max_sequence_length, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = device)
    self.decoder = TransformerDecoder(model_emb, num_heads, hidden, dropout_p, num_layers_dec, tgt_stoi,
                                      max_sequence_length, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = device)
    self.vocab_expand = nn.Linear(model_emb, len(tgt_stoi))
    self.softmax = nn.Softmax(dim = -1)

  def forward(self, x, y, enc_sos_token = False, enc_eos_token = False, 
              dec_sos_token = False, dec_eos_token = False, enc_mask = None, 
              dec_mask = None, dec_cross_mask = None):
    
    x = self.encoder(x, enc_sos_token = enc_sos_token, enc_eos_token = enc_eos_token, mask = enc_mask)
    output = self.decoder(x, y, dec_sos_token = dec_sos_token, dec_eos_token = dec_eos_token, 
                          mask = dec_mask, cross_mask = dec_cross_mask)
    
    output = self.vocab_expand(output)
    return output
