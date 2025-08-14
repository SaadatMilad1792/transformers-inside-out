# Assembling the Transformer
In this section, we will build a single, unified module called the `Transformer`. Since all the individual submodules used in the encoder and decoder have already been introduced, we will not revisit them here—please refer to the **Transformer Encoder** and **Transformer Decoder** sections for detailed explanations.

Our focus will be on combining these components into one cohesive Transformer architecture. This unified model includes some minor modifications to both the encoder and decoder to ensure seamless integration. By the end of this section, you will have the complete code for the entire Transformer model.

Since we are including everything in a single package, we already have all the sub-components, therefore, creating the actual transformer architecture should be a piece of cake:

```python
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
```

You can find an end to end test in [Transformer](/development/transformer_test.ipynb) notebook, where we feed a sentence to the transformer and receive a vector of shape `(max_sequence_length, target_vocab_size)`:

```python
########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import os
import sys
sys.path.append(os.path.abspath(".."))
import torch
import transformer

########################################################################################################################
## -- testing the data handler module -- ###############################################################################
########################################################################################################################
src_vocab_path = "../data/vocabs/en_vocab.json"
tgt_vocab_path = "../data/vocabs/fa_vocab.json"
src_path, src_name = "../data/dataset/Tatoeba.zip", "en.txt"
tgt_path, tgt_name = "../data/dataset/Tatoeba.zip", "fa.txt"
SOS_TOKEN, PAD_TOKEN, EOS_TOKEN = '<SOS>', '<PAD>', '<EOS>'

data_handler = transformer.DataHandler(src_path, src_name, src_vocab_path, tgt_path, tgt_name, tgt_vocab_path, 
                                       SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, max_sequence_length = 256, max_sentences = 1000)

data = data_handler.data()

########################################################################################################################
## -- testing the transformer -- #######################################################################################
########################################################################################################################
batch_size, max_sequence_length, model_emb, hidden = 32, 256, 512, 2048
num_heads, dropout_p, num_layers_enc, num_layers_dec = 8, 0.1, 2, 2

transformer_pipeline = transformer.Transformer(model_emb, hidden, num_heads, dropout_p, num_layers_enc, 
                                               num_layers_dec, max_sequence_length, data.src_stoi, data.tgt_stoi, 
                                               SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = 'cpu')

en_batch, fa_batch = ("Hi",), ("سلام",)
enc_mask, dec_mask, dec_cross_mask = transformer.MaskGenerator(max_sequence_length = max_sequence_length).generate_masks(en_batch, fa_batch)
transformer_pipeline(en_batch, fa_batch, enc_sos_token = False, enc_eof_token = False, 
                     dec_sos_token = False, dec_eos_token = False, enc_mask = enc_mask, 
                     dec_mask = dec_mask, dec_cross_mask = dec_cross_mask).shape
```

And once you run the code, you will see results like this:
```text
torch.Size([1, 256, 85])
```

## Document Navigation
Continue the tutorial by navigating to the previous or next sections, or return to the table of contents using the links below. <br>
[Proceed to the next section: Training the Model](./Training%20the%20Model.md) <br>
[Return to the previous section: Transformer Decoder](./Transformer%20Decoder.md) <br>
[Back to the table of contents](/) <br>