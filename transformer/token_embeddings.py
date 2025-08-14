########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import torch
from .positional_encoding import *

########################################################################################################################
## -- tokenizer -- #####################################################################################################
########################################################################################################################
class Tokenizer():
  def __init__(self, stoi, max_sequence_length, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = 'cpu'):
    super(Tokenizer, self).__init__()
    self.stoi = stoi
    self.device = device
    self.max_sequence_length = max_sequence_length
    self.SOS_TOKEN, self.PAD_TOKEN, self.EOS_TOKEN = SOS_TOKEN, PAD_TOKEN, EOS_TOKEN

  def add_sos_token(self):
    self.tokenized.insert(0, self.stoi[self.SOS_TOKEN])

  def add_eos_token(self):
    self.tokenized.append(self.stoi[self.EOS_TOKEN])
  
  def pad_sentence(self):
    for _ in range(len(self.tokenized), self.max_sequence_length):
      self.tokenized.append(self.stoi[self.PAD_TOKEN])
  
  def tokenize(self, sentence, sos_token = True, eos_token = True):
    self.tokenized = [self.stoi[token] for token in list(sentence)]
    self.add_sos_token() if sos_token else None
    self.add_eos_token() if eos_token else None
    self.pad_sentence()
    return torch.tensor(self.tokenized)
  
  def batch_tokenize(self, sentences, sos_token = True, eos_token = True):
    batch_tokenized = []
    for sentence in sentences:
      tokenized = self.tokenize(sentence, sos_token = sos_token, eos_token = eos_token)
      batch_tokenized.append(tokenized)
    return torch.stack(batch_tokenized).to(self.device)
  
########################################################################################################################
## -- token embedding mapping -- #######################################################################################
########################################################################################################################
class TokenEmbedding(nn.Module):
  def __init__(self, max_sequence_length, model_emb, stoi, dropout_p, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = 'cpu'):
    super(TokenEmbedding, self).__init__()
    self.max_sequence_length = max_sequence_length
    self.model_emb = model_emb
    self.device = device
    self.stoi, self.vocab_len = stoi, len(stoi)
    self.SOS_TOKEN, self.PAD_TOKEN, self.EOS_TOKEN = SOS_TOKEN, PAD_TOKEN, EOS_TOKEN
    self.tokenizer = Tokenizer(self.stoi, self.max_sequence_length, self.SOS_TOKEN, self.PAD_TOKEN, self.EOS_TOKEN, self.device)
    self.embedding = nn.Embedding(self.vocab_len, self.model_emb)
    self.pe = PositionalEncoding(self.max_sequence_length, self.model_emb)
    self.dropout = nn.Dropout(p = dropout_p)

  def forward(self, x, sos_token = True, eos_token = True):
    x = self.tokenizer.batch_tokenize(x, sos_token, eos_token)
    x = self.embedding(x)
    p = self.pe().to(self.device)
    x = self.dropout(x + p)
    return x