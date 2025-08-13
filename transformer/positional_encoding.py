########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import torch
import torch.nn as nn

########################################################################################################################
## -- positional encoding -- ###########################################################################################
########################################################################################################################
class PositionalEncoding(nn.Module):
  def __init__(self, max_sequence_length, model_emb, device = 'cpu'):
    super(PositionalEncoding, self).__init__()
    self.max_sequence_length = max_sequence_length
    self.model_emb = model_emb
    self.device = device

  def forward(self):
    position = torch.arange(0, self.max_sequence_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.model_emb, 2).float() * (-torch.log(torch.tensor(1e4)) / self.model_emb))

    PE = torch.zeros(self.max_sequence_length, self.model_emb)
    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    return PE.to(self.device)