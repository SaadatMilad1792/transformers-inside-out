########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import sys
import json
import torch
import zipfile
import numpy as np
from torch.utils.data import Dataset

########################################################################################################################
## -- data handling, helper functions, and tokenizer -- ################################################################
########################################################################################################################
class SentenceValidation():
  def __init__(self, max_valid_length, src_vocab, tgt_vocab):
    super(SentenceValidation, self).__init__()
    self.max_valid_length = max_valid_length
    self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

  def is_valid_length(self, sentence):
    if len(list(sentence)) < (self.max_valid_length - 1):
      return True
    return False
  
  def is_valid_tokens(self, sentence, vocab):
    for token in list(set(sentence)):
      if token not in vocab:
        return False
    return True

  def valid_indexes(self, src_sentences, tgt_sentences):
    valid_idx_list = []
    for i in range(len(src_sentences)):
      if self.is_valid_length(src_sentences[i]) \
        and self.is_valid_length(tgt_sentences[i]) \
        and self.is_valid_tokens(src_sentences[i], self.src_vocab) \
        and self.is_valid_tokens(tgt_sentences[i], self.tgt_vocab):
        valid_idx_list.append(i)
    return valid_idx_list

class DataHandler():
  def __init__(self, src_path, src_name, src_vocab_path, tgt_path, tgt_name, tgt_vocab_path, 
               SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, max_sequence_length = 256, max_sentences = 10000):
    super(DataHandler, self).__init__()
    self.src_vocab_path = src_vocab_path
    self.tgt_vocab_path = tgt_vocab_path
    self.SOS_TOKEN, self.PAD_TOKEN, self.EOS_TOKEN = SOS_TOKEN, PAD_TOKEN, EOS_TOKEN
    self.src_vocab = self.add_special_token(self.load_vocab(self.src_vocab_path))
    self.tgt_vocab = self.add_special_token(self.load_vocab(self.tgt_vocab_path))
    self.src_path, self.src_name = src_path, src_name
    self.tgt_path, self.tgt_name = tgt_path, tgt_name
    self.max_sequence_length, self.max_sentences = max_sequence_length, max_sentences
    self.validation = SentenceValidation(self.max_sequence_length, self.src_vocab, self.tgt_vocab)

  def load_vocab(self, vocab_path):
    with open(f'{vocab_path}', 'r') as file:
      vocab = list(json.load(file))
    return vocab

  def load_data_from_zip(self, zip_path, file_name, max_sentences = None):
    with zipfile.ZipFile(zip_path, 'r') as z:
      with z.open(file_name) as file:
        text_file = file.read().decode('utf-8')
        text_file = text_file.splitlines()

    text_file = text_file[:max_sentences]
    return [sentence.rstrip('\n').lower() for sentence in text_file]

  def add_special_token(self, vocab):
    return [self.SOS_TOKEN] + vocab + [self.PAD_TOKEN] + [self.EOS_TOKEN]

  def vocab_mapping(self, vocab, mode):
    if mode == 'stoi':
      return {v: k for k, v in enumerate(vocab)}
    elif mode == 'itos':
      return {k: v for k, v in enumerate(vocab)}
    else:
      sys.exit("[!] Invalid 'mode' selected, valid options are ['stoi', 'itos'].")

  def data(self):
    self.src_stoi = self.vocab_mapping(self.src_vocab, mode = 'stoi')
    self.src_itos = self.vocab_mapping(self.src_vocab, mode = 'itos')
    self.tgt_stoi = self.vocab_mapping(self.tgt_vocab, mode = 'stoi')
    self.tgt_itos = self.vocab_mapping(self.tgt_vocab, mode = 'itos')

    self.src_sentences = self.load_data_from_zip(self.src_path, self.src_name, self.max_sentences)
    self.tgt_sentences = self.load_data_from_zip(self.tgt_path, self.tgt_name, self.max_sentences)
    self.valid_indexes = self.validation.valid_indexes(self.src_sentences, self.tgt_sentences)
    self.src_sentences = [self.src_sentences[i] for i in self.valid_indexes]
    self.tgt_sentences = [self.tgt_sentences[i] for i in self.valid_indexes]

    return self

class MaskGenerator():
  def __init__(self, max_sequence_length, NEG_INF = -1e9, device = 'cpu'):
    super(MaskGenerator, self).__init__()
    self.max_sequence_length = max_sequence_length
    self.NEG_INF = NEG_INF
    self.device = device

  def look_ahead_mask(self):
    return torch.triu(torch.ones((self.max_sequence_length, self.max_sequence_length), dtype = torch.bool), diagonal = 1)

  def padding_mask(self, src_batch, tgt_batch, has_eos = True):
    batch_size, shift = len(src_batch), 2 if has_eos else 1
    src_pad_mask = torch.full([batch_size, self.max_sequence_length, self.max_sequence_length], False, dtype = torch.bool)
    tgt_pad_mask = torch.full([batch_size, self.max_sequence_length, self.max_sequence_length], False, dtype = torch.bool)
    tgt_cross_pad_mask = torch.full([batch_size, self.max_sequence_length, self.max_sequence_length], False, dtype = torch.bool)
    
    for i in range(batch_size):
      src_mask = np.arange(len(src_batch[i]), self.max_sequence_length)
      tgt_mask = np.arange(len(tgt_batch[i]) + shift, self.max_sequence_length)
      src_pad_mask[i, :, src_mask], src_pad_mask[i, src_mask, :] = True, True
      tgt_pad_mask[i, :, tgt_mask], tgt_pad_mask[i, tgt_mask, :] = True, True
      tgt_cross_pad_mask[i, :, src_mask] = True
      tgt_cross_pad_mask[i, tgt_mask, :] = True
    return src_pad_mask, tgt_pad_mask, tgt_cross_pad_mask

  def generate_masks(self, src_batch, tgt_batch, has_eos = True):
    look_ahead_mask = self.look_ahead_mask()
    src_pad_mask, tgt_pad_mask, tgt_cross_pad_mask = self.padding_mask(src_batch, tgt_batch, has_eos = has_eos)
    encoder_padding_mask = torch.where(src_pad_mask, self.NEG_INF, 0)
    decoder_padding_mask = torch.where(look_ahead_mask.unsqueeze(0) | tgt_pad_mask, self.NEG_INF, 0)
    decoder_cross_mask = torch.where(tgt_cross_pad_mask, self.NEG_INF, 0)
    return encoder_padding_mask.to(self.device), decoder_padding_mask.to(self.device), decoder_cross_mask.to(self.device)

class TransformerDataset(Dataset):
  def __init__(self, src_sentences, tgt_sentences):
    self.src_sentences = src_sentences
    self.tgt_sentences = tgt_sentences
  
  def __len__(self):
    return len(self.src_sentences)

  def __getitem__(self, idx):
    return self.src_sentences[idx], self.tgt_sentences[idx]