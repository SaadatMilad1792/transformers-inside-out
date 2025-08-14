########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import torch
import torch.nn as nn
from .data_handler import MaskGenerator

########################################################################################################################
## -- transformer training module -- ###################################################################################
########################################################################################################################
class TransformerTrainingModule():
  def __init__(self, model, criterion, optimizer, data_object, max_sequence_length, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, 
    device = 'cpu', save_path = './best_model.pth'):
    
    super(TransformerTrainingModule, self).__init__()
    self.model = model
    self.device = device
    self.criterion = criterion
    self.optimizer = optimizer
    self.data_object = data_object
    self.max_sequence_length = max_sequence_length
    self.mask_gen = MaskGenerator(max_sequence_length = self.max_sequence_length)
    self.SOS_TOKEN, self.PAD_TOKEN, self.EOS_TOKEN = SOS_TOKEN, PAD_TOKEN, EOS_TOKEN
    self.best_loss = float('inf')
    self.save_path = save_path

  def fit(self, data_loader, epochs = 1, verbose = True):

    for epoch in range(epochs):
      epoch_loss = 0
      num_batches = 0
      for batch_idx, batch in enumerate(iter(data_loader)):
        self.model.train()
        src_batch, tgt_batch = batch
        enc_mask, dec_mask, dec_cross_mask = self.mask_gen.generate_masks(src_batch, tgt_batch, has_eos = True)
        self.optimizer.zero_grad()
        output = self.model(src_batch, tgt_batch, enc_mask = enc_mask.to(self.device), 
          dec_mask = dec_mask.to(self.device), dec_cross_mask = dec_cross_mask.to(self.device),
          enc_sos_token = False, enc_eos_token = False, dec_sos_token = True, dec_eos_token = True)
        
        labels = self.model.decoder.embedding.tokenizer.batch_tokenize(tgt_batch, sos_token = False, eos_token = True)
        loss = self.criterion(output.view(-1, len(self.data_object.tgt_vocab)).to(self.device), 
          labels.view(-1).to(self.device)).to(self.device)
        
        non_pad_idx = torch.where(labels.view(-1) == self.data_object.tgt_stoi[self.PAD_TOKEN], False, True)
        loss = loss.sum() / non_pad_idx.sum()
        loss.backward()
        self.optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        print(f"Epoch {epoch + 1} / {epochs}, Batch: {batch_idx + 1} / {len(data_loader)}, " +
          f"Loss: {loss:.5f}", end = 50 * "" + "\r") if verbose else None

      avg_epoch_loss = epoch_loss / num_batches
      if avg_epoch_loss < self.best_loss:
        self.best_loss = avg_epoch_loss
        torch.save(self.model.state_dict(), self.save_path)

  def translate(self, sentence, repetition_penalty = 1.2, temperature = 1.0, top_k = 0):
    self.model.eval()
    with torch.no_grad():
      src_sentence, tgt_sentence = (sentence,), ("",)
      generated_ids = []

      for _ in range(self.max_sequence_length):
        enc_mask, dec_mask, dec_cross_mask = self.mask_gen.generate_masks(src_sentence, tgt_sentence, has_eos = False)
        logits = self.model(src_sentence, tgt_sentence, enc_mask = enc_mask.to(self.device), 
                            dec_mask = dec_mask.to(self.device), dec_cross_mask = dec_cross_mask.to(self.device),
                            enc_sos_token = False, enc_eos_token = False, dec_sos_token = True, 
                            dec_eos_token = False)[0][len(generated_ids)]

        for tid in set(generated_ids):
          logits[tid] = logits[tid] / repetition_penalty if logits[tid] > 0 else logits[tid] * repetition_penalty
        logits = logits / temperature

        if top_k > 0:
          v, i = torch.topk(logits, top_k)
          mask = torch.full_like(logits, float('-inf'))
          mask.scatter_(0, i, v)
          logits = mask

        exp_logits = torch.exp(logits - torch.max(logits))
        probs = exp_logits / exp_logits.sum()
        next_id = torch.multinomial(probs, 1).item()
        token = self.data_object.tgt_itos[next_id]
        if token in [self.SOS_TOKEN, self.PAD_TOKEN, self.EOS_TOKEN]:
          break
        generated_ids.append(next_id)
        tgt_sentence = (tgt_sentence[0] + token,)

      return tgt_sentence[0]

  def load_weights(self, path = None):
    path = path or self.save_path
    self.model.load_state_dict(torch.load(path, map_location = self.device))
    if hasattr(self.model, 'to'):
      self.model.to(self.device)
