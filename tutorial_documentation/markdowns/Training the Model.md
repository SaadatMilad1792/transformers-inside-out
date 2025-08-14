# Training the Model
In this part we implement a class that can be used to train our mode, please keep in mind that after running the following code, the weights will be stored inside of the weights directory, and you can reuse them as many times as you need, the training class allows you to train, translate, and load weights. You can also find practical examples of them in [Training Module](/development/training_module_test.ipynb) notebook.

```python
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

```


And here is the practical testing of this module:

```python
########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import os
import sys
sys.path.append(os.path.abspath(".."))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import DataHandler, TransformerDataset, Transformer, TransformerTrainingModule

########################################################################################################################
## -- testing data handling, helper functions, and tokenizer -- ########################################################
########################################################################################################################
src_vocab_path = "../data/vocabs/en_vocab.json"
tgt_vocab_path = "../data/vocabs/fa_vocab.json"
src_path, src_name = "../data/dataset/Tatoeba.zip", "en.txt"
tgt_path, tgt_name = "../data/dataset/Tatoeba.zip", "fa.txt"
SOS_TOKEN, PAD_TOKEN, EOS_TOKEN = '<SOS>', '<PAD>', '<EOS>'

batch_size = 64
model_emb = 64
hidden = 256
num_heads = 8
dropout_p = 0.1
num_layers_enc = 2
num_layers_dec = 2
max_sequence_length = 128
max_sentences = 1
device = 'cpu'

## -- creating the dataset and dataloader -- ##
torch.manual_seed(42)
data_handler = DataHandler(src_path, src_name, src_vocab_path, tgt_path, tgt_name, tgt_vocab_path, 
                           SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, max_sequence_length = max_sequence_length, 
                           max_sentences = max_sentences)

data = data_handler.data()
dataset = TransformerDataset(data.src_sentences, data.tgt_sentences)
tr_data = DataLoader(dataset, batch_size = batch_size, shuffle = True)

## -- creating mode, loss function and optimizer -- ##
learning_rate = 2e-5
criterion = nn.CrossEntropyLoss(ignore_index = data.tgt_stoi[PAD_TOKEN], reduction = 'none')
transformer_model = Transformer(model_emb, hidden, num_heads, dropout_p, num_layers_enc, 
                                num_layers_dec, max_sequence_length, data.src_stoi, data.tgt_stoi, 
                                SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = device).to(device)

optimizer = torch.optim.Adam(transformer_model.parameters(), lr = learning_rate)

## -- training loop -- ##
trainer = TransformerTrainingModule(transformer_model, criterion, optimizer, data,
                                    max_sequence_length, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, 
                                    device = device, save_path = '../weights/model_weights.pth')

for _ in range(10):
  trainer.fit(data_loader = tr_data, epochs = 250, verbose = True)
  translation = trainer.translate(data.src_sentences[0], repetition_penalty = 1.2, temperature = 0.7, top_k = 12)
  print()
  print(f"src: ", data.src_sentences[0])
  print(f"tgt: ", data.tgt_sentences[0])
  print(f"trn: ", translation)
  print()
```

Once you run the code (note that there is only one sentence in the batch, you can change it later). You will see results similar to this, try using a bigger dataset or more complex model for better results:
```text
rc:  i just don't know what to say.
tgt:  من فقط نمی دانم چه بگویم.
trn:  م   م ه  ۸ط ننم یرغینومغنم

Epoch 250 / 250, Batch: 1 / 1, Loss: 2.43668
src:  i just don't know what to say.
tgt:  من فقط نمی دانم چه بگویم.
trn:  م طم. ن نممان چگوط ینو.م عمیم م.ظ

Epoch 250 / 250, Batch: 1 / 1, Loss: 1.80733
src:  i just don't know what to say.
tgt:  من فقط نمی دانم چه بگویم.
trn:   فمینمی دچووی ه ع یغ چهم.

Epoch 250 / 250, Batch: 1 / 1, Loss: 1.30678
src:  i just don't know what to say.
tgt:  من فقط نمی دانم چه بگویم.
trn:  من انمگط فذ دانم.

Epoch 250 / 250, Batch: 1 / 1, Loss: 0.93974
src:  i just don't know what to say.
tgt:  من فقط نمی دانم چه بگویم.
trn:  من فق^ دانگویم چه چویم.

Epoch 250 / 250, Batch: 1 / 1, Loss: 0.69524
src:  i just don't know what to say.
tgt:  من فقط نمی دانم چه بگویم.
trn:  من فقط دانم دانم بگوی چم.

Epoch 250 / 250, Batch: 1 / 1, Loss: 0.50033
src:  i just don't know what to say.
tgt:  من فقط نمی دانم چه بگویم.
trn:  من فقط نمی چه می  بگویم.

Epoch 250 / 250, Batch: 1 / 1, Loss: 0.36712
src:  i just don't know what to say.
tgt:  من فقط نمی دانم چه بگویم.
trn:  من فقط دانم نم بگوی چ.

Epoch 250 / 250, Batch: 1 / 1, Loss: 0.33816
src:  i just don't know what to say.
tgt:  من فقط نمی دانم چه بگویم.
trn:  من فقط نمی دانم چه بگویم.

Epoch 250 / 250, Batch: 1 / 1, Loss: 0.23796
src:  i just don't know what to say.
tgt:  من فقط نمی دانم چه بگویم.
trn:  من فقط دانم دانم چه بگویم.
```


## Document Navigation
Continue the tutorial by navigating to the previous or next sections, or return to the table of contents using the links below. <br>
[Proceed to the next section: Inferencing the Model](./Inferencing%20the%20Model.md) <br>
[Return to the previous section: Assembling the Transformer](./Assembling%20the%20Transformer.md) <br>
[Back to the table of contents](/) <br>