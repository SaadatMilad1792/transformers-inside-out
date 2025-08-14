# Tokenization
Transformers, like any other neural network or, more generally, any conventional computer system, understand only numbers. This means that a sentence has no inherent meaning to a computer.

In this part of the tutorial, we will explore **tokenization**, a technique that bridges the gap between how humans communicate and how machines process data. Think of tokenization as a pre-written cheat sheet that maps elements from the human language into a form the computer can understand (and also the other way around as well).

|     | Topic                                                                 | Description                                                                           |
|-----|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| 1   | [Helper Functions](#helper-functions)                                 | Utility functions used throughout the project for various tasks                       |
| 2   | [Dataset](#dataset)                                                   | Implementation of the Torch Dataset class to handle data loading and preprocessing    |
| 3   | [DataLoader](#dataloader)                                             | Torch DataLoader module for batching and shuffling the dataset                        |
| 4   | [Tokenization](#tokenization-1)                                       | Converting raw text into tokens compatible with model input                           |
| 5   | [Padding Mask and Look Ahead Mask](#padding-mask-and-look-ahead-mask) | Creating masks to manage padding tokens and control attention flow in the Transformer |
| 6   | [Vocab Embedding](#vocab-embedding)                                   | Mapping tokens to dense vector representations for model input                        |

## Helper Functions
We will begin by examining helper functions used to load and preprocess the data. Then, we will tokenize the data so it becomes suitable for machine consumption.

The provided helper functions and classes handle loading data from the zip files, adding special tokens such as `START_TOKEN`, `PADDING_TOKEN`, and `END_TOKEN`, and validating the data to ensure that the sequence length is appropriate and all tokens exist in our vocabulary. You can find the dataset and vocabulary files in the `datasets` directory of this repository. A practical example of the following code can be found here [Data Handler](/development/data_handler_test.ipynb), and the code is provided below as well:

```python
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
```

The following code will generate the `data` which is a class with pretty much all the properties we are going to need in this project, including vocabularies, stoi and itos for source and target language mapping, and etc. 

```python
########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import os
import sys
sys.path.append(os.path.abspath(".."))
import transformer
from torch.utils.data import DataLoader

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

print(f"The DataHandler module preprocesses the data, and provides us with all properties we require: ")
print(f"Total Sentence Count After Validation: {len(data.src_sentences)}")
print(f"English (EN) sentence from dataset: \n{data.src_sentences[0]}")
print(f"Persian (FA) sentence from dataset: \n{data.tgt_sentences[0]}")
```

After running the code provided above, you should see output similar to the example below. Unless the validation mechanism is changed, the results are expected to remain consistent. For simplicity, we limit the maximum sentence length to 256 characters. In practice, the maximum sequence length is often chosen based on the statistical distribution of sentence lengths in the dataset. A common approach is to set it around two to three times the standard deviation, which typically covers between 95% and 99.7% of the data within that range.

```text
The DataHandler module preprocesses the data, and provides us with all properties we require: 
Total Sentence Count After Validation: 948
English (EN) sentence from dataset: 
i don't speak japanese.
Persian (FA) sentence from dataset: 
من ژاپنی صحبت نمی‌کنم.
```

## Dataset 
Pytorch allows us to modify the Dataset class and overwrite some of the components of this class as we see fit, in this example we will do the minimum required changes to the Dataset module to make it compatible with the task we have at hand, which includes the constructor, the __len__ and lastly the __getitem__ functions that exist within Dataset:

```python
class TransformerDataset(Dataset):
  def __init__(self, src_sentences, tgt_sentences):
    self.src_sentences = src_sentences
    self.tgt_sentences = tgt_sentences
  
  def __len__(self):
    return len(self.src_sentences)

  def __getitem__(self, idx):
    return self.src_sentences[idx], self.tgt_sentences[idx]
```

## Dataloader 
Before moving forward, it is important to introduce the concept of a DataLoader. Mathematically, the ideal loss function minimization occurs when the model sees the entire dataset in every iteration of optimization. However, this is practically impossible due to the sheer size of most datasets.

A common solution is to split the large dataset into smaller batches. For example, consider a dataset with 100,000 sentences, each with a maximum length of 256 tokens and an embedding size of 512. The dataset can be represented as `100,000 * 256 * 512` floating-point numbers. Assuming each floating-point number is stored as a 32-bit float (4 bytes), the total memory required is `100,000 * 256 * 512 * 4 bytes ≈ 50 GB`.

Feeding this entire dataset into the model at once is impossible for most hardware. Instead, if we split the data into 1,000 batches, each batch contains 100 sentences. Each sentence has 256 tokens, and each token is represented by 512 floating-point numbers. This reduces the size per batch to `100 * 256 * 512 * 4 bytes ≈ 50 MB`, which is much more manageable for the model.

The trade-off is that the training loop needs to process all batches sequentially, which may slow down training slightly, but it effectively solves the memory constraint problem. In this part of the test, we can take a look at the components of our dataset, which we have already introduced, each element in the dataLoader, is a batch of pairs from the dataset, which we will use later to train the transformer model:
```python
########################################################################################################################
## -- testing the transformer dataset module -- ########################################################################
########################################################################################################################
batch_size = 32
dataset = transformer.TransformerDataset(data.src_sentences, data.tgt_sentences)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

for batch_idx, (en_batch, fa_batch) in enumerate(dataloader):
  print(f"In batch {batch_idx}, there are {batch_size} English and Persian sentences")
  print(f"The first English sentence in this batch is: \n{en_batch[0]}")
  print(f"The first Persian sentence in this batch is: \n{fa_batch[0]}")
  break
```

And the output of the code will be the following:
```text
In batch 0, there are 32 English and Persian sentences
The first English sentence in this batch is: 
i just don't know what to say.
The first Persian sentence in this batch is: 
من فقط نمی دانم چه بگویم.
```

## Tokenization
Finally, we can focus on the **tokenizer**. Its job is to map every character in a sentence to its corresponding index. In addition to this basic mapping, it also:

- Injects an **`sos_token`** (*start-of-sentence token*) at the beginning.  
- Adds an **`eos_token`** (*end-of-sentence token*) at the end.  
- Pads the sequence with **`pad_token`** values after the sentence ends, ensuring a uniform length.  

The result is a **tokenized vector of integers** that can be fed into our model. Below is the implementation, along with an example of how the output looks:
```python
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
```

The outputs of the given code are shown below. For visualization purposes, we temporarily set the **`max_sequence_length`** to `36` so the entire vector can be displayed. We can observe **two vectors**:  

- Each starts with `0`, representing the **`sos_token`** (*start-of-sentence token*).  
- At some point, they peak to `97` and `84` respectively, which are followed by `96` and `83` — the **`eos_token`** values (*end-of-sentence token*).  
- The rest of the sequence is filled with **padding tokens**, ensuring the total length is exactly `36`.
- You can find a complete example of this in [Tokenization](/development/token_embeddings_test.ipynb) notepad.

```python
########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import os
import sys
sys.path.append(os.path.abspath(".."))
import transformer
from torch.utils.data import DataLoader

########################################################################################################################
## -- testing the tokenizer module -- ##################################################################################
########################################################################################################################
src_vocab_path = "../data/vocabs/en_vocab.json"
tgt_vocab_path = "../data/vocabs/fa_vocab.json"
src_path, src_name = "../data/dataset/Tatoeba.zip", "en.txt"
tgt_path, tgt_name = "../data/dataset/Tatoeba.zip", "fa.txt"
SOS_TOKEN, PAD_TOKEN, EOS_TOKEN = '<SOS>', '<PAD>', '<EOS>'

data_handler = transformer.DataHandler(src_path, src_name, src_vocab_path, tgt_path, tgt_name, tgt_vocab_path, 
                                       SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, max_sequence_length = 36, max_sentences = 1000)

data = data_handler.data()

max_sequence_length = 36
en_tokenizer = transformer.Tokenizer(data.src_stoi, max_sequence_length = max_sequence_length,
                                     SOS_TOKEN = SOS_TOKEN, PAD_TOKEN = PAD_TOKEN, EOS_TOKEN = EOS_TOKEN)
fa_tokenizer = transformer.Tokenizer(data.tgt_stoi, max_sequence_length = max_sequence_length, 
                                     SOS_TOKEN = SOS_TOKEN, PAD_TOKEN = PAD_TOKEN, EOS_TOKEN = EOS_TOKEN)

batch_size = 4
dataset = transformer.TransformerDataset(data.src_sentences, data.tgt_sentences)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

for batch_idx, (en_batch, fa_batch) in enumerate(dataloader):
  print(f"In batch {batch_idx}, there are {batch_size} English and Persian sentences")
  print(f"The first English sentence in this batch is: \n{en_batch[0]}")
  print(f"The first Persian sentence in this batch is: \n{fa_batch[0]}")

  en_batch_tokenized = en_tokenizer.batch_tokenize(en_batch, sos_token = True, eos_token = True)
  fa_batch_tokenized = fa_tokenizer.batch_tokenize(fa_batch, sos_token = True, eos_token = True)
  print()
  print(f"In batch {batch_idx}, there are {batch_size} English and Persian sentences")
  print(f"The first tokenized English sentence in this batch is: \n{en_batch_tokenized.shape}, {en_batch_tokenized[0, :36]}")
  print(f"The first tokenized Persian sentence in this batch is: \n{fa_batch_tokenized.shape}, {fa_batch_tokenized[0, :36]}")
  break
```
```text
In batch 0, there are 4 English and Persian sentences
The first English sentence in this batch is: 
i just don't know what to say.
The first Persian sentence in this batch is: 
من فقط نمی دانم چه بگویم.

In batch 0, there are 4 English and Persian sentences
The first tokenized English sentence in this batch is: 
torch.Size([4, 36]), tensor([ 0,  9, 95, 10, 21, 19, 20, 95,  4, 15, 14, 87, 20, 95, 11, 14, 15, 23,
        95, 23,  8,  1, 20, 95, 20, 15, 95, 19,  1, 25, 91, 97, 96, 96, 96, 96])
The first tokenized Persian sentence in this batch is: 
torch.Size([4, 36]), tensor([ 0, 28, 29, 82, 23, 24, 19, 82, 29, 28, 32, 82, 10,  1, 29, 28, 82,  7,
        31, 82,  2, 26, 30, 32, 28, 78, 84, 83, 83, 83, 83, 83, 83, 83, 83, 83])
```

## Padding Mask and Look Ahead Mask
We solved the problem of varying sentence lengths by padding them, but this introduced a new challenge: the model might start paying attention to the padding instead of the actual sentence. Fortunately, there’s a solution. Recall that in the self-attention mechanism, we end up with a square matrix of size `max_sequence_length × max_sequence_length`.

The idea is to keep only the top-left square of meaningful data (corresponding to the real sentence) and fill the rest with a very large negative value. When this padded matrix is passed through a softmax, attention naturally focuses on the top-left portion where the actual sentence resides, while the rest is effectively ignored.

There are two important things to note here. First, avoid using `float(-inf)` because it can cause numerical instability and potentially break the model by producing NaN values. Second, when applying softmax, the padded rows will still represent a probability distribution, but since all their values are equal, the model gains no useful information from them and is less likely to focus on the padding.

Another useful masking technique is the look-ahead mask, which we already discussed in the decoder section. This is used in the decoder’s self-attention to prevent it from “cheating” by looking at future tokens. Since we’ll need both types of masks often, we can write a class that generates them on demand.

English and Persian sentences, are not always the same length, that's why we are going to need another mask for cross attention as well, which is very similar to the padding mask in the encoder self attention, but since Key and Value are coming from encoder and Query from the decoder, we expect to see rows after the target sentence and columns after source sentence set to `-inf`. You might also notice that we set two additional rows to `-inf` for decoder as well, that that is for `<SOS_TOKEN>`, and `<EOS_TOKEN>`. You can find an example of this in [Padding Mask and Look-Ahead Mask](/development/data_handler_test.ipynb).

```python
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
```

Here is how we test the masks:
```python
########################################################################################################################
## -- testing the mask generator module -- #############################################################################
########################################################################################################################
mask_gen = transformer.MaskGenerator(max_sequence_length = 7)
en_test_batch = ("Hi",)
fa_test_batch = ("سلام",)
torch.set_printoptions(precision = 1)
enc_mask, dec_mask, dec_cross_mask = mask_gen.generate_masks(en_test_batch, fa_test_batch)
print(f"Encoder Padding Mask (shape: {enc_mask.shape}):")
print(enc_mask, end = "\n\n")
print(f"Decoder Padding + Look-Ahead Mask (shape: {dec_mask.shape}):")
print(dec_mask, end = "\n\n")
print(f"Decoder Cross Attention Mask (shape: {dec_cross_mask.shape}):")
print(dec_cross_mask)
```

And you will see the following results:
```text
Encoder Padding Mask (shape: torch.Size([1, 7, 7])):
tensor([[[ 0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [-1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [-1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [-1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [-1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [-1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09]]])

Decoder Padding + Look-Ahead Mask (shape: torch.Size([1, 7, 7])):
tensor([[[ 0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00,  0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00,  0.0e+00,  0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00,  0.0e+00,  0.0e+00,  0.0e+00,  0.0e+00, -1.0e+09],
         [-1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09]]])

Decoder Cross Attention Mask (shape: torch.Size([1, 7, 7])):
tensor([[[ 0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [ 0.0e+00,  0.0e+00, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09],
         [-1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09, -1.0e+09]]])
```

## Vocab Embedding
Now that we have learned how to tokenize the input and generate the necessary masks, there is still one crucial step left.  

If you recall, we decided to represent our source and target sentences as two matrices of shape `max_sequence_length × model_emb`. However, after tokenization, we only have a flat vector of integer token IDs. These integers simply index into our vocabulary — they are not yet in a form that captures any semantic meaning.  

To bridge this gap, we need to map each token ID to a dense, learnable vector representation. This is done using the `nn.Embedding` layer, which acts like a lookup table whose entries are learned during training. Each token in the vocabulary is assigned an embedding vector of size `model_emb`, and these embeddings are updated so that tokens with similar roles or meanings in the data end up with similar vector representations.  

By applying this embedding step to our tokenized inputs, we produce two matrices that are now fully compatible with our Transformer Encoder and Transformer Decoder.  

Congratulations — at this point, we have completed all the steps necessary to transform human-readable sentences into a machine-friendly representation that can be processed by our model! Here is the last missing part of the pipeline which is implemented below:

```python
########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import torch
from .positional_encoding import *

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
```

As we can see, this part basically maps every token to it's corresponding embedding, additionally, these are learnable parameters, and will be updated over time, the following shows the shape of the vectors after the embedding layer, you may find [TokenEmbedding](/development/token_embeddings_test.ipynb) notebook usefull to test this part:

```python
########################################################################################################################
## -- testing the token embedding module -- ############################################################################
########################################################################################################################
max_sequence_length, model_emb, stoi, dropout_p = 256, 512, data.src_stoi, 0.1
token_embedding = transformer.TokenEmbedding(max_sequence_length, model_emb, stoi, dropout_p,
                                             SOS_TOKEN = SOS_TOKEN, PAD_TOKEN = PAD_TOKEN, 
                                             EOS_TOKEN = EOS_TOKEN, device = 'cpu')
inp = ("Hello", "Bye Bye", "Wait for me too", )
out = token_embedding(inp, sos_token = True, eos_token = True)
out.shape
```

And the output will be:
```text
torch.Size([3, 256, 512])
```

## Document Navigation
Continue the tutorial by navigating to the previous or next sections, or return to the table of contents using the links below. <br>
[Proceed to the next section: Transformer Encoder](./Transformer%20Encoder.md) <br>
[Return to the previous section: Positional Encoding](./Positional%20Encoding.md) <br>
[Back to the table of contents](/) <br>