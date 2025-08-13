# Positional Encoding
Unlike recurrent or convolutional architectures, the Transformer has no inherent sense of word order in a sequence. To give the model information about the position of each token, we use positional encoding which is a method of injecting position-dependent signals into the token embeddings.

The most common approach, introduced in the original Transformer paper, is to add a set of deterministic sine and cosine functions to each embedding vector. This allows the model to learn relative positions and distances between tokens without relying on recurrence.

The positional encoding for a position $pos$ and dimension $i$ is defined as:

$$
PE_{(pos, 2i)} = \sin \left( \frac{pos}{10000^{\frac{2i}{d_{model}}}} \right)
$$

$$
PE_{(pos, 2i+1)} = \cos \left( \frac{pos}{10000^{\frac{2i}{d_{model}}}} \right)
$$

Where:
- $pos$ is the position index (starting at 0 for the first token)
- $i$ is the dimension index within the embedding vector
- $d_{model}$ is the model’s embedding size

Even indices ($2i$) use the sine function, and odd indices ($2i+1$) use the cosine function. The denominator $10000^{\frac{2i}{d_{model}}}$ ensures that each dimension corresponds to a sinusoid of different wavelength.

Finally, the positional encoding is added directly to the token embedding before being fed into the encoder or decoder:

$$
\text{InputEmbeddingWithPE} = \text{TokenEmbedding} + \text{PositionalEncoding}
$$

This addition lets the model combine semantic meaning from the token embeddings with positional information from the encoding, enabling it to capture order-dependent patterns. The following code creates the positional embedding for a given `max_sequence_length` and `model_embedding`:

```python
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
```

In this tutorial, we have included a directory named `development` that contains practical examples for almost every module and component discussed. For instance, see the [Positional Encoding Example](/development/positional_encoding_test.ipynb) to explore how it works in practice. For convenience, all modules have been bundled into a package named `transformer`. In all examples, you can simply import `transformer` and use the desired component without extra setup. Once you run the following code in the specified example, you should observe results similar to those shown:

```python
########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import os
import sys
sys.path.append(os.path.abspath(".."))
import transformer

########################################################################################################################
## -- testing positional encoding -- ###################################################################################
########################################################################################################################
max_sequence_length, model_emb = 8, 6
positional_encoding = transformer.PositionalEncoding(max_sequence_length, model_emb)
positional_encoding_matrix = positional_encoding()
print(f"Positional encoding matrix for [max_sequence_length = {max_sequence_length}, model_emb = {model_emb}]" +
      f"\n{'-' * 80}\n{positional_encoding_matrix}")
```

The output of the provided code, should be something similar to the following:

```text
Positional encoding matrix for [max_sequence_length = 8, model_emb = 6]
--------------------------------------------------------------------------------
tensor([[ 0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000],
        [ 0.8415,  0.5403,  0.0464,  0.9989,  0.0022,  1.0000],
        [ 0.9093, -0.4161,  0.0927,  0.9957,  0.0043,  1.0000],
        [ 0.1411, -0.9900,  0.1388,  0.9903,  0.0065,  1.0000],
        [-0.7568, -0.6536,  0.1846,  0.9828,  0.0086,  1.0000],
        [-0.9589,  0.2837,  0.2300,  0.9732,  0.0108,  0.9999],
        [-0.2794,  0.9602,  0.2749,  0.9615,  0.0129,  0.9999],
        [ 0.6570,  0.7539,  0.3192,  0.9477,  0.0151,  0.9999]])
```

## Document Navigation
Continue the tutorial by navigating to the previous or next sections, or return to the table of contents using the links below. <br>
[Proceed to the next section: Tokenization](./Tokenization.md) <br>
[Return to the previous section: Tools](./Tools.md) <br>
[Back to the table of contents](/) <br>