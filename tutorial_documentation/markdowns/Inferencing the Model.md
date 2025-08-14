# Inferencing the Model
Inference is the process of using a trained transformer model to generate predictions or sequences without updating the model weights. Unlike training, where the model sees the full target sequence and computes gradients, inference requires **generating the output token by token**.

### How it works in general

1. **Start with the SOS token**: Sequence generation begins by feeding the special "Start of Sequence" (SOS) token to the decoder.  
2. **Generate token probabilities**: The model predicts the probability distribution over the target vocabulary for the next token.  
3. **Select the next token**: Common strategies include:
   - **Greedy decoding**: select the token with the highest probability.
   - **Sampling**: draw from the probability distribution, possibly applying temperature to control randomness.
   - **Top-k or nucleus sampling**: restrict sampling to the top k most probable tokens to improve quality.
4. **Append the token** to the growing sequence and feed it back into the decoder.
5. **Repeat** until an EOS (End of Sequence) token is generated or the maximum sequence length is reached.

### How it works in this code

In the `translate` method of `TransformerTrainingModule`:

- The method starts with an empty target sentence and the source sentence provided by the user.
- It repeatedly generates one token at a time:
  - Masks are generated for encoder and decoder to prevent attention to padding or future tokens.
  - The model outputs logits for the next token.
  - A **repetition penalty** is applied to avoid generating the same token multiple times.
  - Temperature scaling and optional top-k filtering are applied to adjust randomness.
  - The next token is sampled from the adjusted probability distribution.
- The token is appended to the target sentence.
- The process stops if an EOS, SOS, or PAD token is generated, or if the maximum sequence length is reached.
- Finally, the fully generated target sentence is returned.

This step-by-step token-by-token generation allows the transformer to produce coherent sequences at inference time, even though the model was trained on complete target sequences. You can try loading the weights, and performing inference in [Transformer Training Module](/development/training_module_test.ipynb) notebook.


## Document Navigation
Continue the tutorial by navigating to the previous sections, or return to the table of contents using the links below. <br>
[Return to the previous section: Training the Model](./Training%20the%20Model.md) <br>
[Back to the table of contents](/) <br>