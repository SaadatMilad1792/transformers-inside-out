# transformers-inside-out
This tutorial is intended to provide a thorough and comprehensive explanation of transformer architectures. It includes detailed discussions of each component and the corresponding code implementations, from the fundamental building blocks to the complete training loop. The tutorial concludes with a practical application of the developed transformer model to an English (en) to Farsi (fa) translation task.

Here is the table of content covered in this tutorial:
|     | Topic                                                                                             | Description                                                         |
|-----|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| 1   | [Introduction](/tutorial_documentation/markdowns/Introduction.md)                                 | Overview of transformer architectures and their significance.       |
| 2   | [Tools](/tutorial_documentation/markdowns/Tools.md)                                               | Required tools and environment setup for the tutorial.              |
| 3   | [Positional Encoding](/tutorial_documentation/markdowns/Positional%20Encoding.md)                 | Explaining the role of positional information in transformers.      |
| 4   | [Tokenization](/tutorial_documentation/markdowns/Tokenization.md)                                 | How input text is processed and converted into tokens.              |
| 5   | [Transformer Encoder](/tutorial_documentation/markdowns/Transformer%20Encoder.md)                 | Detailed explanation and implementation of the encoder block.       |
| 6   | [Transformer Decoder](/tutorial_documentation/markdowns/Transformer%20Decoder.md)                 | Detailed explanation and implementation of the decoder block.       |
| 7   | [Assembling the Transformer](/tutorial_documentation/markdowns/Assembling%20the%20Transformer.md) | Integrating all components to build the complete transformer model. |
| 8   | [Training the Model](/tutorial_documentation/markdowns/Training%20the%20Model.md)                 | Setting up the training loop and optimization process.              |
| 9   | [Inferencing the Model](/tutorial_documentation/markdowns/Inferencing%20the%20Model.md)           | Performing inference and generating predictions from the model.     |


## Acknowledgments

This project leverages the Tatoeba dataset for English–Persian translation. The Tatoeba corpus is a freely available collection of sentences and translations across multiple languages. For access to the dataset, see the [Tatoeba Dataset](https://tatoeba.org/en/downloads).

The development of this project also benefited greatly from several valuable resources:

- **Masking in Transformers**: A clear explanation of padding and look-ahead masks is available in this [Medium article](https://medium.com/@g.martino8/all-the-questions-about-transformer-model-answered-part-5-the-padding-mask-73be0941bc1e).

- **Transformer Training Loops**: For guidance on constructing training loops, loss computation, and optimization strategies, see this [Machine Learning Mastery guide](https://machinelearningmastery.com/training-the-transformer-model/).

- **Inference, Teacher Forcing, and Sequence Generation**: For understanding inference strategies in transformers, including teacher forcing and token-by-token generation, consult this [Stack Overflow discussion](https://stackoverflow.com/questions/57099613/how-is-teacher-forcing-implemented-for-the-transformer-training).

- **Transformer Architecture and Implementation**: In-depth coverage of transformer models, from theory to practical implementation, is available in these resources:
  - [PyTorch Transformer Tutorials](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
  - [DataCamp Transformer Guide](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch)

- **Ajay Halthor's Transformer Tutorials**: The YouTube playlist by Ajay Halthor provides a comprehensive end-to-end explanation of transformer architecture with code examples. 
  - Playlist: [YouTube](https://www.youtube.com/watch?v=QCJQG4DuHT0&list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4)  
  - GitHub: [ajhalthor](https://github.com/ajhalthor)  

We are grateful to all the authors and communities whose openly available resources made this project possible.



## Document Navigation
Navigate through the tutorial using the table of contents, or continue by clicking the link below to proceed to the next section. <br>
[Proceed to the next section: Introduction](/tutorial_documentation/markdowns/Introduction.md) <br>