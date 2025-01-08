
## What are Few-shot and Zero-shot generalization capabilities?

Zero-shot learning (ZSL) is the ability of a model to perform a task or recognize a concept _it has never explicitly been trained on_

Few-shot learning (FSL) is the ability of a model to learn a new task or concept from only a _very small number of examples_ (typically 1 to 5).A common approach is meta-learning, also known as "learning to learn." The model is trained on a variety of different tasks, each with a small number of examples. The goal is for the model to learn general strategies for quickly adapting to new tasks, rather than memorizing specific tasks.

## Transformers

Originally introduced in the paper [_Attention Is All You Need_ (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), the Transformer revolutionized sequence-to-sequence tasks such as language translation by relying primarily on **attention mechanisms** without recurrence or convolution. Over time, the architecture has scaled from millions to hundreds of billions of parameters

Key components are encoder-decoder, multi-head self-attention, positional encoding and residual connections and normalization. 

### Attention Mechanism
The attention output is basically a weighted sum of the values VVV, where weights come from how well Q matches each K
$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
