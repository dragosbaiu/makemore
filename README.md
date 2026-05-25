# Makemore from Scratch
A character-level language model built from scratch following Andrej Karpathy's makemore series.

## What's included
### Part 1
- Bigram model — character-level language model using bigram statistics
- `torch.Tensor` — efficient neural network evaluation using PyTorch tensors
- Language modeling framework — model training, sampling, and loss evaluation (negative log likelihood)

### Part 2
- MLP language model — multi-layer perceptron following Bengio et al. (2003)
- Character embeddings — learned 10-dimensional embedding table mapping each of the 27 characters to a dense vector
- Context window — uses the previous 3 characters to predict the next one
- Architecture: embedding (27×10) → hidden layer (300 units, tanh) → output (27 logits), ~17.7k parameters
- Dataset split — 80% training, 10% dev, 10% testing
- Trained over 50k iterations with batch size 32, reaching ~2.18 train loss and ~2.22 dev loss

### Part 3
- Forward pass activations and backward pass gradients — deep dive into MLP internals across multiple layers, inspecting statistics and identifying pitfalls from improper scaling
- Diagnostic tools — visualizations for activation distributions, gradient distributions, weight gradient histograms, and update-to-data ratio plots to assess the health of a deep network
- Kaiming initialization — proper weight scaling to avoid vanishing/exploding activations at init ([paper](https://arxiv.org/abs/1502.01852))
- Batch Normalization — first modern technique that made training deep nets much easier; implemented from scratch and as a reusable `BatchNorm1d` class with running statistics for inference ([paper](https://arxiv.org/abs/1502.03167))
- Module system — reusable `Linear`, `BatchNorm1d`, and `Tanh` classes enabling clean composition of deep architectures
- Notable todos for later: residual connections and the Adam optimizer

### Part 4
- Manual backpropagation — 4 exercises implementing the backward pass by hand:
  1. Backprop through every intermediate variable of the forward pass one by one (cross-entropy, softmax, batch norm, tanh, linear layers)
  2. Fused backward pass through the full cross-entropy loss in a single closed-form expression
  3. Fused backward pass through batch normalization in a single closed-form expression
  4. Full training loop using only the manual backward pass (no `loss.backward()`)

## Usage
Open the desired notebook in Jupyter and run the cells in order.

## Dependencies
- torch
- matplotlib

## Credits
Based on [Andrej Karpathy's Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) series.
