# miniLM

A character-level GPT language model built from scratch in PyTorch, trained on *The Wizard of Oz*. Implements the core transformer decoder architecture — multi-head self-attention, feedforward blocks, positional embeddings, and layer normalization.

## Architecture

- **Type**: Decoder-only Transformer (GPT-style)
- **Tokenization**: Character-level
- **Vocabulary Size**: 81 characters
- **Embedding Dim**: 384
- **Layers**: 2 transformer blocks
- **Attention Heads**: 2
- **Context Length**: 32 tokens
- **Dropout**: 0.2

## Getting Started

### Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> For other CUDA versions, visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your setup.

### Dataset

This project was trained on *Dorothy and the Wizard in Oz* by L. Frank Baum, available for free on [Project Gutenberg](https://www.gutenberg.org/ebooks/22566). Download the plain text version and place it in the `data/` folder.

You can also use **any plain text dataset** you like — just update the file path in the notebook and adjust `vocab_size` accordingly.

### Training

Open `gpt-v1.ipynb` and run all cells. The model trains for 3000 iterations using AdamW and saves a checkpoint to `models/model_01.pkl`.

### Inference

```python
context = torch.zeros(size=(1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, 500)[0].tolist())
print(generated_chars)
```

## Training Results

| Iteration | Train Loss | Val Loss |
|-----------|------------|----------|
| 0         | 4.48       | 4.49     |
| 1000      | 1.96       | 2.05     |
| 2000      | 1.82       | 1.92     |
| 3000      | ~1.78      | ~1.83    |

## Sample Output

```text
boo; "but wurt troad.

Doothy, ans wern of nind to seawi n tely eata tear abain,
andly etreple--onath meaned,e, so in thly,eee nive goable st dame...
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
