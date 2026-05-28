# Transformer Architecture Explanation

## 1. First: What did your Bigram Model do?

Your bigram model learned:

> “Given the current character, what is the next character?”

Example:

```txt
"h" -> maybe "e"
"e" -> maybe "l"
"l" -> maybe "l"
```

So prediction depends on only **1 previous token**.

Mathematically:

P(x_t \mid x_{t-1})

That’s why it’s called **bi-gram**.

---

## Problem with Bigram

Suppose sentence is:

```txt
"The animal didn't cross the street because IT was too tired"
```

What is “it”?

Bigram only sees previous token:

```txt
"was" -> next?
```

It forgets long context.

So transformers solve this using **Attention**.

---

## 2. What is Attention? (Main idea)

Attention means:

> “While predicting next word, look at ALL previous words and decide which ones are important.”

Example sentence:

```txt
I love eating pizza because pizza tastes amazing
```

When predicting `"tastes"`:

Model strongly focuses on `"pizza"`.

So instead of only previous word:

```txt
amazing -> ?
```

it can look at:

```txt
I
love
eating
pizza
because
pizza
```

This is the breakthrough.

---

## 3. Understanding the lower-left image (Scaled Dot Product Attention)

This is the core formula of attention.

You see:

* Q = Query
* K = Key
* V = Value

Think of it like:

| Thing | Meaning                            |
| ----- | ---------------------------------- |
| Query | What current word is searching for |
| Key   | What each word offers              |
| Value | Actual information of word         |

---

## Example with Bigram-style thinking

Sentence:

```txt
"The cat sat on the mat"
```

Suppose model is predicting after:

```txt
"The cat sat on the"
```

Current word = `"the"`

Query asks:

> “Which previous words matter?”

Keys from all previous words are compared.

Maybe scores become:

| Word | Attention Score |
| ---- | --------------- |
| The  | 0.05            |
| cat  | 0.10            |
| sat  | 0.15            |
| on   | 0.20            |
| the  | 0.10            |
| mat  | 0.40            |

So model focuses mostly on `"mat"`.

---

## 4. Dot Product = Similarity

The line in diagram:

```txt
Q × Kᵀ
```

means:

> Compare current word with every other word.

Large value = very relevant.

Formula:

\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

Simple meaning:

1. Compare words
2. Get importance scores
3. Softmax converts to probabilities
4. Use probabilities to mix information

---

## 5. Why "Scaled"?

You see divide by:

```txt
√dk
```

Without scaling, values become huge and softmax becomes unstable.

So scaling keeps training stable.

---

## 6. What is Multi-Head Attention? (lower-right image)

Instead of ONE attention:

Transformer uses MANY attentions in parallel.

Each head learns different relationships.

Example:

Sentence:

```txt
"The boy kicked the ball because he was angry"
```

Different heads may learn:

| Head   | Focus                  |
| ------ | ---------------------- |
| Head 1 | Grammar                |
| Head 2 | Subject                |
| Head 3 | Emotion                |
| Head 4 | Long-distance relation |

Then all results are combined.

That’s why diagram shows many Q,K,V blocks.

---

## 7. Now the BIG transformer diagram

Left side = Encoder
Right side = Decoder

GPT mainly uses decoder-style architecture.

---

## 8. Input Embedding

Words/chars become vectors.

Example:

```txt
"h" -> [0.2, -0.8, 1.1 ...]
```

Instead of raw text, model uses numbers.

---

## 9. Positional Encoding

Transformer processes all words together.

So it needs position information.

Otherwise:

```txt
dog bites man
```

and

```txt
man bites dog
```

look same.

Positional encoding adds order.

---

## 10. Masked Multi-Head Attention (IMPORTANT for GPT)

In GPT:

While predicting next token,

model CANNOT see future words.

Example:

```txt
I love eating ___
```

It cannot peek ahead.

Masking hides future tokens.

So token 3 only sees:

```txt
token1 token2
```

not future ones.

This is exactly how your bigram training worked too.

---

## 11. Feed Forward Layer

After attention:

small neural network processes each token.

Think:

```txt
attention = communication
feedforward = thinking
```

---

## 12. Add & Norm

Helps stable training.

Residual connection:

```txt
output = input + new_information
```

Prevents forgetting earlier information.

---

## 13. Final Linear + Softmax

At end:

Transformer predicts probabilities for next token.

Exactly like your bigram model.

Example:

| Token    | Probability |
| -------- | ----------- |
| "pizza"  | 0.70        |
| "burger" | 0.20        |
| "car"    | 0.01        |

Softmax converts scores into probabilities.

---

## 14. BIGRAM vs TRANSFORMER

| Bigram                      | Transformer                  |
| --------------------------- | ---------------------------- |
| Sees 1 previous token       | Sees all previous tokens     |
| Simple lookup probabilities | Learns relationships         |
| No context understanding    | Strong context understanding |
| Tiny model                  | Deep neural network          |
| No attention                | Uses attention               |

---

## 15. Simplest intuition

Your bigram model was like:

> “Next word depends on previous word.”

Transformer says:

> “Next word depends on ALL previous words, but some matter more.”

That “some matter more” idea is attention.

---
