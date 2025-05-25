#!/usr/bin/env python3

import torch


def mydot(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


inputs = torch.tensor(

    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

print(f"inputs: {inputs}")


query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])

print(f"attn_scores_2: {attn_scores_2}")

# Calculate the dot product for each attention score
for i, x_i in enumerate(inputs):
    the_score = torch.dot(x_i, query)
    bry_score = mydot(x_i, query)
    print(f"score for #{i} ({x_i}) . ({query}) is {
          the_score} (bry={bry_score})")
    attn_scores_2[i] = the_score

print(f"attn_scores_2: {attn_scores_2}")

# normalise those suckers to a total of 1.0
attn_weights = attn_scores_2 / attn_scores_2.sum()
print(f"attn_weights: {attn_weights} => {attn_weights.sum()}")


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_sm = softmax_naive(attn_scores_2)
print(f"attn_weights_sm: {attn_weights_sm} => {attn_weights_sm.sum()}")

attn_weights_good_sm = torch.softmax(attn_scores_2, dim=0)
print(f"attn_weights_good_sm: {attn_weights_good_sm} => {
      attn_weights_good_sm.sum()}")
