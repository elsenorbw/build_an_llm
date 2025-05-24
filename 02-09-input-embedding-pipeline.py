#!/usr/bin/env python

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


filename = 'data/the-verdict.txt'


def loadfile(filename: str) -> str:
    with open(filename, 'r', encoding="utf-8") as f:
        result = f.read()
    return result


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokeniser, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Turn the whole text into tokens
        tokens = tokeniser.encode(txt)

        # Using a sliding window, chunk into
        # overlapping sequences of max_length
        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i:i + max_length]
            target_chunk = tokens[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(txt,
                         tokeniser,
                         batch_size=4,
                         max_length=256,
                         stride=128,
                         shuffle=True,
                         drop_last=True,
                         num_workers=0):
    dataset = GPTDatasetV1(txt, tokeniser, max_length, stride)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader


def main():
    # We don't want the random kind of random, we want the fixed kind of random
    torch.manual_seed(123)

    # We will keep this out here as we need the size of the vocab
    tokeniser = tiktoken.get_encoding("gpt2")
    vocab_size = tokeniser.max_token_value + 1
    output_dimension = 256

    print(f"Embedding will have a vocab size of {
          vocab_size} and a dimension of {output_dimension}")

    raw_text = loadfile(filename)
    print(f"Input file {filename} has {len(raw_text)} characters.")
    print(f"Start:{raw_text[:90]}")

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dimension)

    max_length = 4
    data_loader = create_dataloader_v1(
        raw_text,
        tokeniser,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False)

    data_iter = iter(data_loader)
    inputs, targets = next(data_iter)
    print(f"Token IDs: {inputs}")
    print(f"Inputs Shape: {inputs.shape}")

    token_embeddings = token_embedding_layer(inputs)
    print(f"token_embeddings Shape: {token_embeddings.shape}")

    # Absolute embedding approach (apparently)
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dimension)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(f"pos_embeddings shape: {pos_embeddings.shape}")
    print(f"pos_embeddings:\n{pos_embeddings}")
    print(f"torch.arange(6): {torch.arange(6)}")


if __name__ == "__main__":
    main()
