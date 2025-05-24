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
                         batch_size=4,
                         max_length=256,
                         stride=128,
                         shuffle=True,
                         drop_last=True,
                         num_workers=0):
    tokeniser = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokeniser, max_length, stride)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader


def main():
    raw_text = loadfile(filename)
    print(f"Input file {filename} has {len(raw_text)} characters.")
    print(f"Start:{raw_text[:90]}")

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=1,
        max_length=8,
        stride=2,
        shuffle=False)

    data_iter = iter(dataloader)
    for idx in range(2):
        this_batch = next(data_iter)
        print(f"#{idx}: {this_batch}")


if __name__ == "__main__":
    main()
