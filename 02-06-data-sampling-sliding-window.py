#!/usr/bin/env python

from importlib.metadata import version
import tiktoken

filename = 'data/the-verdict.txt'


def loadfile(filename: str) -> str:
    with open(filename, 'r', encoding="utf-8") as f:
        result = f.read()
    return result


def main():
    raw_text = loadfile(filename)
    print(f"Input file {filename} has {len(raw_text)} characters.")
    print(f"Start:{raw_text[:90]}")

    tt_version = version("tiktoken")
    print(f"tiktoken version is {tt_version}")

    tokeniser = tiktoken.get_encoding("gpt2")

    encoded = tokeniser.encode(raw_text, allowed_special={"<|endoftext|>"})
    decoded = tokeniser.decode(encoded)

    # create the sliding window inputs
    encoded_sample = encoded[50:]
    context_size = 4
    x = encoded_sample[:context_size]
    y = encoded_sample[1:context_size + 1]
    print(f"x:{x}")
    print(f"y:{y}")

    for i in range(1, context_size + 1):
        context = encoded_sample[:i]
        desired = encoded_sample[i]
        print(f"{context} ---> {desired}")
        print(f"{tokeniser.decode(context)
                 } ---> {tokeniser.decode([desired])}")


if __name__ == "__main__":
    main()
