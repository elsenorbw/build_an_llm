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

    print(raw_text[:90])
    print(encoded[:30])
    print(decoded[:30])

    new_text = "Hello, this is a small weasel called Charlie"
    encoded = tokeniser.encode(new_text, allowed_special={"<|endoftext|>"})
    decoded = tokeniser.decode(encoded)

    print(new_text)
    print(encoded[:100])
    print(decoded[:100])

    tokalicious = [(x, tokeniser.decode([x])) for x in encoded]
    for id, decoded in tokalicious:
        print(f"{id}={decoded}")


if __name__ == "__main__":
    main()
