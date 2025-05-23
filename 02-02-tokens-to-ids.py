#!/usr/bin/env python

import re

filename = 'data/the-verdict.txt'

def loadfile(filename: str) -> str:
    with open(filename, 'r', encoding="utf-8") as f:
        result = f.read()
    return result

def tokenise(text: str) -> list[str]:
    tokens = re.split(r'([,.:;<>()?_!"\']|--|\s)', text)
    clean_tokens = [x.strip() for x in tokens if x.strip()]
    print(f"Text to tokenise: {text}")
    print(f"Tokens: {tokens}")
    print(f"Clean tokens: {clean_tokens}")
    return clean_tokens

def vocabulary_from_tokens(tokens: list[str]) -> dict[str:int]:
    words = sorted(set(tokens))
    print(f"Unique words for vocabulary: {len(words)}")
    vocab = {word: idx for idx, word in enumerate(words)}
    return vocab 


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    result = [vocab[x] for x in text]
    return result

def decode(token_ids: list[int], vocab: dict[str, int]) -> list[str]:
    inverted_vocab = {b: a for a, b in vocab.items()}
    result = [inverted_vocab[x] for x in token_ids]
    return result 

def main():
    raw_text = loadfile(filename)
    print(f"Input file {filename} has {len(raw_text)} characters.")
    print(f"Start:{raw_text[:90]}")
    toks = tokenise(raw_text)
    print(f"Token Count: {len(toks)}")
    print(f"Token Sample: {toks[:20]}")
    vocab = vocabulary_from_tokens(toks)
    print(vocab)

    print(toks[:30])
    encoded = encode(toks, vocab)
    print(encoded[:30])
    decoded = decode(encoded, vocab)
    print(decoded[:30])



if __name__ == "__main__":
    main()

