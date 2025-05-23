#!/usr/bin/env python

import re

filename = 'data/the-verdict.txt'


def loadfile(filename: str) -> str:
    with open(filename, 'r', encoding="utf-8") as f:
        result = f.read()
    return result


class SimpleTokeniserV1:
    def __init__(self):
        self.vocab = None
        self.vocab_inverted = None

    def _tokenise(self, text: str) -> list[str]:
        tokens = re.split(r'([,.:;<>()?_!"\']|--|\s)', text)
        clean_tokens = [x.strip() for x in tokens if x.strip()]
        return clean_tokens

    def _set_vocabulary_from_tokens(self, tokens: list[str]):
        words = sorted(set(tokens))
        self.vocab = {word: idx for idx, word in enumerate(words)}
        self.vocab_inverted = {idx: word for idx, word in enumerate(words)}

    def encode(self, text: str) -> list[int]:
        toks = self._tokenise(text)
        if self.vocab is None:
            self._set_vocabulary_from_tokens(toks)
        result = [self.vocab[x] for x in toks]
        return result

    def decode(self, token_ids: list[int]) -> list[str]:
        if self.vocab is None:
            raise RuntimeError(
                "You are trying to decode something before loading the vocab dicts.")
        result = [self.vocab_inverted[x] for x in token_ids]
        return result


def main():
    raw_text = loadfile(filename)
    print(f"Input file {filename} has {len(raw_text)} characters.")
    print(f"Start:{raw_text[:90]}")

    tokeniser = SimpleTokeniserV1()

    encoded = tokeniser.encode(raw_text)
    decoded = tokeniser.decode(encoded)

    print(raw_text[:90])
    print(encoded[:30])
    print(decoded[:30])


if __name__ == "__main__":
    main()
