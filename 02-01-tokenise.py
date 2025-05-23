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

def main():
    raw_text = loadfile(filename)
    print(f"Input file {filename} has {len(raw_text)} characters.")
    print(f"Start:{raw_text[:90]}")
    toks = tokenise(raw_text)
    print(f"Token Count: {len(toks)}")
    print(f"Token Sample: {toks[:20]}")



if __name__ == "__main__":
    main()

