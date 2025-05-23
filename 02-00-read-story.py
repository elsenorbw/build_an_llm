#!/usr/bin/env python


filename = 'data/the-verdict.txt'

def loadfile(filename: str) -> str:
    with open(filename, 'r', encoding="utf-8") as f:
        result = f.read()
    return result


def main():
    raw_text = loadfile(filename)
    print(f"Input file {filename} has {len(raw_text)} characters.")
    print(f"Start:{raw_text[:90]}")


if __name__ == "__main__":
    main()

