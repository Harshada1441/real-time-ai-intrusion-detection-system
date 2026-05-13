import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIRECTORIES = ["src", "models", "ui"]


def main() -> None:
    for name in DIRECTORIES:
        os.makedirs(ROOT / name, exist_ok=True)
    print("Created directories: " + ", ".join(DIRECTORIES))


if __name__ == "__main__":
    main()
