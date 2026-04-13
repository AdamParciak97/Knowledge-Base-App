from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
TARGET_DIR = Path("models") / MODEL_NAME


def main() -> None:
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME, cache_folder=str(TARGET_DIR.parent / ".cache"))
    model.save(str(TARGET_DIR))
    print(f"Model cached in {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    main()
