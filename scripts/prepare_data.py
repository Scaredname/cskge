"""Extract bundled datasets without overwriting existing files."""
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = (ROOT / "data").resolve()
    with ZipFile(ROOT / "data.zip") as archive:
        for member in archive.infolist():
            target = (data / member.filename).resolve()
            if not target.is_relative_to(data):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if not target.exists():
                archive.extract(member, data)
    print(f"Datasets ready: {data}")


if __name__ == "__main__":
    main()
