"""Download and unpack the original PhysioNet Challenge 2012 Set A data.

Run from the repository root: python3 scripts/download_data.py
Uses only the Python standard library.
"""
from pathlib import Path
import shutil
import tarfile
import urllib.request

BASE_URL = 'https://physionet.org/files/challenge-2012/1.0.0/'


def main():
    data = Path(__file__).resolve().parents[1] / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for name in ('set-a.tar.gz', 'Outcomes-a.txt'):
        destination = data / name
        if not destination.exists():
            temporary = data / (name + '.part')
            print(f'Downloading {name}...')
            try:
                with urllib.request.urlopen(BASE_URL + name, timeout=120) as source:
                    with temporary.open('wb') as target:
                        shutil.copyfileobj(source, target)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            print(f'Using existing {name}.')

    records = data / 'set-a'
    records.mkdir(exist_ok=True)
    count = 0
    with tarfile.open(data / 'set-a.tar.gz', 'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) != 2 or parts[0] != 'set-a' or not parts[1].endswith('.txt'):
                raise ValueError(f'Unexpected archive entry: {member.name}')
            with archive.extractfile(member) as source:
                with (records / parts[1]).open('wb') as target:
                    shutil.copyfileobj(source, target)
            count += 1
    if count != 4000:
        raise ValueError(f'Expected 4000 record files; found {count}.')
    print(f'Ready: {count} record files in {records}')
    print(f'Outcomes: {data / "Outcomes-a.txt"}')


if __name__ == '__main__':
    main()
