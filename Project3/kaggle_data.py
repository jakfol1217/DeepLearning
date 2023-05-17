import zipfile
import os
import shutil
import glob
from tqdm import tqdm


# This dataset doesn't seem to be available via API, so download it and put into the .data folder manually
def process_lsun_dataset():
    path = glob.glob('.data/*.zip')[0]
    with zipfile.ZipFile(path) as f:
        for file in tqdm(f.infolist(), desc="Extracting"):
            try:
                f.extract(file, '.data/')
            except zipfile.error as e:
                print(e)
    print("Data extracted")
    os.remove(path)
    shutil.rmtree('.data/sample/')
    print("Unnecessary files removed")


if __name__ == '__main__':
    process_lsun_dataset()
