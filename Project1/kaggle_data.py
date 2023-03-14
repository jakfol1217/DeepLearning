import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import py7zr

# Get data from kaggle
# You need to generate api key from kaggle and put it in ~/.kaggle/kaggle.json or C:\Users<Windows-username>.kaggle\kaggle.json
def get_cifar10_from_kaggle():
    api = KaggleApi()
    api.authenticate()

    api.competition_download_files('cifar-10', '.data-cifar')
    print('Files downloaded')
    with zipfile.ZipFile('.data-cifar/cifar-10.zip') as file:
        file.extractall('.data-cifar/')

    os.remove('.data-cifar/cifar-10.zip')
    print('First file extraction completed')
    # I had to extract data because accessing each zipped file (to create a pytorch dataset and dataloader) was super slow
    with py7zr.SevenZipFile('.data-cifar/train.7z') as file:
        file.extractall('.data-cifar/')

    print("Train dataset extracted")

    with py7zr.SevenZipFile('.data-cifar/test.7z') as file:
        file.extractall('.data-cifar/')

    print("Test dataset extracted")

    os.remove('.data-cifar/train.7z')
    os.remove('.data-cifar/test.7z')


if __name__ == '__main__':
    get_cifar10_from_kaggle()