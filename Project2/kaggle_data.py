import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import py7zr


# Get data from kaggle You need to generate api key from kaggle and put it in ~/.kaggle/kaggle.json or
# C:\Users<Windows-username>.kaggle\kaggle.json
def get_audioset_from_kaggle(load_test=False):

    # This function does not unpack test data by default, as it takes a lot of time and is unnecessary for training
    # It is advices to simply extract the test data by hand later

    api = KaggleApi()
    api.authenticate()

    api.competition_download_files('tensorflow-speech-recognition-challenge', '.data-audioset')
    print('Files downloaded')
    with zipfile.ZipFile('.data-audioset/tensorflow-speech-recognition-challenge.zip') as file:
        file.extractall('.data-audioset/')

    os.remove('.data-audioset/tensorflow-speech-recognition-challenge.zip')
    print('First file extraction completed')
    os.remove('.data-audioset/link_to_gcp_credits_form.txt')
    with py7zr.SevenZipFile('.data-audioset/train.7z') as file:
        file.extractall('.data-audioset/')

    print("Train dataset extracted")
    if load_test:
        with py7zr.SevenZipFile('.data-audioset/test.7z') as file:
            file.extractall('.data-audioset/')

        print("Test dataset extracted")

    with py7zr.SevenZipFile('.data-audioset/sample_submission.7z') as file:
        file.extractall('.data-audioset/')

    print("Sample submission extracted")
    # Remove left files
    os.remove('.data-audioset/train.7z')
    if load_test:
        os.remove('.data-audioset/test.7z')
    os.remove('.data-audioset/sample_submission.7z')


if __name__ == '__main__':
    get_audioset_from_kaggle()
