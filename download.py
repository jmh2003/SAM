# pip install gdown

import os
import zipfile

import gdown


def download_utkface_from_drive():
    data_dir = "./data"
    utkface_dir = "./data/UTKFace"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(utkface_dir, exist_ok=True)

    # https://drive.google.com/drive/folders/1HROmgviy4jUUUaCdvvrQ8PcqtNg2jn3G
    folder_id = "1HROmgviy4jUUUaCdvvrQ8PcqtNg2jn3G"

    try:
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{folder_id}",
            output=utkface_dir,
            quiet=False,
            use_cookies=False,
        )
        return True

    except Exception as e:
        return False


if __name__ == "__main__":
    download_utkface_from_drive()
