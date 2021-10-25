import os
from pathlib import Path
import shutil

SANTANDER_HOME = str(Path("~/.data/santander").expanduser())


def _check_or_unzip_santander():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = Path(SANTANDER_HOME)
    if (target_dir / "train.csv").exists():
        if (target_dir / "test.csv").exists():
            print("Found files!")
            return True

    archive = Path(current_dir) / "santander-customer-satisfaction.zip"
    if archive.exists() and not archive.is_dir():
        shutil.unpack_archive(str(archive), extract_dir=SANTANDER_HOME)
        print(f"Unpacked archive to {SANTANDER_HOME}")
    else:
        print(str(archive), "does not exist!")



def load_santander():
    import pandas as pd
    _check_or_unzip_santander()
    train_data = pd.read_csv(f"{SANTANDER_HOME}/train.csv", index_col=0)
    test_data = pd.read_csv(f"{SANTANDER_HOME}/test.csv", index_col=0)
    return train_data, test_data
