import argparse
import glob
import os
import random
from pathlib import Path
import numpy as np
from sklearn.cross_validation import train_test_split
from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    tfrecords = glob.glob(os.path.join(source, "*.tfrecord"))
    train_files, val_test_files = train_test_split(tfrecords, test_size=0.4)
    val_files, test_files= train_test_split(val_test_files, test_size=0.5)
    training = os.mkdir(os.path.join(source, 'train'), exist_ok=True)
    val = os.mkdir(os.path.join(source, 'val'), exist_ok=True)
    test = os.mkdir(os.path.join(source, 'test'), exist_ok=True)

    dirs = [training, val, test]
    files = [train_files, val_files, test_files]

    for dir, file in zip(dirs, files):
        print(f"Moving files to {dir}.")
        for file in files:
            Path(file).rename(os.path.join(dir, Path(file).name))
        print(f"Loaded {len(files)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)