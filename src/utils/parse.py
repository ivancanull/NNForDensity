import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='program setting file')
    return parser.parse_args()

def get_filename(file_path: str) -> str:
    """
    Extract the file name and file extension.
    """

    # get the filename without extension
    file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]

    # get the file extension
    file_extension = os.path.splitext(file_path)[1]

    return [file_name_without_extension, file_extension]
