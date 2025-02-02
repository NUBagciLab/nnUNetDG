import os
import shutil
import argparse

def rename(folder_dir):
    # out_folder = folder_dir+"_rename"
    file_list = [file for file in os.listdir(folder_dir) if file.endswith('.nii.gz')]

    for file in file_list:
        os.rename(os.path.join(folder_dir, file),
                  os.path.join(folder_dir, file[:-7]+"_0000.nii.gz"))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    parser = get_parser()
    rename(parser.input_folder)