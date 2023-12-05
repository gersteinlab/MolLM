import os
import glob
import zipfile
import re
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

# Directory path
main_dir_path = "./output-text/"

# Function to handle processing of individual zip files
def process_zip_file(zip_file):
    good_cids = []
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            files = zip_ref.namelist()
            for file_name in files:
                match = re.search(r"mol_(\d+)_smiles\.txt", file_name)
                if match:
                    cid = match.group(1)
                    if f'mol_{cid}_text.txt' in files:
                        good_cids.append(cid)
    except Exception as e:
        print(f'Failed due to {e}')
    return good_cids

if __name__ == "__main__":
    all_zip_files = []
    for folder_number in sorted(os.listdir(main_dir_path)):
        sub_dir_path = os.path.join(main_dir_path, folder_number)

        # Check if path is a directory
        if os.path.isdir(sub_dir_path):
            # Get the list of zip files
            zip_files = sorted(glob.glob(os.path.join(sub_dir_path, "*.zip")))
            all_zip_files.extend(zip_files)

    # Create a multiprocessing Pool
    pool = Pool(processes=5)

    # Create a tqdm progress bar
    pbar = tqdm(total=len(all_zip_files))

    # Create a list to hold the results
    good_cids = []

    # Use imap_unordered to process zip files as results come in
    for result in pool.imap_unordered(process_zip_file, all_zip_files):
        # Update the progress bar
        pbar.update()
        # Extend the list of good_cids with the results
        good_cids.extend(result)

    # Close the progress bar
    pbar.close()

    # Write good_cids to a text file
    with open('cids.txt', 'w') as f:
        for cid in good_cids:
            f.write("%s\n" % cid)

    print(f'Processed {len(all_zip_files)} zips. Total cids: {len(good_cids)}')
