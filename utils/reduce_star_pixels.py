import sep
import astropy.io.fits as fits
import astropy.wcs as WCS
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import tqdm
import multiprocessing

# Reduce the star pixel fits file into a fits file
sample_name = "/home/yichengrui/data/image/new_img_TrES5/out/single/general/1/debackgrounded/img_2_star_1_nobkg.fit"
def get_id_from_path(path):
    id_this = int(path.split('/')[-1].split('_')[1])
    return id_this
print(get_id_from_path(sample_name))

def merge_individual_fits_file(id_star):
    print(f'processing star {id_star}')
    debackgrounded_path = f"/home/yichengrui/data/image/new_img_TrES5/out/single/general/{id_star}/debackgrounded/"
    notdebackgrounded_path = f"/home/yichengrui/data/image/new_img_TrES5/out/single/general/{id_star}/notdebackgrounded/"
    debackgrounded_target_path = f"/home/yichengrui/data/image/new_img_TrES5/out/single/merged/debackgrounded/star_{id_star}_nobkg.fit"
    notdebackgrounded_target_path = f"/home/yichengrui/data/image/new_img_TrES5/out/single/merged/notdebackgrounded/star_{id_star}_with_bkg.fit"
    merge_files(debackgrounded_path,debackgrounded_target_path)
    merge_files(notdebackgrounded_path,notdebackgrounded_target_path)

def merge_files(source_path, target_path):
    if os.path.exists(target_path):
        print(f"{target_path} already exists, skipping...")
        return
    # Get all the files in the source path
    files = glob.glob(os.path.join(source_path, "*"))
    # sort files by get_id_from_path
    files.sort(key=lambda x: get_id_from_path(x))
    # Get the first file to use as a template for the output file
    # Create an empty list to store the data
    data_list = []
    # Loop through all the files and read the data
    for file in tqdm.tqdm(files):
        data = fits.getdata(file)
        if data.shape[0] ==128 and data.shape[1] == 128:
            data_list.append(data)
        else:
            print(f"Skipping file {file} due to unexpected shape {data.shape}")
            continue
    # Stack the data along the first axis

    stacked_data = np.stack(data_list, axis=0)

    # Write the stacked data to a new file
    fits.writeto(target_path, stacked_data, overwrite=True)

# run merge_individual_fits_file for star id 1->2677 using 47 cores in multiprocessing
if __name__ == '__main__':
    star_id_list = list(range(1,2678))
    with multiprocessing.Pool(processes=47) as pool:
        pool.map(merge_individual_fits_file, star_id_list)
    print('All done!')