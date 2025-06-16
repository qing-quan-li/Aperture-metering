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
import time

star_info_table_path = "/home/yichengrui/data/image/new_img_TrES5/out/catalog/star_catalog.csv"
star_info_table = pd.read_csv(star_info_table_path)
tab_list = []
for i in tqdm.tqdm(range(1,2678)):
    tab_path_this = f"/home/yichengrui/data/image/new_img_TrES5/out/catalog/separate_cat/star_{i}.csv"
    tab_this = pd.read_csv(tab_path_this)
    tab_list.append(tab_this)
tab = pd.concat(tab_list)
# Merge by star_id
star_info_table = star_info_table.merge(tab, on='star_id', how='left')
# Save the merged table
save_path = "/home/yichengrui/data/image/new_img_TrES5/out/catalog/merged_star_catalog.csv"
star_info_table.to_csv(save_path, index=False)
star_info_table = star_info_table[star_info_table['best_good_snr']<0]
print(len(star_info_table[star_info_table['feasible']==0]))