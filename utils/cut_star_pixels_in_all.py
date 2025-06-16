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


def cut_pixels(img,x,y):
    cut_range = 128
    cut_half_range = cut_range // 2
    # round the coordinates
    x = int(round(x)+0.5)
    y = int(round(y)+0.5)
    # cut the pixels
    cut_img = img[y-cut_half_range:y+cut_half_range, x-cut_half_range:x+cut_half_range]
    return cut_img

star_catalog_path = "/home/yichengrui/data/image/new_img_TrES5/out/catalog/star_catalog.csv"
star_catalog_df = pd.read_csv(star_catalog_path)


first_image_path = "/home/yichengrui/data/image/new_img_TrES5/single/cal_174601968348527217960132_TrES5-0002.fit"
def get_id_from_path(path):
    id_this = int(path.split('-')[-1].strip('.fit').strip(".fits"))
    return id_this
def register_one_image(image_path, star_catalog_df,image_id):
    print('processing image %d' % image_id)
    image_data = fits.getdata(image_path)
    image_data = image_data.astype(image_data.dtype.newbyteorder('='))

    bkg = sep.Background(image_data)
    debkg_image_data = image_data - bkg.back()

    header = fits.getheader(image_path)
    wcs = WCS.WCS(header)


    # enumerate the star catalog, cut the pixels and save them
    for i in tqdm.tqdm(range(len(star_catalog_df))):
        save_chopped_path_no_bkg = f"/home/yichengrui/data/image/new_img_TrES5/out/single/general/{i+1}/debackgrounded"
        save_chopped_path_with_bkg = f"/home/yichengrui/data/image/new_img_TrES5/out/single/general/{i+1}/notdebackgrounded"

        star_id = star_catalog_df['star_id'][i]
        ra = star_catalog_df['ra'][i]
        dec = star_catalog_df['dec'][i]
        # convert ra and dec to x and y
        x, y = wcs.all_world2pix(ra, dec, 0)
        x,y = float(x), float(y)
        cut_img = cut_pixels(image_data, x, y)
        cut_img_debkg = cut_pixels(debkg_image_data, x, y)
        # save the cut image
        fits.writeto(os.path.join(save_chopped_path_no_bkg, f"img_{image_id}_star_{star_id}_nobkg.fit"), cut_img_debkg, overwrite=True)
        fits.writeto(os.path.join(save_chopped_path_with_bkg, f"img_{image_id}_star_{star_id}__with_bkg.fit"), cut_img, overwrite=True)

#print(get_id_from_path(first_image_path))
files = glob.glob("/home/yichengrui/data/image/new_img_TrES5/single/*.fit")
# sort the files by get_id_from_path
files.sort(key=get_id_from_path)
print(files)

def process_image(image_path):
    image_id = get_id_from_path(image_path)
    register_one_image(image_path, star_catalog_df, image_id)

# Create a pool of worker processes maxnumber = 47
pool = multiprocessing.Pool(processes=47)

# Map the process_image function to each file in parallel
pool.map(process_image, files)

# Close the pool and wait for all processes to finish
pool.close()
pool.join()