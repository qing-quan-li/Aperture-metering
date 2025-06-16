import sep
import astropy.io.fits as fits
import astropy.wcs as WCS
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import tqdm

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
image_data = fits.getdata(first_image_path)
image_data = image_data.astype(image_data.dtype.newbyteorder('='))
save_chopped_path_no_bkg = "/home/yichengrui/data/image/new_img_TrES5/out/single/first/debackgrounded/"
save_chopped_path_with_bkg = "/home/yichengrui/data/image/new_img_TrES5/out/single/first/notdebackgrounded"

bkg = sep.Background(image_data)
debkg_image_data = image_data - bkg.back()

header = fits.getheader(first_image_path)
wcs = WCS.WCS(header)


# enumerate the star catalog, cut the pixels and save them
for i in tqdm.tqdm(range(len(star_catalog_df))):
    star_id = star_catalog_df['star_id'][i]
    ra = star_catalog_df['ra'][i]
    dec = star_catalog_df['dec'][i]
    # convert ra and dec to x and y
    x, y = wcs.all_world2pix(ra, dec, 0)
    x,y = float(x), float(y)
    cut_img = cut_pixels(image_data, x, y)
    cut_img_debkg = cut_pixels(debkg_image_data, x, y)
    # save the cut image
    fits.writeto(os.path.join(save_chopped_path_no_bkg, f"star_{star_id}_nobkg.fit"), cut_img_debkg, overwrite=True)
    fits.writeto(os.path.join(save_chopped_path_with_bkg, f"star_{star_id}_with_bkg.fit"), cut_img, overwrite=True)