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




for rot_deg in range(0, 360, 36):
    print('processing',rot_deg)
    image_path = f"/home/yichengrui/data/image/new_img_TrES5/stacked/rotated/rot_{rot_deg}_TrES5.fits"
    image_data = fits.getdata(image_path)
    image_data = image_data.astype(image_data.dtype.newbyteorder('='))
    save_chopped_path_with_bkg = "/home/yichengrui/data/image/new_img_TrES5/out/stacked/rotated_notdebackgrounded/"

    bkg = sep.Background(image_data)
    debkg_image_data = image_data - bkg.back()

    header = fits.getheader(image_path)
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
        # save the cut image

        fits.writeto(os.path.join(save_chopped_path_with_bkg, f"rot_{rot_deg}_star_{star_id}_with_bkg.fit"), cut_img, overwrite=True)