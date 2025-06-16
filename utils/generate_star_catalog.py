import sep
import astropy.io.fits as fits
import astropy.wcs as WCS
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

path_stacked_img = "/home/yichengrui/data/image/new_img_TrES5/stacked/n_506_PID_174601970006980221375065_from_cal_174601970005800309185279_TrES5-0522.fit"
output_catalog_path = "/home/yichengrui/data/image/new_img_TrES5/out/catalog/"
visualize_output_path = "/home/yichengrui/data/image/new_img_TrES5/out/visualize/"
output_catalog_name = "star_catalog.csv"
# star catalog
# id ra dec x_template y_template fwhm
# read stacked image
data = fits.getdata(path_stacked_img)
data = data.astype(data.dtype.newbyteorder('='))
# read header
header = fits.getheader(path_stacked_img)
# get wcs
wcs = WCS.WCS(header)
# detect stars using sep
bkg = sep.Background(data)
data_sub = data - bkg.back()
# detect objects 
# use a mask to exclude the background
bkgrms = bkg.rms()
mask = np.zeros_like(data, dtype=bool)
mask[:150,:] = True
# mask[-65:,:] = True
mask[:, 9450:] = True
# mask[:, :65] = True
# data_sub[mask] = 0
objects = sep.extract(data_sub,7,minarea = 7, mask=mask,err = bkg.globalrms)
# convert to pandas dataframe

df = pd.DataFrame(objects)
y_max,x_max = data.shape
boundary_limit = 66
df = df[(df['x'] > 2*boundary_limit) & (df['x'] < x_max-5*boundary_limit) & (df['y'] > 4*boundary_limit) & (df['y'] < y_max-2*boundary_limit)]
#reindex the dataframe
df = df.reset_index(drop=True)
print('found %d objects' % len(df))
# convert x and y to ra and dec
df['ra'], df['dec'] = wcs.all_pix2world(df['x'], df['y'], 0)
#save catalog star_id start from 1
df['star_id'] = df.index + 1

# calculate fwhm using a,b
df['fwhm'] = 2.355*np.sqrt(df['a'] * df['b'])
print('visualizing')
# visualize the result (x,y,fwhm) with matplotlib, save it as pdf
plt.figure(figsize=(10, 10))
m = np.mean(data)
s = np.std(data)
plt.imshow(data, cmap='gray', origin='lower',vmin=m-s, vmax=m+s)
plt.colorbar()
# circle out the stars
for i in range(len(df)):
    circle = plt.Circle((df['x'][i]+1, df['y'][i]+1), df['fwhm'][i]/2, color='red', fill=False)
    plt.gca().add_artist(circle)
#save the figure
plt.savefig(os.path.join(visualize_output_path, 'star_catalog.pdf'), dpi=300)

print('saving_result')
# save catalog
print(df.columns)
df = df[['star_id', 'ra', 'dec', 'x', 'y', 'a', 'b', 'theta', 'flux', 'fwhm']]
df.to_csv(os.path.join(output_catalog_path, output_catalog_name), index=False)

