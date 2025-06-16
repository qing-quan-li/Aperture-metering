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

def predicate_ring_polluted(rin,rout,subcat):
    fwhm_factor_limit = 3.
    if np.all((subcat['distance']+fwhm_factor_limit*subcat['fwhm'] < rin)|(subcat['distance']-fwhm_factor_limit*subcat['fwhm'] > rout)):
        return False
    else:
        return True 

def predicate_inner_polluted(rin_list,subcat):
    fwhm_factor_limit = 3
    list_predicate = []
    for rin in rin_list:
        if np.all((subcat['distance']-fwhm_factor_limit*subcat['fwhm'] > rin)):
            list_predicate.append(False)
        else:
            list_predicate.append(True)
    return np.array(list_predicate)


def light_curve(data,aperture,outer_radius=None):
    # data is n*128*128 array
    # aperture is a list of rin, [r,...]

    r = np.array(aperture)#.reshape(-1,1)
    lc = np.zeros((len(data),len(aperture)))
    lc_err = np.zeros((len(data),len(aperture)))
    pos = np.ones(len(aperture))*63.5
    for i in range(len(data)):
        # do photometry on each image
        #data[i] = data[i].astype(data[i].dtype.newbyteorder('='))
        #s,serr,flag = sep.sum_circle(data[i],[63.5],[63.5],[r],  bkgann=bkgann, gain=1.3)
        #print(data[i],pos,r, bkgann)
        s,serr,flag = sep.sum_circle(data[i],pos,pos,r, bkgann=outer_radius, gain=1.3)
        #print(s,serr,flag)
        lc[i,:] = s
        lc_err[i,:] = serr
    return lc,lc_err
def snr_lc(lc,lc_err):
    #SNR obtained by weighted mean and weight standard deviation, weight = 1/err^2
    weight = 1/lc_err**2
    weight = weight/np.sum(weight,axis = 0,keepdims=True)
    lc_mean = np.sum(lc*weight,axis = 0,keepdims=True)
    lc_std = np.sqrt(np.sum(weight*(lc-lc_mean)**2,axis = 0,keepdims=True))
    snr = lc_mean/lc_std
    return np.squeeze(snr)

def enumerate_aperture_for_a_star(row_record,tree_star,star_catalog):
    t0 = time.time()

    res = tree_star.query_ball_point((row_record['x'], row_record['y']),r = 128*np.sqrt(2))
    subcat = star_catalog.iloc[res]
    subcat = subcat[subcat['star_id'] != row_record['star_id']]
    
    subcat['x'] = subcat['x'] - row_record['x'] # relative position
    subcat['y'] = subcat['y'] - row_record['y']
    subcat['distance'] = np.sqrt(subcat['x']**2 + subcat['y']**2)
    print(subcat)
    star_id = int(row_record['star_id'])
    sample_image_path = f"/home/yichengrui/data/image/new_img_TrES5/out/single/merged/notdebackgrounded/star_{star_id}_with_bkg.fit"
    star_data = fits.getdata(sample_image_path)
    star_data = star_data.astype(star_data.dtype.newbyteorder('='))
    best_snr = -np.inf
    best_good_snr = -np.inf
    best_aperture = [-1,-1,-1]
    best_good_aperture = [-1,-1,-1]
    fwhm = float(row_record['fwhm'])
    inner_aper_radius = fwhm * np.arange(1, 5, 0.05)
    middle_aper_radius_factor = np.arange(1.1,2.1,0.1)
    additional_outer_aper_radius =  np.arange(3,16,3)
    print(inner_aper_radius,middle_aper_radius_factor,additional_outer_aper_radius)
    inner_polluted_list = predicate_inner_polluted(inner_aper_radius,subcat)
    for i in tqdm.tqdm(range(len(inner_aper_radius))):
        inn_ann = inner_aper_radius[i]
        for j in middle_aper_radius_factor:
            mid_ann = j * inn_ann
            for k in additional_outer_aper_radius:
                #print(i,j,k)
                out_ann = k + mid_ann
                
                if out_ann >= 64:
                    break

                ring_polluted = predicate_ring_polluted(mid_ann, out_ann, subcat)
                polluted_this = inner_polluted_list[i] | ring_polluted
                #print(polluted_this)
                light_curve_data = light_curve(star_data,np.array([inn_ann]), outer_radius=[mid_ann, out_ann])
                snr = snr_lc(light_curve_data[0], light_curve_data[1])
                # print(snr_list)
                # for ithsnr in range(len(snr_list)):
                # ithsnr = snr_list
                if snr > best_snr:
                    # print('snr updated')
                    best_snr = snr
                    #print('snr updated')
                    best_aperture = [inn_ann, mid_ann, out_ann]
                    #print(snr,inn_ann,mid_ann,out_ann)
                if snr > best_good_snr and not polluted_this:
                    best_good_snr = snr
                    best_good_aperture = [inn_ann, mid_ann, out_ann]
                    #print('good snr updated')
                    #print(snr,inn_ann,mid_ann,out_ann)
    t1 = time.time()
    enumerate_time = t1 - t0
    return best_snr,best_aperture,best_good_snr,best_good_aperture, enumerate_time
star_catalog_path = '/home/yichengrui/data/image/new_img_TrES5/out/catalog/star_catalog.csv'
star_catalog = pd.read_csv(star_catalog_path)


sample_image_path = "/home/yichengrui/data/image/new_img_TrES5/out/single/merged/notdebackgrounded/star_9_with_bkg.fit"
sample_image_data = fits.getdata(sample_image_path)
print(sample_image_data.shape)
sample_image_data = sample_image_data.astype(sample_image_data.dtype.newbyteorder('='))
light_curve_data = light_curve(sample_image_data, [10,11,12,13,14,15,23],outer_radius = [24,34])
snr = snr_lc(light_curve_data[0], light_curve_data[1])
print(snr)
#construct KD-tree according to xy of star_catalog
from scipy.spatial import KDTree
star_catalog['x'] = star_catalog['x'].astype(float)
star_catalog['y'] = star_catalog['y'].astype(float)
tree_star = KDTree(star_catalog[['x','y']].values)


def process_star(r,tree_star = tree_star,star_catalog = star_catalog):
    print('starting process star',r['star_id'])
    star_id = int(r['star_id'])
    file_path = f"/home/yichengrui/data/image/new_img_TrES5/out/catalog/separate_cat/star_{star_id}.csv"
    if os.path.exists(file_path):
        print(f"file {file_path} already exists, skipping")
        return
    best_snr,best_aperture,best_good_snr,best_good_aperture, enumerate_time = enumerate_aperture_for_a_star(r,tree_star,star_catalog)
    if best_good_snr < 0:
        feasible = 0
    else:
        feasible = 1
    
    
    best_snr = float(best_snr)
    best_inner_radius = float(best_aperture[0])
    best_middle_radius = float(best_aperture[1])
    best_outer_radius = float(best_aperture[2])
    best_good_snr = float(best_good_snr)
    best_good_inner_radius = float(best_good_aperture[0])
    best_good_middle_radius = float(best_good_aperture[1])
    best_good_outer_radius = float(best_good_aperture[2])
    enumerate_time = float(enumerate_time)
    #save to csv
    
    df = pd.DataFrame({
        'star_id': [star_id],
        'best_snr': [best_snr],
        'best_inner_radius': [best_inner_radius],
        'best_middle_radius': [best_middle_radius],
        'best_outer_radius': [best_outer_radius],
        'best_good_snr': [best_good_snr],
        'best_good_inner_radius': [best_good_inner_radius],
        'best_good_middle_radius': [best_good_middle_radius],
        'best_good_outer_radius': [best_good_outer_radius],
        'enumerate_time': [enumerate_time],
        'feasible': [feasible]
    })
    df.to_csv(file_path, index=False)
    print(f"file {file_path} saved")


#multiprocessing for each rows of star_catalog
# use 47 cores

print(star_catalog)
list_rows = []
for i,r in star_catalog.iterrows():
    list_rows.append(r)
with multiprocessing.Pool(processes=47) as pool:
    pool.map(process_star, list_rows)
print('All done!')



