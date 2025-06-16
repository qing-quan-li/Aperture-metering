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
#star_catalog_path = '/home/yichengrui/data/image/new_img_TrES5/out/catalog/star_catalog.csv'
#star_catalog = pd.read_csv(star_catalog_path)


sample_image_path = "E:\\data\\out\\single\\merged\\notdebackgrounded\\star_582_with_bkg.fit"
sample_image_data = fits.getdata(sample_image_path)
header = fits.getheader(sample_image_path)
print(header)
print(sample_image_data.shape)
sample_image_data = sample_image_data.astype(sample_image_data.dtype.newbyteorder('='))
# light_curve_data = light_curve(sample_image_data, [10,11,12,13,14,15,23],outer_radius = [24,34])
# snr = snr_lc(light_curve_data[0], light_curve_data[1])
# #
inner = 12.737135
middle = 24.2005573
outer = 39.200557
light_curve_data_enum,light_curve_data_enum_err = light_curve(sample_image_data, [inner],outer_radius = [middle,outer])
snr_enumerate = snr_lc(light_curve_data_enum, light_curve_data_enum_err)
inner = 8.796395
middle = 10.4
outer = 12.35
light_curve_data_resnet,light_curve_data_resnet_err = light_curve(sample_image_data, [inner],outer_radius = [middle,outer])
snr_resnet = snr_lc(light_curve_data_resnet, light_curve_data_resnet_err)
light_curve_data_enum = np.squeeze(light_curve_data_enum)
light_curve_data_resnet = np.squeeze(light_curve_data_resnet)
light_curve_data_enum_err = np.squeeze(light_curve_data_enum_err)
light_curve_data_resnet_err = np.squeeze(light_curve_data_resnet_err)
print(f"Enumerate SNR: {snr_enumerate}, Aperture: {inner}, {middle}, {outer}")
print(f"ResNet SNR: {snr_resnet}, Aperture: {inner}, {middle}, {outer}")
# plt.errorbar(np.arange(len(light_curve_data_enum)), light_curve_data_enum, yerr=light_curve_data_enum_err, label='Enumerate SNR = {snr_enumerate}', fmt='o')
# plt.errorbar(np.arange(len(light_curve_data_resnet)), light_curve_data_resnet, yerr=light_curve_data_resnet_err, label='ResNet SNR = {snr_ResNet}', fmt='o')
# plt.xlabel('Time')
# plt.ylabel('Flux')
# plt.show()
# plt.figure()
first_frame = np.mean(sample_image_data,axis=0)
m = np.mean(first_frame)
print(m)
std = np.std(first_frame)
# fig, ax = plt.subplots()
# ax.tick_params(labelbottom=False)

# # Remove y-axis tick labels
# ax.tick_params(labelleft=False)
# plt.imshow(first_frame, cmap='gray', origin='lower',vmin=m-0.5*std, vmax=m+3*std)
phase = np.linspace(0, 2*np.pi, 100)
# plt.plot(63.5 + inner * np.cos(phase), 63.5 + inner * np.sin(phase), color='red')
# plt.plot(63.5 + middle * np.cos(phase), 63.5 + middle * np.sin(phase), color='blue')
# plt.plot(63.5 + outer * np.cos(phase), 63.5 + outer * np.sin(phase), color='green')
# plt.axis('off')
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.show()
#perturbation = np.zeros(first_frame.shape)
x = np.arange(128)
y = np.arange(128)
X, Y = np.meshgrid(x, y)
perturbation = np.exp(-((X-93)**2 + (Y-44)**2) / (2 * 5**2)).reshape(1,128,128)
ndpoint = len(light_curve_data_enum)
resnet_snr_list = []
enumerate_snr_list = []
dmag_list = []
for i in range(15):
    dmag = i*2/10
    dmag_list.append(dmag)
    factor = (100**(dmag/5)-1)*11*np.sin(np.arange(ndpoint)*2*np.pi/ndpoint).reshape(-1,1,1)
    sample_image_data+= perturbation * factor

    # plt.imshow(perturbation, cmap='gray', origin='lower',vmin=0, vmax=1)
    # plt.show()
    # print(np.mean(light_curve_data_resnet/light_curve_data_resnet_err))
    # print(np.mean(light_curve_data_enum/light_curve_data_enum_err))

    inner = 12.737135
    middle = 24.2005573
    outer = 39.200557
    light_curve_data_enum,light_curve_data_enum_err = light_curve(sample_image_data, [inner],outer_radius = [middle,outer])
    snr_enumerate = snr_lc(light_curve_data_enum, light_curve_data_enum_err)
    inner = 8.796395
    middle = 10.35
    outer = 12.35
    light_curve_data_resnet,light_curve_data_resnet_err = light_curve(sample_image_data, [inner],outer_radius = [middle,outer])
    snr_resnet = snr_lc(light_curve_data_resnet, light_curve_data_resnet_err)
    light_curve_data_enum = np.squeeze(light_curve_data_enum)
    light_curve_data_resnet = np.squeeze(light_curve_data_resnet)
    light_curve_data_enum_err = np.squeeze(light_curve_data_enum_err)
    light_curve_data_resnet_err = np.squeeze(light_curve_data_resnet_err)
    print(f"Enumerate SNR: {snr_enumerate}, Aperture: {inner}, {middle}, {outer}")
    print(f"ResNet SNR: {snr_resnet}, Aperture: {inner}, {middle}, {outer}")
    resnet_snr_list.append(snr_resnet)
    enumerate_snr_list.append(snr_enumerate)
    if i== 0 or i==5 or i==10:
        plt.figure()
        plt.errorbar(np.arange(len(light_curve_data_resnet)), light_curve_data_resnet, yerr=light_curve_data_resnet_err, label=f'ResNet-18 SNR = {str(snr_resnet)[:5]}; '+r'$\Delta$mag = '+str(dmag), fmt='o',alpha=0.5)
        plt.errorbar(np.arange(len(light_curve_data_enum)), light_curve_data_enum, yerr=light_curve_data_enum_err, label=f'Enumerate SNR = {str(snr_enumerate)[:5]}; '+r'$\Delta$mag = '+str(dmag), fmt='o',alpha=0.5)
        
        plt.xlabel('Number of frame')
        plt.ylabel('Flux')
        plt.legend(loc='upper left')
        #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

plt.figure()
plt.plot(dmag_list, resnet_snr_list, '.-',label='ResNet-18')
plt.plot(dmag_list, enumerate_snr_list,'.-', label='Aperture enumeration')

plt.xlabel(r'$\Delta$mag')
plt.ylabel('SNR')
plt.legend()
plt.show()    