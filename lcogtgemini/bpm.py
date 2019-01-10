import lcogtgemini
from pyraf import iraf
from astropy.io import fits
import numpy as np
import pkg_resources


def get_bad_pixel_mask(binnings, yroi):
    if lcogtgemini.detector == 'Hamamatsu':
        if lcogtgemini.is_GS:
            bpm_file = pkg_resources.resource_filename('lcogtgemini', 'bpm_gs.fits')
        else:
            bpm_file = pkg_resources.resource_filename('lcogtgemini', 'bpm_gn.fits')

        bpm_hdu = fits.open(bpm_file, uint16=True)

        for i in range(1, lcogtgemini.namps + 1):
            bpm_hdu[i].data = bpm_hdu[i].data[yroi[0]-1:yroi[1]]
            bpm_hdu[i].writeto('bpm.{i}.unbinned.fits'.format(i=i), overwrite=True)
            for binning in binnings:
                binning_list = binning.split('x')
                binx, biny = int(binning_list[0]), int(binning_list[1])
                iraf.unlearn('blkavg')
                binned_bpm_filename = 'bpm.{i}.{x}x{y}.fits'.format(i=i, x=binx, y=biny)
                iraf.blkavg('bpm.{i}.unbinned.fits[1]'.format(i=i),
                            binned_bpm_filename, binx, biny)
                averaged_bpm = fits.open(binned_bpm_filename)
                # Remove some header keywords from the BPM that confuses fixpix
                for keyword in ['LTV1', 'LTV2', 'LTM1_1', 'LTM2_2']:
                    averaged_bpm[0].header.remove(keyword, ignore_missing=True)
                averaged_bpm[0].data[averaged_bpm[0].data > 0.1] = 1
                averaged_bpm[0].data = averaged_bpm[0].data.astype(np.uint8)
                averaged_bpm.writeto(binned_bpm_filename, overwrite=True)

    else:
        for binning in binnings:
            binning_list = binning.split('x')
            binx, biny = int(binning_list[0]), int(binning_list[1])
            for i in range(1, lcogtgemini.namps + 1):
                bpm_data = np.zeros((2048 // int(biny), 1080 // int(binx)), dtype=np.uint8)
                binned_bpm_filename = 'bpm.{i}.{x}x{y}.fits'.format(i=i, x=binx, y=biny)
                fits.writeto(binned_bpm_filename, bpm_data)
