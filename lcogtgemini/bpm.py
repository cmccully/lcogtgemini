import lcogtgemini
from pyraf import iraf
from astropy.io import fits
import numpy as np


def get_bad_pixel_mask(binx, biny, yroi):
    if lcogtgemini.detector == 'Hamamatsu':
        if lcogtgemini.is_GS:
            bpm_file = 'bpm_gs.fits'
            bpm_hdu = fits.open(bpm_file, uint16=True)
            for i in range(1, 13):
                bpm_hdu[i].data = bpm_hdu[i].data[yroi[0]-1:yroi[1]]
                bpm_hdu[i].writeto('bpm.{i}.unbinned.fits'.format(i=i), overwrite=True)
                iraf.unlearn('blkavg')
                iraf.blkavg('bpm.{i}.unbinned.fits[1]'.format(i=i), 'bpm.{i}.fits'.format(i=i), binx, biny)
                averaged_bpm = fits.open('bpm.{i}.fits'.format(i=i))
                # Remove some header keywords from the BPM that confuses fixpix
                for keyword in ['LTV1', 'LTV2', 'LTM1_1', 'LTM2_2']:
                    averaged_bpm[0].header.remove(keyword, ignore_missing=True)
                averaged_bpm[0].data[averaged_bpm[0].data > 0.1] = 1
                averaged_bpm[0].data = averaged_bpm[0].data.astype(np.uint8)
                averaged_bpm.writeto('bpm.{i}.fits'.format(i=i), overwrite=True)
