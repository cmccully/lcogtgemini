import numpy as np
import lcogtgemini
from astropy.io import fits, ascii
from lcogtgemini import fits_utils

from pyraf import iraf

def reject_bad_pixels(hdu, threshold=20.0):
    # Take the abs next pixel diff
    absdiff = np.abs(hdu[0].data[1:] - hdu[0].data[:-1])
    # divide by square root of 2
    scatter = absdiff / np.sqrt(2.0)
    # Take the median
    # multply by 1.48 to go from mad to stdev
    scatter = np.median(scatter) * 1.48

    # Anything that is a 10 sigma outlier, set to 0
    deviant_pixels = absdiff > (threshold * scatter)
    bad_pixels = np.zeros(scatter.shape, dtype=bool)
    bad_pixels[0:-1] = bad_pixels
    bad_pixels[1:] = np.logical_or(bad_pixels[1:], bad_pixels)

    hdu[0].data[bad_pixels] = 0.0
    return hdu


def speccombine(fs, outfile):
    hdu = fits.open(fs[0])
    first_hdu = reject_bad_pixels(hdu)
    first_wavelengths = fits_utils.fitshdr_to_wave(first_hdu[0].header)
    scales = []
    for f in fs:
        hdu = fits.open(f)
        # Reject outliers
        hdu = reject_bad_pixels(hdu)
        wavelengths = fits_utils.fitshdr_to_wave(hdu[0].header)
        # Take the median of the ratio of each spectrum to the first to get the rescaling
        fluxes = np.interp(first_wavelengths, wavelengths, hdu[0].data, left=0.0, right=0.0)
        good_pixels = np.logical_and(first_hdu[0].data != 0.0, fluxes != 0.0)
        scales.append(np.median(first_hdu[0].data[good_pixels] / fluxes[good_pixels]))
        hdu[0].data[hdu[0].data == 0.0] = -10000.0
        hdu.writeto('b' + f)

    spectra_to_combine = ['b' + f for f in fs]
    # write the best fit parameters into the headers of the files
    # Dump the list of spectra into a string that iraf can handle
    iraf_filelist = str(spectra_to_combine).replace('[', '').replace(']', '').replace("'", '')


    # write the scales into a file
    ascii.write({'scale': scales}, 'scales.dat', names=['scale'], format='fast_no_header')

    iraf.unlearn(iraf.scombine)
    iraf.scombine(iraf_filelist, outfile, scale='@scales.dat', mclip=True,
                  reject='avsigclip', lthreshold='-1.0', w1=lcogtgemini.bluecut)
