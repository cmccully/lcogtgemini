import numpy as np
import lcogtgemini
from astropy.io import fits, ascii
from lcogtgemini import fits_utils

from pyraf import iraf

def find_bad_pixels(data, threshold=20.0):
    # Take the abs next pixel diff
    absdiff = np.abs(data[1:] - data[:-1])
    # divide by square root of 2
    scatter = absdiff / np.sqrt(2.0)
    # Take the median
    # multply by 1.48 to go from mad to stdev
    scatter = np.median(scatter) * 1.48

    # Anything that is a 10 sigma outlier, set to 0
    deviant_pixels = absdiff > (threshold * scatter)
    bad_pixels = np.zeros(data.shape, dtype=bool)
    bad_pixels[0:-1] = deviant_pixels
    bad_pixels[1:] = np.logical_or(bad_pixels[1:], deviant_pixels)
    return bad_pixels


def speccombine(fs, outfile):
    first_hdu = fits.open(fs[0])
    bad_pixels = find_bad_pixels(first_hdu['SCI'].data)
    first_hdu['SCI'].data[bad_pixels] = 0.0
    first_wavelengths = fits_utils.fitshdr_to_wave(first_hdu['SCI'].header)
    scales = []
    for f in fs:
        hdu = fits.open(f)
        # Reject outliers
        bad_pixels = find_bad_pixels(hdu['SCI'].data[0])
        wavelengths = fits_utils.fitshdr_to_wave(hdu['SCI'].header)[~bad_pixels]
        # Take the median of the ratio of each spectrum to the first to get the rescaling
        data = hdu['SCI'].data[0][~bad_pixels]
        fluxes = np.interp(first_wavelengths, wavelengths, data, left=0.0, right=0.0)
        good_pixels = np.logical_and(first_hdu['SCI'].data[0] != 0.0, fluxes != 0.0)
        scales.append(np.median(first_hdu['SCI'].data[0][good_pixels] / fluxes[good_pixels]))
        hdu['SCI'].data[0][hdu['SCI'].data[0] == 0.0] = -10000.0
        hdu.writeto('b' + f)

    spectra_to_combine = ['b' + f + '[SCI]' for f in fs]
    # write the best fit parameters into the headers of the files
    # Dump the list of spectra into a string that iraf can handle
    iraf_filelist = str(spectra_to_combine)[1:-1].replace("'", '')


    # write the scales into a file
    ascii.write({'scale': scales}, 'scales.dat', names=['scale'], format='fast_no_header')

    iraf.unlearn(iraf.scombine)
    iraf.scombine(iraf_filelist, outfile, scale='@scales.dat', mclip=True,
                  reject='avsigclip', lthreshold='-1.0', w1=lcogtgemini.bluecut)
