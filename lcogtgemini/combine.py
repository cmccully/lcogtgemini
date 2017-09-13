import numpy as np
import lcogtgemini
from astropy.io import fits, ascii
from lcogtgemini import fits_utils

from pyraf import iraf

def find_bad_pixels(data, threshold=10.0):
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
    scales = []
    wavelengths = []
    for f in fs:
        hdu = fits.open(f)
        wavelengths.append(fits_utils.fitshdr_to_wave(hdu['SCI'].header))

    # Get the overlaps
    overlap_min_w = 0.0
    overlap_max_w = 100000.0
    min_w = 1000000.0
    max_w = 0.0
    wavelength_step = 1000.0
    for wavelength in wavelengths:
        overlap_min_w = max([overlap_min_w, np.min(wavelength)])
        overlap_max_w = min([overlap_max_w, np.max(wavelength)])
        min_w = min([min_w, np.min(wavelength)])
        max_w = max([max_w, np.max(wavelength)])
        wavelength_step = min([wavelength_step, wavelength[1] - wavelength[0]])

    min_w = np.max([min_w, lcogtgemini.bluecut])
    first_hdu = fits.open(fs[0])
    first_wavelengths = fits_utils.fitshdr_to_wave(first_hdu['SCI'].header)
    bad_pixels = find_bad_pixels(first_hdu['SCI'].data[0])
    first_hdu['SCI'].data[0][bad_pixels] = 0.0

    wavelength_grid = np.arange(min_w, max_w + wavelength_step, wavelength_step)
    data_to_combine = np.zeros((len(fs), wavelength_grid.shape[0]))
    for i, f in enumerate(fs):
        hdu = fits.open(f)
        wavelengths = fits_utils.fitshdr_to_wave(hdu['SCI'].header)
        overlap = np.logical_and(wavelengths >= overlap_min_w, wavelengths <= overlap_max_w)

        # Reject outliers
        bad_pixels = find_bad_pixels(hdu['SCI'].data[0], threshold=20)

        # Take the median of the ratio of each spectrum to the first to get the rescaling

        first_fluxes = np.interp(wavelengths[overlap], first_wavelengths, first_hdu['SCI'].data[0], left=0.0, right=0.0)
        good_pixels = np.ones(hdu['SCI'].data[0].shape[0], dtype=bool)
        good_pixels[overlap] = np.logical_and(hdu['SCI'].data[0][overlap] != 0.0, first_fluxes != 0.0)
        good_pixels = np.logical_and(good_pixels, ~(bad_pixels))

        scale = np.median(first_fluxes[good_pixels[overlap]] / hdu['SCI'].data[0][overlap][good_pixels[overlap]])
        scales.append(scale)
        hdu['SCI'].data[0][hdu['SCI'].data[0] == 0.0] = -1.e9
        hdu['SCI'].data[0][bad_pixels] = -1.e9
        data_to_combine[i] = np.interp(wavelength_grid, wavelengths, hdu['SCI'].data[0], left=0.0, right=0.0)
        data_to_combine[data_to_combine < 0.0] = 0.0
        data_to_combine[i] *= scale


    # write the scales into a file
    ascii.write({'scale': scales}, 'scales.dat', names=['scale'], format='fast_no_header')

    combined_data = data_to_combine.sum(axis=0)
    weights = (data_to_combine > 0).sum(axis=0)
    weights[weights == 0.0] = 1.0
    combined_data /= weights

    first_hdu[0].data = combined_data
    first_hdu[0].header['CRPIX1'] = 1
    first_hdu[0].header['CRVAL1'] = min_w
    first_hdu[0].header['CD1_1'] = wavelength_step
    first_hdu[0].header['CD2_2'] = 1
    first_hdu[0].header['CTYPE1'] = 'LINEAR  '
    first_hdu[0].header['CTYPE2'] = 'LINEAR  '
    first_hdu[0].header['WAT1_001'] = 'wtype=linear label=Wavelength units=angstroms'
    first_hdu[0].header['WAT0_001'] = 'system=equispec'
    first_hdu[0].header['WAT2_001'] = 'wtype=linear'
    first_hdu[0].header['APNUM1'] = hdu['SCI'].header['APNUM1']

    first_hdu[0].writeto(outfile)
    iraf.unlearn(iraf.scombine)

