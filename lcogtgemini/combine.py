import numpy as np
import lcogtgemini
from astropy.io import fits, ascii
from lcogtgemini import fits_utils
from lcogtgemini import file_utils
from astropy.convolution import convolve, Gaussian1DKernel
from lcogtgemini import utils


def find_bad_pixels(data, threshold=30.0):
    # Take the abs next pixel diff
    absdiff = np.abs(data[1:] - data[:-1])
    scaled_diff = absdiff / convolve(data, Gaussian1DKernel(stddev=20.0))[1:]
    # divide by square root of 2
    scatter = scaled_diff / np.sqrt(2.0)
    # Take the median
    # multply by 1.48 to go from mad to stdev
    scatter = np.median(scatter) * 1.48

    # Anything that is a 10 sigma outlier, set to 0
    deviant_pixels = scaled_diff > (threshold * scatter)
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
    max_w = np.min([max_w, lcogtgemini.redcut])
    first_hdu = fits.open(fs[0])
    first_wavelengths = fits_utils.fitshdr_to_wave(first_hdu['SCI'].header)
    first_fluxes = first_hdu['SCI'].data[0, 0]
    bad_pixels = find_bad_pixels(first_fluxes)
    first_fluxes[bad_pixels] = 0.0

    wavelength_grid = np.arange(min_w, max_w, wavelength_step)
    data_to_combine = np.zeros((len(fs), wavelength_grid.shape[0]))
    for i, f in enumerate(fs):
        hdu = fits.open(f)
        wavelengths = fits_utils.fitshdr_to_wave(hdu['SCI'].header)
        fluxes = hdu['SCI'].data[0, 0]
        in_chips = np.zeros(wavelengths.shape, dtype=bool)
        basename = file_utils.get_base_name(f)
        wavelengths_hdu = fits.open(basename + '.wavelengths.fits')
        chips = utils.get_wavelengths_of_chips(wavelengths_hdu)
        for chip in chips:
            in_chip = np.logical_and(wavelengths >= min(chip), wavelengths <= max(chip))
            in_chips[in_chip] = True

        fluxes[~in_chips] = 0.0
        overlap = np.logical_and(wavelengths >= overlap_min_w, wavelengths <= overlap_max_w)

        # Reject outliers
        bad_pixels = find_bad_pixels(fluxes)
        in_telluric = np.logical_and(wavelengths >= 6640.0, wavelengths <= 7040.0)
        in_telluric = np.logical_or(in_telluric, np.logical_and(wavelengths >= 7550.0, wavelengths <= 7750.0))
        bad_pixels[in_telluric] = False

        first_fluxes_interp = np.interp(wavelengths[overlap], first_wavelengths, first_fluxes, left=0.0, right=0.0)
        good_pixels = (fluxes[overlap] * first_fluxes_interp != 0.) & ~bad_pixels[overlap]

        # Take the median of the ratio of each spectrum to the first to get the rescaling
        scale = np.nanmedian(first_fluxes_interp[good_pixels] / fluxes[overlap][good_pixels])
        scales.append(scale)
        fluxes[fluxes == 0.] = np.nan
        fluxes[bad_pixels] = np.nan
        data_to_combine[i] = np.interp(wavelength_grid, wavelengths, fluxes, left=0.0, right=0.0)
        data_to_combine[i] *= scale

    # write the scales into a file
    ascii.write({'scale': scales}, 'scales.dat', names=['scale'], format='fast_no_header')

    first_hdu[0].data = np.nanmedian(data_to_combine, axis=0)
    first_hdu[0].header['CRPIX1'] = 1
    first_hdu[0].header['CRVAL1'] = min_w
    first_hdu[0].header['CD1_1'] = wavelength_step
    first_hdu[0].header['CD2_2'] = 1
    first_hdu[0].header['CTYPE1'] = 'LINEAR  '
    first_hdu[0].header['CTYPE2'] = 'LINEAR  '
    first_hdu[0].header['WAT1_001'] = 'wtype=linear label=Wavelength units=angstroms'
    first_hdu[0].header['WAT0_001'] = 'system=equispec'
    first_hdu[0].header['WAT2_001'] = 'wtype=linear'
    first_hdu[0].header['APNUM1'] = first_hdu['SCI'].header['APNUM1']

    first_hdu[0].writeto(outfile)
