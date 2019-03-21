import os

import numpy as np
from astropy.io import ascii
from astropy.io import fits
import lcogtgemini.file_utils
from lcogtgemini import combine
from lcogtgemini import fits_utils
from lcogtgemini import fitting

# Taken from the Berkley telluric correction
# telluricWaves = [(2000., 3190.), (3216., 3420.), (5500., 6050.), (6250., 6360.),
#                 (6450., 6530.), (6840., 7410.), (7550., 8410.), (8800., 9800.)]
# Taken from Moehler et al 2014 (on eg274)
# telluricWaves = [(5855., 5992.), (6261., 6349.), (6438., 6600.), (6821., 7094.),
#                 (7127., 7434.), (7562., 7731.), (7801., 8613.), (8798., 10338.)]
from lcogtgemini.file_utils import read_telluric_model

telluricWaves = [(5500., 6050.), (6250., 6360.), (6438., 6530.), (6821., 7434.), (7550., 8613.), (8798., 10338.)]


def telluric_correct(input_files):
    output_files = []
    for filename in input_files:

        # Get the standard to use for telluric correction
        hdu = fits.open(filename)
        waves = fits_utils.fitshdr_to_wave(hdu['SCI'].header)

        telluric_correction = ascii.read('telcor.dat')
        telwave = telluric_correction['col1']
        telspec = telluric_correction['col2']

        # Cross-correlate the standard star and the sci spectra
        # to find wavelength shift of standard star.
        w = np.logical_and(waves > 7500., waves < 7800.)
        tw = np.logical_and(telwave > 7500., telwave < 7800.)
        good_pixels = hdu['SCI'].data[0][w] != 0
        if good_pixels.sum() == 0:
            p = [1.0, 0.0]
        else:
            cleaned_data = np.interp(waves[w], waves[w][good_pixels], hdu['SCI'].data[0][w][good_pixels])
            p = fitting.fitxcor(waves[w], cleaned_data, telwave[tw], telspec[tw])
        # shift and stretch standard star spectrum to match science
        # spectrum.
        telcorr = np.interp(waves, p[0] * telwave + p[1], telspec, left=1.0, right=1.0)

        # Correct for airmass
        airmass = float(hdu[0].header['AIRMASS'])
        telcorr = telcorr ** (airmass ** 0.55)

        # Divide science spectrum by transformed standard star sub-spectrum
        hdu['SCI'].data[0] /= telcorr
        outfile = 't'+filename
        output_files.append(outfile)
        # Copy telluric-corrected data to new file.
        hdu.writeto(outfile)
    return output_files


def mktelluric(filename, objname):
    observed_hdu = fits.open(filename)
    observed_wavelengths = fits_utils.fitshdr_to_wave(observed_hdu[0].header)
    observed_data = observed_hdu[0].data

    maskname = observed_hdu[0].header['MASKNAME']
    telluric_model = read_telluric_model(maskname)
    # Read in the standard file
    standard_filename = lcogtgemini.file_utils.get_standard_file(objname)
    standard = lcogtgemini.file_utils.read_standard_file(standard_filename, maskname)

    good_pixels = observed_data > 0

    # Clip the edges of the detector where craziness happen.
    good_pixels[:20] = False
    good_pixels[-20:] = False

    bad_pixels = combine.find_bad_pixels(observed_data)
    in_telluric = np.logical_and(observed_wavelengths >= 6640.0, observed_wavelengths <= 7040.0)
    in_telluric = np.logical_or(in_telluric, np.logical_and(observed_wavelengths >= 7550.0, observed_wavelengths <= 7750.0))
    bad_pixels[in_telluric] = False
    good_pixels = np.logical_and(good_pixels, ~bad_pixels)

    standard_scale = np.median(np.interp(observed_wavelengths[good_pixels], standard['col1'], standard['col2']))
    standard['col2'] /= standard_scale
    # Interpolate the standard file onto the science wavelengths
    # Shift the standard star model by the same amount used in the sensitivity function
    # sensitivity
    best_fit = fit_standard(observed_wavelengths, observed_hdu[0].data,
                            telluric_model['col1'], telluric_model['col2'], standard['col1'], standard['col2'],
                            good_pixels)

    standard['col2'] *= standard_scale
    normalized_standard_wavelengths = (standard['col1'] - best_fit['xmin']) / best_fit['x_range']
    recaled_standard_wavelengths = best_fit['popt'][5] * (normalized_standard_wavelengths - best_fit['popt'][4])
    standard_wavelengths = recaled_standard_wavelengths * best_fit['x_range'] + best_fit['xmin']
    standard_flux = np.interp(observed_wavelengths, standard_wavelengths, best_fit['popt'][6] * standard['col2'])

    # In the telluric regions
    in_telluric = np.zeros(observed_wavelengths.shape, dtype=bool)
    for region in telluricWaves:
        in_region = np.logical_and(observed_wavelengths >= region[0], observed_wavelengths <= region[1])
        in_telluric[in_region] = True

    # Divide the observed by the standard
    # Fill the rest with ones
    correction = np.ones(observed_wavelengths.shape)

    in_telluric = np.logical_and(observed_hdu[0].data > 0, in_telluric)

    correction[in_telluric] = observed_hdu[0].data[in_telluric] / standard_flux[in_telluric]

    # Raise the whole telluric correction to the airmass ** -0.55 power
    # See matheson's paper. This normalizes things to airmass 1
    correction **= float(observed_hdu[0].header['AIRMASS']) ** -0.55
    ascii.write({'wavelengths': observed_wavelengths, 'telluric': correction}, 'telcor.dat',
                names=['wavelengths', 'telluric'], format='fast_no_header')


def telluric_correction_exists():
    return os.path.exists('telcor.dat')


def make_standard_model(telluric_waves, telluric_correction, std_waves, std_flux,
                        wavelength_min, wavelength_range):
    normalized_telluric_wavelengths = (telluric_waves - wavelength_min) / wavelength_range
    normalized_standard_wavelengths = (std_waves - wavelength_min) / wavelength_range

    def standard_mode(x, *p):
        # p 0, 1, 2 are for telluric fitting.
        # 0 and 1 linear wavelength shift and scale for telluric
        # 2 is power of telluric correction for the O2 A and B bands
        # 3 is the power of the telluric correction for the water bands (the rest of the telluric features)
        shifted_telluric_wavelengths = p[1] * (normalized_telluric_wavelengths - p[0])
        telluric_model = np.interp(x, shifted_telluric_wavelengths, telluric_correction,
                                   left=1.0, right=1.0)

        in_A = np.logical_and(x >= (6821. - wavelength_min) / wavelength_range, x <= (7094. - wavelength_min) / wavelength_range)
        in_B = np.logical_and(x <= (7731. - wavelength_min) / wavelength_range, x >= (7562. - wavelength_min) / wavelength_range)
        in_AB = np.logical_or(in_A, in_B)

        telluric_model[in_AB] **= (p[2] ** 0.55)
        telluric_model[~in_AB] **= (p[3] ** 0.55)
        # p 3, 4 are linear wavelength shift and scale for the standard model
        std_model = np.interp(x, p[5] * (normalized_standard_wavelengths - p[4]), std_flux)
        return p[6] * telluric_model * std_model

    return standard_mode


def fit_standard(wavelengths, data, telluric_waves, telluric_correction, std_waves, std_flux, good_pixels,
                 weight_scale=20.0):

    _, wavelength_min, wavelength_range = fitting.normalize_fitting_coordinate(wavelengths)
    function_to_fit = make_standard_model(telluric_waves, telluric_correction, std_waves, std_flux,
                                          wavelength_min, wavelength_range)

    def init_p0():
        p0 = np.zeros(7)
        p0[0] = 0.0
        p0[1] = 1.0
        p0[2] = 1.0
        p0[3] = 1.0
        p0[4] = 0.0
        p0[5] = 1.0
        p0[6] = 1.0
        return p0

    p0 = init_p0()
    errors = np.sqrt(np.abs(data) * 0.01)
    best_fit = fitting.run_fit(wavelengths, data, errors, function_to_fit, p0, weight_scale, good_pixels)
    # Go into a while loop
    while True:
        # Ask if the user is not happy with the fit,
        response = fitting.user_input('Does this fit look good? y or n:', ['y', 'n'], 'y')
        if response == 'y':
            break
        # If not have the user put in new values
        else:
            weight_scale = fitting.user_input('Scale for outlier rejection:', default=weight_scale, is_number=True)
            p0 = init_p0()
            function_to_fit = make_standard_model(telluric_waves, telluric_correction, std_waves,
                                                  std_flux, wavelength_min, wavelength_range)
            best_fit = fitting.run_fit(wavelengths, data, errors, function_to_fit, p0, weight_scale, good_pixels)

    return best_fit
