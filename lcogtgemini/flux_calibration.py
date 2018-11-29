import os

import numpy as np
from astropy.io import fits, ascii
from pyraf import iraf

import lcogtgemini.file_utils
from lcogtgemini import combine
from lcogtgemini import fits_utils
from lcogtgemini import file_utils
from lcogtgemini import fitting
from lcogtgemini import utils
from lcogtgemini.file_utils import getredorblue, get_standard_file


def flux_calibrate(scifiles):
    for f in scifiles:
        redorblue = getredorblue(f)

        # Read in the sensitivity function (in magnitudes)
        sensitivity_hdu = fits.open('sens{redorblue}.fits'.format(redorblue=redorblue))
        sensitivity_wavelengths = fits_utils.fitshdr_to_wave(sensitivity_hdu['SCI'].header)

        # Interpolate the sensitivity onto the science wavelengths
        science_hdu = fits.open('xet' + f.replace('.txt', '.fits'))
        science_wavelengths = fits_utils.fitshdr_to_wave(science_hdu['SCI'].header)
        sensitivity_correction = np.interp(science_wavelengths, sensitivity_wavelengths, sensitivity_hdu['SCI'].data)

        # Multiply the science spectrum by the corrections
        science_hdu['SCI'].data *= 10 ** (-0.4 * sensitivity_correction)
        science_hdu['SCI'].data /= float(science_hdu[0].header['EXPTIME'])
        science_hdu.writeto('cxet' + f[:-4] + '.fits')

        if os.path.exists('cxet' + f[:-4] + '.fits'):
            iraf.unlearn(iraf.splot)
            iraf.splot('cxet' + f.replace('.txt', '.fits') + '[sci]')  # just to check


def makesensfunc(scifiles, objname, base_stddir):
    for f in scifiles:
        # Find the standard star file
        standard_file = get_standard_file(objname, base_stddir)
        redorblue = getredorblue(f)
        setupname = file_utils.getsetupname(f)
        # If this is a standard star, run standard
        # Standards will have an observation class of either progCal or partnerCal
        obsclass = fits.getval(f[:-4] + '.fits', 'OBSCLASS')
        if obsclass == 'progCal' or obsclass == 'partnerCal':
            wavelengths_filename = setupname + '.wavelengths.fits'
            specsens('xet' + f[:-4] + '.fits', 'sens' + redorblue + '.fits', standard_file, wavelengths_filename)


def specsens(specfile, outfile, stdfile, wavelengths_filename):

    # Read in the reference star spectrum
    standard = ascii.read(stdfile, comment='#')

    # Read in the observed data
    observed_hdu = fits.open(specfile)
    observed_data = observed_hdu[2].data[0]
    observed_wavelengths = fits_utils.fitshdr_to_wave(observed_hdu[2].header)

    telluric_model = lcogtgemini.file_utils.read_telluric_model(observed_hdu[0].header['MASKNAME'])
    # ignored the chip gaps
    good_pixels = observed_data > 0

    # Clip the edges of the detector where craziness happen.
    good_pixels[:10] = False
    good_pixels[-10:] = False

    bad_pixels = combine.find_bad_pixels(observed_data)
    in_telluric = np.logical_and(observed_wavelengths >= 6640.0, observed_wavelengths <= 7040.0)
    in_telluric = np.logical_or(in_telluric, np.logical_and(observed_wavelengths >= 7550.0, observed_wavelengths <= 7750.0))
    bad_pixels[in_telluric] = False
    good_pixels = np.logical_and(good_pixels, ~bad_pixels)

    # Fit a combination of the telluric absorption multiplied by a constant + a polynomial-fourier model of
    # sensitivity

    wavelengths_hdu = fits.open(wavelengths_filename)
    chips = utils.get_wavelengths_of_chips(wavelengths_hdu)
    need_to_interplolate = np.ones(observed_hdu[2].data[0].shape, dtype=bool)

    for chip in chips:
        in_chip = np.logical_and(observed_wavelengths >= min(chip), observed_wavelengths <= max(chip))
        standard_scale = np.median(np.interp(observed_wavelengths[in_chip], standard['col1'], standard['col2']))

        standard['col2'] /= standard_scale
        if good_pixels[in_chip].sum() > 32:  # 32 = 7 + 3 + 2 * 11 = number of parameters in default model
            best_fit, n_poly, n_fourier = fit_sensitivity(observed_wavelengths[in_chip], observed_data[in_chip],
                                                          telluric_model['col1'], telluric_model['col2'], standard['col1'], standard['col2'],
                                                          3, 11, float(observed_hdu['SCI'].header['RDNOISE']), good_pixels[in_chip])

            # Strip out the telluric correction
            best_fit['popt'] = best_fit['popt'][6:]
            best_fit['model_function'] = fitting.polynomial_fourier_model(n_poly, n_fourier)

            # Save the sensitivity in magnitudes
            sensitivity = standard_scale / fitting.eval_fit(best_fit, observed_wavelengths[in_chip]) * float(observed_hdu[0].header['EXPTIME'])
        else:
            sensitivity = np.ones(in_chip.sum())

        observed_hdu[2].data[0][in_chip] = utils.fluxtomag(sensitivity)
        need_to_interplolate[in_chip] = False
        standard['col2'] *= standard_scale

    observed_hdu[2].data = observed_hdu[2].data[0]
    observed_hdu[2].data[need_to_interplolate] = np.interp(observed_wavelengths[need_to_interplolate],
                                                           observed_wavelengths[~need_to_interplolate],
                                                           observed_hdu[2].data[~need_to_interplolate])
    observed_hdu[2].header += observed_hdu[0].header
    observed_hdu[2].header['OBSTYPE'] = 'SENS'
    observed_hdu[2].header['OBSCLASS'] = 'sensitivity'
    observed_hdu[2].writeto(outfile)


def make_sensitivity_model(n_poly, n_fourier, telluric_waves, telluric_correction, std_waves, std_flux,
                           wavelength_min, wavelength_range):
    poly_fourier_model = fitting.polynomial_fourier_model(n_poly, n_fourier)
    normalized_telluric_wavelengths = (telluric_waves - wavelength_min) / wavelength_range
    normalized_standard_wavelengths = (std_waves - wavelength_min) / wavelength_range

    def sensitivity_model(x, *p):
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
        return poly_fourier_model(x, *p[6:]) * telluric_model * std_model
    return sensitivity_model


def fit_sensitivity(wavelengths, data, telluric_waves, telluric_correction, std_waves, std_flux, n_poly,
                    n_fourier, readnoise, good_pixels, weight_scale=2.0):

    _, wavelength_min, wavelength_range = fitting.normalize_fitting_coordinate(wavelengths)
    function_to_fit = make_sensitivity_model(n_poly, n_fourier, telluric_waves, telluric_correction, std_waves, std_flux,
                                             wavelength_min, wavelength_range)

    def init_p0(n_poly, n_fourier):
        p0 = np.zeros(7 + n_poly + 2 * n_fourier)
        p0[0] = 0.0
        p0[1] = 1.0
        p0[2] = 1.0
        p0[3] = 0.1
        p0[4] = 0.0
        p0[5] = 1.0
        p0[6] = 1.0
        return p0

    p0 = init_p0(n_poly, n_fourier)
    errors = np.sqrt(np.abs(data) + readnoise ** 2.0)
    best_fit = fitting.run_fit(wavelengths, data, errors, function_to_fit, p0, weight_scale, good_pixels)

    # Go into a while loop
    while True:
        # Ask if the user is not happy with the fit,
        response = fitting.user_input('Does this fit look good? y or n:', ['y', 'n'], 'y')
        if response == 'y':
            break
        # If not have the user put in new values
        else:
            n_poly = int(fitting.user_input('Order of polynomial to fit:', [str(i) for i in range(100)], n_poly))
            n_fourier = int(fitting.user_input('Order of Fourier terms to fit:', [str(i) for i in range(100)], n_fourier))
            weight_scale = fitting.user_input('Scale for outlier rejection:', default=weight_scale, is_number=True)
            p0 = init_p0(n_poly, n_fourier)
            function_to_fit = make_sensitivity_model(n_poly, n_fourier, telluric_waves, telluric_correction, std_waves,
                                                     std_flux, wavelength_min, wavelength_range)
            best_fit = fitting.run_fit(wavelengths, data, errors, function_to_fit, p0, weight_scale, good_pixels)

    return best_fit, n_poly, n_fourier
