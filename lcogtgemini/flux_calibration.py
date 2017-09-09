from pyraf import iraf
import os
from lcogtgemini.file_utils import getredorblue
import numpy as np
from astropy.io import fits, ascii
from lcogtgemini import fits_utils
from astropy.convolution import convolve, Gaussian1DKernel
from lcogtgemini import fitting
from lcogtgemini import utils


def flux_calibrate(scifiles):
    for f in scifiles:
        redorblue = getredorblue(f)

        # Read in the sensitivity function (in magnitudes)
        sensitivity_hdu = fits.open('sens{redorblue}.fits'.format(redorblue=redorblue))
        sensitivity_wavelengths = fits_utils.fitshdr_to_wave(sensitivity_hdu['SCI'].header)

        # Interpolate the sensitivity onto the science wavelengths
        science_hdu = fits.open('xet'+ f.replace('.txt', '.fits'))
        science_wavelengths = fits_utils.fitshdr_to_wave(science_hdu['SCI'].header)
        sensitivity_correction = np.interp(science_wavelengths, sensitivity_wavelengths, sensitivity_hdu['SCI'].data)

        # Multiply the science spectrum by the corrections
        science_hdu['SCI'].data *= 10 ** (-0.4 * sensitivity_correction)
        science_hdu.writeto('cxet' + f[:-4] + '.fits')

        if os.path.exists('cxet' + f[:-4] + '.fits'):
            iraf.unlearn(iraf.splot)
            iraf.splot('cxet' + f.replace('.txt', '.fits') + '[sci]')  # just to check


def makesensfunc(scifiles, objname, base_stddir):
    for f in scifiles:
        # Find the standard star file
        standard_file = get_standard_file(objname, base_stddir)
        redorblue = getredorblue(f)
        # If this is a standard star, run standard
        # Standards will have an observation class of either progCal or partnerCal
        obsclass = fits.getval(f[:-4] + '.fits', 'OBSCLASS')
        if obsclass == 'progCal' or obsclass == 'partnerCal':
            specsens('xet' + f[:-4] + '.fits', 'sens' + redorblue + '.fits',
                     standard_file, float(fits.getval(f[:-4] + '.fits', 'EXPTIME')))


def specsens(specfile, outfile, stdfile, exptime=None,
             stdzp=3.68e-20, thresh=8, clobber=True):

    # Read in the reference star spectrum
    standard = ascii.read(stdfile, comment='#')

    # Read in the observed data
    observed_hdu = fits.open(specfile)
    observed_data = observed_hdu[2].data[0]
    observed_wavelengths = fits_utils.fitshdr_to_wave(observed_hdu[2].header)

    # Read in the telluric model
    telluric = ascii.read('telluric_model.dat')


    # Smooth the telluric spectrum to the size of the slit
    # I measured 5 angstroms FWHM for a 1 arcsecond slit
    # the 2.355 converts to sigma
    smoothing_scale = 5.0 * float(observed_hdu[0].header['MASKNAME'].split('arc')[0]) / 2.355

    telluric['col2'] = convolve(telluric['col2'], Gaussian1DKernel(stddev=smoothing_scale))

    # Interpolate the reference spectrum onto the observed wavelengths
    standard_fluxes = np.interp(observed_wavelengths, standard['col1'], standard['col2'])

    # ignored the chip gaps
    good_pixels = observed_data > 0

    telluric_correction = np.interp(observed_wavelengths[good_pixels], telluric['col1'], telluric['col2'])
    # Divide the reference by the science
    sensitivity_ratio = standard_fluxes[good_pixels] / observed_data[good_pixels]

    # Fit a combination of the telluric absorption multiplied by a constant + a polynomial-fourier model of
    # sensitivity
    best_fit, n_poly, n_fourier = fit_sensitivity(observed_wavelengths[good_pixels], sensitivity_ratio,
                                                  telluric_correction, 7, 21)

    # Strip out the telluric correction
    best_fit['popt'] = best_fit['popt'][1:]
    best_fit['model_function'] = fitting.polynomial_fourier_model(n_poly, n_fourier)
    # Save the sensitivity in magnitudes
    sensitivity = fitting.eval_fit(best_fit, observed_wavelengths) * float(observed_hdu[0].header['EXPTIME'])

    observed_hdu[2].data = utils.fluxtomag(sensitivity)
    observed_hdu[2].writeto(outfile)


def make_sensitivity_model(n_poly, n_fourier, telluric):
    poly_fourier_model = fitting.polynomial_fourier_model(n_poly, n_fourier)
    def sensitivity_model(x, *p):
        correction = np.ones(x.shape)
        correction[telluric < 1] = telluric[telluric < 1] ** (p[0] ** 0.55)
        return poly_fourier_model(x, *p[1:]) / correction
    return sensitivity_model

def fit_sensitivity(wavelengths, sensitivity_ratio, telluric_correction, n_poly, n_fourier, weight_scale=2.0):

    function_to_fit = make_sensitivity_model(n_poly, n_fourier, telluric_correction)
    p0 = np.zeros(2 + n_poly + 2 * n_fourier)
    p0[1] = 1.0
    p0[0] = 1.0

    best_fit = fitting.run_fit(wavelengths, sensitivity_ratio, 0.1 * np.abs(sensitivity_ratio),
                              function_to_fit, p0, weight_scale=weight_scale)
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
            best_fit = fitting.run_fit(wavelengths, sensitivity_ratio, 0.1 * np.abs(sensitivity_ratio),
                                       function_to_fit, p0, weight_scale)

    return best_fit, n_poly, n_fourier


def get_standard_file(objname, base_stddir):
    if os.path.exists(objname+'.std.dat'):
        standard_file = objname+'.std.dat'
    else:
        standard_file = os.path.join(iraf.osfn('gmisc$lib/onedstds/'), base_stddir, objname + '.dat')
    return standard_file
