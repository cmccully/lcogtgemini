from lcogtgemini import fits_utils
from lcogtgemini import fitting
from lcogtgemini import flux_calibration
from astropy.io import fits
import numpy as np
import os
from astropy.io import ascii


# Taken from Moehler et al 2014 (on eg274)
telluricWaves = [(5855., 5992.), (6261., 6349.), (6438., 6600.), (6821., 7094.),
                 (7127., 7434.), (7562., 7731.), (7801., 8613.), (8798., 10338.)]

def telluric_correct(filename, outfile):

    # Get the standard to use for telluric correction
    stdfile = 'telcor.dat'

    hdu = fits.open(filename)
    spec = hdu[0].data.copy()
    hdr = hdu[0].header.copy()
    hdu.close()
    waves = fits_utils.fitshdr_to_wave(hdr)

    telwave, telspec = np.genfromtxt(stdfile).transpose()
    # Cross-correlate the standard star and the sci spectra
    # to find wavelength shift of standard star.
    w = np.logical_and(waves > 7550., waves < 8410.)
    tw = np.logical_and(telwave > 7550., telwave < 8410.)
    p = fitting.fitxcor(waves[w], spec[w], telwave[tw], telspec[tw])
    # shift and stretch standard star spectrum to match science
    # spectrum.
    telcorr = np.interp(waves, p[0] * telwave + p[1], telspec)

    # Correct for airmass
    airmass = float(hdr['AIRMASS'])
    telcorr = telcorr ** (airmass ** 0.55)

    # Divide science spectrum by transformed standard star sub-spectrum
    correct_spec = spec / telcorr

    # Copy telluric-corrected data to new file.
    fits_utils.tofits(outfile, correct_spec, hdr=hdr)


def mktelluric(filename, objname, base_stddir):
    # Read in the standard file
    standard = ascii.read(flux_calibration.get_standard_file(objname, base_stddir))
    observed_hdu = fits.open(filename)
    observed_wavelengths = fits_utils.fitshdr_to_wave(observed_hdu[0].header)

    # Interpolate the standard file onto the science wavelengths
    standard_flux = np.interp(observed_wavelengths, standard['col1'], standard['col2'])

    # In the telluric regions
    in_telluric = np.zeros(observed_wavelengths.shape, dtype=bool)
    for region in telluricWaves:
        in_region = np.logical_and(observed_wavelengths >= region[0], observed_wavelengths <= region[1])
        in_telluric[in_region] = True

    # Divide the observed by the standard
    # Fill the rest with ones
    correction = np.ones(observed_wavelengths.shape)
    correction[in_telluric] = observed_hdu[0].data[in_telluric] / standard_flux[in_telluric]

    # Raise the whole telluric correction to the airmass ** -0.55 power
    # See matheson's paper. This normalizes things to airmass 1
    correction **= float(observed_hdu[0].header['AIRMASS']) ** -0.55
    ascii.write({'wavelengths': observed_wavelengths, 'telluric': correction}, 'telcor.dat',
                names=['wavelengths', 'telluric'], format='fast_no_header')


def telluric_correction_exists():
    return os.path.exists('telcor.dat')
