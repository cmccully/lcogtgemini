from astropy.io import ascii, fits
from lcogtgemini import fits_utils
import numpy as np


def correct_for_extinction(scifiles, extfile):
    # Read in the extinction file
    extinction_correction = ascii.read(extfile)
    # Convert the extinction to flux
    extinction_correction['col2'] = 10**(-0.4 * extinction_correction['col2'])
    for f in scifiles:
        # read in the science file
        hdu = fits.open('et' + f.replace('.txt', '.fits'))
        airmass = float(hdu[0].header['AIRMASS'])
        wavelengths = fits_utils.fitshdr_to_wave(hdu['SCI'].header)
        # Linearly interpolate the extinction correction to the wavelengths of the spectrum
        corrections = np.interp(wavelengths, extinction_correction['col1'], extinction_correction['col2'])

        corrections *= 10 ** (-0.4 * airmass)
        # Divide the spectrum by the extinction correction
        hdu['SCI'].data /= corrections

        # Save the file to an extinction corrected file
        hdu.writeto('xet' + f.replace('.txt', '.fits'))
