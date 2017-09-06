from astropy.io import ascii, fits
import numpy as np
from pyraf import iraf


def mad(d):
    return np.median(np.abs(np.median(d) - d))


def magtoflux(wave, mag, zp):
    # convert from ab mag to flambda
    # 3e-19 is lambda^2 / c in units of angstrom / Hz
    return zp * 10 ** (-0.4 * mag) / 3.33564095e-19 / wave / wave


def fluxtomag(flux):
    return -2.5 * np.log10(flux)


def boxcar_smooth(spec_wave, spec_flux, smoothwidth):
    # get the average wavelength separation for the observed spectrum
    # This will work best if the spectrum has equal linear wavelength spacings
    wavespace = np.diff(spec_wave).mean()
    # kw
    kw = int(smoothwidth / wavespace)
    # make sure the kernel width is odd
    if kw % 2 == 0:
        kw += 1
    kernel = np.ones(kw)
    # Conserve flux
    kernel /= kernel.sum()
    smoothed = spec_flux.copy()
    smoothed[(kw / 2):-(kw / 2)] = np.convolve(spec_flux, kernel, mode='valid')
    return smoothed


def get_binning(txt_filename, rawpath):
    with open(txt_filename) as f:
        lines = f.readlines()
    return fits.getval(rawpath + lines[0].rstrip(), 'CCDSUM', 1).replace(' ', 'x')


def fixpix(filename, maskname):
    # Run fixpix to interpolate over cosmic rays and bad pixels
    iraf.unlearn(iraf.fixpix)
    iraf.fixpix(filename, maskname, mode='h')


def convert_pixel_list_to_array(filename, nx, ny):
    data = ascii.read(filename)
    return data.reshape(ny, nx)
