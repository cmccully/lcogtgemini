from astropy.io import ascii, fits
import numpy as np
from lcogtgemini import file_utils
import os
from scipy.signal import butter, lfilter


def mad(d):
    return np.median(np.abs(np.median(d) - d))


def magtoflux(wave, mag, zp):
    # convert from ab mag to flambda
    # 3e-19 is lambda^2 / c in units of angstrom / Hz
    return zp * 10 ** (-0.4 * mag) / 3.33564095e-19 / wave / wave


def fluxtomag(flux):
    return -2.5 * np.log10(flux)


def get_y_roi(txtfile, rawpath):
    images = file_utils.get_images_from_txt_file(txtfile)
    hdu = fits.open(os.path.join(rawpath, images[0]))
    return [int(i) for i in hdu[1].header['DETSEC'][1:-1].split(',')[1].split(':')]


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


def convert_pixel_list_to_array(filename, nx, ny):
    data = ascii.read(filename, format='fast_no_header')
    return data['col3'].reshape(ny, nx)


def rescale1e15(filename):
    hdu = fits.open(filename, mode='update')
    hdu[0].data *= 1e15
    hdu.flush()
    hdu.close()



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
