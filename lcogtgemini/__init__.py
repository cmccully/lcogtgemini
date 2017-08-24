#!/usr/bin/env python
'''
Created on Nov 7, 2014

@author: cmccully
'''
import os
from glob import glob
import numpy as np
from astroscrappy import detect_cosmics
from pyraf import iraf
from scipy import interpolate, ndimage, signal, optimize
import pf_model as pfm
import statsmodels as sm
from astropy.modeling import models, fitting
import astropy
from matplotlib import pyplot
from astropy.io import fits
from astropy.io.fits import PrimaryHDU, HDUList

iraf.cd(os.getcwd())
iraf.gemini()
iraf.gmos()
iraf.onedspec()

bluecut = 3450

iraf.gmos.logfile = "log.txt"
iraf.gmos.mode = 'h'
iraf.set(clobber='yes')

iraf.set(stdimage='imtgmos')

dooverscan = False
is_GS = False
do_qecorr = False
dobias = False


def normalize_fitting_coordinate(x):
    xrange = x.max() - x.min()
    return (x - x.min()) / xrange


class offset_left_model(astropy.modeling.Fittable1DModel):

    cutoff = astropy.modeling.Parameter(default=0)
    scale = astropy.modeling.Parameter(default=1)

    c0 = astropy.modeling.Parameter(default=1)
    c1 = astropy.modeling.Parameter(default=0)
    c2 = astropy.modeling.Parameter(default=0)
    c3 = astropy.modeling.Parameter(default=0)


    @staticmethod
    def evaluate(x, cutoff, scale, c0, c1, c2, c3):
        y = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3
        y[x <= cutoff] *= scale

        return y

class offset_right_model(astropy.modeling.Fittable1DModel):

    cutoff = astropy.modeling.Parameter(default=0)
    scale = astropy.modeling.Parameter(default=1)

    c0 = astropy.modeling.Parameter(default=1)
    c1 = astropy.modeling.Parameter(default=0)
    c2 = astropy.modeling.Parameter(default=0)
    c3 = astropy.modeling.Parameter(default=0)


    @staticmethod
    def evaluate(x, cutoff, scale, c0, c1, c2, c3):
        y = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3
        y[x >= cutoff] *= scale

        return y

class blackbody_model(astropy.modeling.Fittable1DModel):

    temperature = astropy.modeling.Parameter(default=10000)
    normalization = astropy.modeling.Parameter(default=1)

    @staticmethod
    def evaluate(x, temperature=10000., normalization=1.0):
        # Note x needs to be in microns and temperature needs to be in K
        flam = normalization * x ** -5
        flam /= np.exp(14387.7696 / x / temperature) - 1
        return flam


# Iterative reweighting linear least squares
def irls(x, data, errors, model, tol=1e-6, M=sm.robust.norms.AndrewWave(), maxiter=10):
    fitter = fitting.LevMarLSQFitter()

    if x is None:
        # Make x and y arrays out of the indicies
        x = np.indices(data.shape, dtype=np.float)

        if len(data.shape) == 2:
            y, x = x
        else:
            x = x[0]

        #Normalize to make fitting easier
        x = normalize_fitting_coordinate(x)
        if len(data.shape) == 2:
            y = normalize_fitting_coordinate(y)

    scatter = errors
    # Do an initial fit of the model
    # Use 1 / sigma^2 as weights
    weights = (errors ** -2.0).flatten()

    if len(data.shape) == 2:
        fitted_model = fitter(model, x, y, data, weights=weights)
    else:
        fitted_model = fitter(model, x, data, weights=weights)

    notconverged=True
    last_chi = np.inf
    iter = 0
    # Until converged
    while notconverged:
        # Update the weights
        if len(data.shape) == 2:
            residuals = data - fitted_model(x, y)
        else:
            residuals = data - fitted_model(x)
        # Save the chi^2 to check for convergence
        chi = ((residuals / scatter) ** 2.0).sum()

        # update the scaling (the MAD of the residuals)
        scatter = mad(residuals)  * 1.4826 # To convert to standard deviation
        weights = M.weights(residuals / scatter).flatten()

        # refit
        if len(data.shape) == 2:
            fitted_model = fitter(model, x, y, data, weights=weights)
        else:
            fitted_model = fitter(model, x, data, weights=weights)
        # converged when the change in the chi^2 (or l2 norm or whatever) is
        # less than the tolerance. Hopefully this should converge quickly.
        if iter >= maxiter or np.abs(chi - last_chi) < tol:
            notconverged = False
        else:
            last_chi = chi
            iter += 1

    return fitted_model

def sanitizeheader(hdr):
    # Remove the mandatory keywords from a header so it can be copied to a new
    # image.
    hdr = hdr.copy()

    # Let the new data decide what these values should be
    for i in ['SIMPLE', 'BITPIX', 'BSCALE', 'BZERO']:
        if i in hdr.keys():
            hdr.pop(i)

#    if hdr.has_key('NAXIS'):
    if 'NAXIS' in hdr.keys():
        naxis = hdr.pop('NAXIS')
        for i in range(naxis):
            hdr.pop('NAXIS%i' % (i + 1))

    return hdr


def tofits(filename, data, hdr=None, clobber=False):
    """simple pyfits wrapper to make saving fits files easier."""
    hdu = PrimaryHDU(data)
    if not (hdr is None):
        hdu.header += hdr
    hdulist = HDUList([hdu])
    hdulist.writeto(filename, overwrite=clobber, output_verify='ignore')


def mad(d):
    return np.median(np.abs(np.median(d) - d))


def magtoflux(wave, mag, zp):
    # convert from ab mag to flambda
    # 3e-19 is lambda^2 / c in units of angstrom / Hz
    return zp * 10 ** (-0.4 * mag) / 3.33564095e-19 / wave / wave


def fluxtomag(flux):
    return -2.5 * np.log10(flux)


def spectoascii(infilename, outfilename):
    hdu = fits.open(infilename)
    try:
        lam = fitshdr_to_wave(hdu['SCI'].header.copy())
        flux = hdu['SCI'].data.copy()
    except:
        lam = fitshdr_to_wave(hdu[0].header.copy())
        flux = hdu[0].data.copy()
    hdu.close()
    d = np.zeros((2, len(lam)))
    d[0] = lam
    d[1] = flux
    np.savetxt(outfilename, d.transpose())

def get_binning(txt_filename, rawpath):
    with open(txt_filename) as f:
        lines = f.readlines()
    return fits.getval(rawpath + lines[0].rstrip(), 'CCDSUM', 1).replace(' ', 'x')


def specsens(specfile, outfile, stdfile, extfile, airmass=None, exptime=None,
             stdzp=3.68e-20, thresh=8, clobber=True):

    # read in the specfile and create a spectrum object
    obs_hdu = fits.open(specfile)
    try:
        obs_flux = obs_hdu[2].data.copy()[0]
        obs_hdr = obs_hdu[2].header.copy()
    except:
        obs_flux = obs_hdu[0].data.copy()
        obs_hdr = obs_hdu[0].header.copy()
    obs_hdu.close()
    obs_wave = fitshdr_to_wave(obs_hdr)

    # Mask out everything below 3450 where there is no signal
    obs_flux = obs_flux[obs_wave >= bluecut]
    obs_wave = obs_wave[obs_wave >= bluecut]

    # Figure out where the chip gaps are
    chip_edges = get_chipedges(obs_flux)

    try:
        chip_gaps = np.ones(obs_flux.size, dtype=np.bool)
        for edge in chip_edges:
            chip_gaps[edge[0]: edge[1]] = False
    except:
        chip_gaps = np.zeros(obs_flux.size, dtype=np.bool)

    template_spectrum = signal.savgol_filter(obs_flux, 21, 3)
    noise = np.abs(obs_flux - template_spectrum)
    noise = ndimage.filters.gaussian_filter1d(noise, 100.0)

    if chip_gaps.sum() != len(chip_gaps):
        # Smooth the chip gaps
        intpr = interpolate.splrep(obs_wave[np.logical_not(chip_gaps)],
                                   obs_flux[np.logical_not(chip_gaps)],
                                   w=1 / noise[np.logical_not(chip_gaps)], k=2,
                                   s=20 * np.logical_not(chip_gaps).sum())
        obs_flux[chip_gaps] = interpolate.splev(obs_wave[chip_gaps], intpr)
    # smooth the observed spectrum
    # read in the std file and convert from magnitudes to fnu
    # then convert it to fwave (ergs/s/cm2/A)
    std_wave, std_mag, _stdbnd = np.genfromtxt(stdfile).transpose()
    std_flux = magtoflux(std_wave, std_mag, stdzp)

    # Get the typical bandpass of the standard star,
    std_bandpass = np.max([50.0, np.diff(std_wave).mean()])
    # Smooth the observed spectrum to that bandpass
    obs_flux = boxcar_smooth(obs_wave, obs_flux, std_bandpass)
    # read in the extinction file (leave in magnitudes)
    ext_wave, ext_mag = np.genfromtxt(extfile).transpose()

    # calculate the calibrated spectra
    cal_flux = cal_std(obs_wave, obs_flux, std_wave, std_flux, ext_wave,
                             ext_mag, airmass, exptime)

    # Normalize the fit variables so the fit is well behaved
    fitme_x = (obs_wave - obs_wave.min()) / (obs_wave.max() - obs_wave.min())
    fitme_y = cal_flux / np.median(cal_flux)
    coeffs = pfm.pffit(fitme_x, fitme_y, 5 , 7, robust=True,
                    M=sm.robust.norms.AndrewWave())

    fitted_flux = pfm.pfcalc(coeffs, fitme_x) * np.median(cal_flux)

    cal_mag = -1.0 * fluxtomag(fitted_flux)
    # write the spectra out
    cal_hdr = sanitizeheader(obs_hdr.copy())
    cal_hdr['OBJECT'] = 'Sensitivity function for all apertures'
    cal_hdr['CRVAL1'] = obs_wave.min()
    cal_hdr['CRPIX1'] = 1
    if do_qecorr:
        cal_hdr['QESTATE'] = True
    else:
        cal_hdr['QESTATE'] = False

    tofits(outfile, cal_mag, hdr=cal_hdr, clobber=True)


def hdr_pixel_range(x0, x1, y0, y1):
    return '[{0:d}:{1:d},{2:d}:{3:d}]'.format(x0, x1, y0, y1)

def get_x_pixel_range(keyword_value):
    """
    Get the x part of a section keyword
    :param keyword_value: Header keyword string
    :return: list xrange: 2 element list with start and end 1-indexed pixel values
    """
    # Strip off the brackets and split the coordinates
    pixel_sections = keyword_value[1:-1].split(',')
    return pixel_sections[0].split(':')


def cut_gs_image(filename, output_filename, pixel_range):
    """

    :param filename:
    :param output_filename:
    :param pixel_range: array-like, The range of pixels to keep, python indexed,
                        given in binned pixels
    :return:
    """
    hdu = fits.open(filename, unit16=True)
    for i in range(1, 13):
        ccdsum = hdu[i].header['CCDSUM']
        ccdsum = np.array(ccdsum.split(), dtype=np.int)

        y_ccdsec = [(pixel_range[0]  * ccdsum[1]) + 1,
                    (pixel_range[1]) * ccdsum[1]]

        x_detector_section = get_x_pixel_range(hdu[i].header['DETSEC'])
        hdu[i].header['DETSEC'] = hdr_pixel_range(int(x_detector_section[0]), int(x_detector_section[1]), y_ccdsec[0], y_ccdsec[1])

        x_ccd_section = get_x_pixel_range(hdu[i].header['CCDSEC'])
        hdu[i].header['CCDSEC'] = hdr_pixel_range(int(x_ccd_section[0]), int(x_ccd_section[1]), y_ccdsec[0], y_ccdsec[1])


        numpix = pixel_range[1] - pixel_range[0]

        x_bias_section = get_x_pixel_range(hdu[i].header['BIASSEC'])
        hdu[i].header['BIASSEC'] = hdr_pixel_range(int(x_bias_section[0]), int(x_bias_section[1]), 1, numpix)
        x_data_section = get_x_pixel_range(hdu[i].header['DATASEC'])
        hdu[i].header['DATASEC'] = hdr_pixel_range(int(x_data_section[0]), int(x_data_section[1]), 1, numpix)

        hdu[i].data = hdu[i].data[pixel_range[0]:pixel_range[1], :]

    hdu.writeto(output_filename)
    hdu.close()

def get_chipedges(data):
        # Get the x coordinages of all of the chip gap pixels
        # recall that pyfits opens images with coordinates y, x
        if len(data.shape) > 1:
            data = data[0]

        try:
            w = np.where(data == 0.0)[0]

            # Note we also grow the chip gap by 8 pixels on each side
            # Chip 1
            chip_edges = []
            left_chipedge = 10
            morechips = True
            while morechips:
                try:
                    right_chipedge = np.min(w[w > (left_chipedge + 25)]) - 10
                except:
                    right_chipedge = data.size - 10
                    morechips = False
                chip_edges.append((left_chipedge, right_chipedge))

                left_chipedge = np.max(w[w < (right_chipedge + 200)]) + 10
        except:
            chip_edges = []
        return chip_edges


def split1d(filename):

    hdu = fits.open(filename)
    chipedges = get_chipedges(hdu['SCI'].data[0])
    lam = fitshdr_to_wave(hdu['SCI'].header)
    # Copy each of the chips out seperately. Note that iraf is 1 indexed
    for i in range(3):
        # get the wavelengths that correspond to each chip
        w1 = lam[chipedges[i][0]]
        w2 = lam[chipedges[i][1]]
        iraf.scopy(filename+ '[SCI]', output=filename[:-5] + 'c%i.fits' % (i + 1), w1=w1,
                   w2=w2, rebin='no')
        hdu.close()


def mask_chipedges(filename):
    """
    Mask the edges of the chips with zeros to minimize artifacts.
    :param filename: Name of file that contains the spectrum
    :return:
    """
    hdu = fits.open(filename, mode='update')
    chip_edges = get_chipedges(hdu['SCI'].data[0])
    print(chip_edges)
    # Set the data = 0 in the chip gaps
    # Assume 3 chips for now.
    for i in range(2):
        hdu['SCI'].data[0, chip_edges[i][1]:chip_edges[i+1][0]] = 0.0

    hdu.flush()
    hdu.close()


def cal_std(obs_wave, obs_flux, std_wave, std_flux, ext_wave, ext_mag, airmass, exptime):
    """Given an observe spectra, calculate the calibration curve for the
       spectra.  All data is interpolated to the binning of the obs_spectra.
       The calibrated spectra is then calculated from
       C =  F_obs/ F_std / 10**(-0.4*A*E)/T/dW
       where F_obs is the observed flux from the source,  F_std  is the
       standard spectra, A is the airmass, E is the
       extinction in mags, T is the exposure time and dW is the bandpass
    """
    
    # re-interpt the std_spectra over the same wavelength
    std_flux = np.interp(obs_wave, std_wave, std_flux)

    # re-interp the ext_spetra over the same wavelength
    ext_mag = np.interp(obs_wave, ext_wave, ext_mag)
    
    # create the calibration spectra
    # set up the bandpass
    bandpass = np.diff(obs_wave).mean()

    # correct for extinction
    cal_flux = obs_flux / 10 ** (-0.4 * airmass * ext_mag)

    # correct for the exposure time and calculation the sensitivity curve
    cal_flux = cal_flux / exptime / bandpass / std_flux

    return cal_flux

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

# The last wavelength region was originally 9900. I bumped it down to 9800 to make
# sure we have an anchor point at the end of the spectrum.
telluricWaves = [(2000., 3190.), (3216., 3420.), (5500., 6050.), (6250., 6360.),
                 (6450., 6530.), (6840., 7410.), (7550., 8410.), (8800., 9800.)]
def combine_spec_chi2(p, lam, specs, specerrs):
    # specs should be an array with shape (nspec, nlam)
    nspec = specs.shape[0]
    # scale each spectrum by the given value
    # Assume 3 chips here
    scales = np.repeat(p, 3)

    scaledspec = (specs.transpose() * scales).transpose()
    scaled_spec_err = (specerrs.transpose() * scales).transpose()

    chi = 0.0
    # loop over each pair of spectra
    for i in range(nspec):
        for j in range(i + 1, nspec):
            # Calculate the chi^2 for places that overlap
            # (i.e. spec > 0 in both)
            w = np.logical_and(scaledspec[i] != 0.0, scaledspec[j] != 0)
            if w.sum() > 0:
                residuals = scaledspec[i][w] - scaledspec[j][w]
                errs2 = scaled_spec_err[i][w] ** 2.0
                errs2 += scaled_spec_err[j][w] ** 2.0
                chi += (residuals ** 2.0 / errs2).sum()
    return chi

def speccombine(fs, outfile):
    nsteps = 8001
    lamgrid = np.linspace(3000.0, 11000.0, nsteps)

    nfs = len(fs)
    # for each aperture
    # get all of the science images
    specs = np.zeros((nfs, nsteps))
    specerrs = np.zeros((nfs, nsteps))
    for i, f in enumerate(fs):
        hdu = fits.open(f)
        lam = fitshdr_to_wave(hdu[0].header.copy())

        # interpolate each spectrum onto a common wavelength scale

        specs[i] = np.interp(lamgrid, lam, hdu[0].data,
                             left=0.0, right=0.0)
        # Also calculate the errors. Right now we assume that the variances
        # interpolate linearly. This is not strictly correct but it should be
        # close. Also we don't include terms in the variance for the
        # uncertainty in the wavelength solution.
        specerrs[i] = 0.1 * specs[i]

    # minimize the chi^2 given free parameters are multiplicative factors
    # We could use linear or quadratic, but for now assume constant
    # Assume 3 chips for now
    p0 = np.ones(nfs / 3)

    results = optimize.minimize(combine_spec_chi2, p0,
                                args=(lamgrid, specs, specerrs),
                                method='Nelder-Mead',
                                options={'maxfev': 1e5, 'maxiter': 1e5, 'ftol':1e-5})

    # write the best fit parameters into the headers of the files
    # Dump the list of spectra into a string that iraf can handle
    iraf_filelist = str(fs).replace('[', '').replace(']', '').replace("'", '') #.replace(',', '[SCI],')
    #iraf_filelist += '[SCI]'

    # write the best fit results into a file
    lines = []

    for p in np.repeat(results['x'], 3):
        lines.append('%f\n' % (1.0 / p))
    f = open('scales.dat', 'w')
    f.writelines(lines)
    f.close()
    # run scombine after multiplying the spectra by the best fit parameters
    if os.path.exists(outfile):
        os.remove(outfile)
    iraf.unlearn(iraf.scombine)
    iraf.scombine(iraf_filelist, outfile, scale='@scales.dat',
                  reject='avsigclip', lthreshold='INDEF', w1=bluecut)

def fitshdr_to_wave(hdr):
    crval = float(hdr['CRVAL1'])
    crpix = float(hdr['CRPIX1'])
    # Convert crpix to be zero indexed
    crpix -= 1
    if 'CDELT1' in hdr.keys():
        cdelt = float(hdr['CDELT1'])
    else:
        cdelt = float(hdr['CD1_1'])
    npix = float(hdr['NAXIS1'])
    lam = np.arange(crval - cdelt * crpix ,
                    crval + cdelt * (npix - crpix) - 1e-4,
                    cdelt)
    return lam

def telluric_mask(waves):
    # True where not telluric contaminated
    not_telluric = np.ones(waves.shape, dtype=np.bool)
    for wavereg in telluricWaves:
        in_telluric_region = np.logical_and(waves >= wavereg[0],
                                            waves <= wavereg[1])
        not_telluric = np.logical_and(not_telluric,
                                         np.logical_not(in_telluric_region))
    return not_telluric

def mktelluric(filename):
    #TODO Try fitting a black body instead of interpolating.
    # if it is a standard star combined file
    # read in the spectrum and calculate the wavelengths of the pixels
    hdu = fits.open(filename)
    spec = hdu[0].data.copy()
    hdr = hdu[0].header.copy()
    hdu.close()
    waves = fitshdr_to_wave(hdr)

    # Start by interpolating over the chip gaps
    chip_edges = get_chipedges(spec)
    chip_gaps = np.ones(spec.size, dtype=np.bool)
    for edge in chip_edges:
        chip_gaps[edge[0]: edge[1]] = False

    template_spectrum = signal.savgol_filter(spec, 21, 3)
    noise = np.abs(spec - template_spectrum)
    noise = ndimage.filters.gaussian_filter1d(noise, 100.0)

    # Smooth the chip gaps
    intpr = interpolate.splrep(waves[np.logical_not(chip_gaps)],
                               spec[np.logical_not(chip_gaps)],
                               w=1 / noise[np.logical_not(chip_gaps)], k=2,
                               s=10 * np.logical_not(chip_gaps).sum())
    spec[chip_gaps] = interpolate.splev(waves[chip_gaps], intpr)

    not_telluric = telluric_mask(waves)
    # Smooth the spectrum so that the spline doesn't go as crazy
    # Use the Savitzky-Golay filter to presevere the edges of the
    # absorption features (both atomospheric and intrinsic to the star)
    sgspec = signal.savgol_filter(spec, 31, 3)
    # Get the number of data points to set the smoothing criteria for the 
    # spline
    m = not_telluric.sum()
    intpr = interpolate.splrep(waves[not_telluric], sgspec[not_telluric],
                               w=1 / noise[not_telluric], k=2, s=20 * m)

    # Replace the telluric with the smoothed function
    smoothedspec = interpolate.splev(waves, intpr)

    # Extrapolate the ends linearly
    # Blue side
    w = np.logical_and(waves > 3420, waves < 3600)
    bluefit = np.poly1d(np.polyfit(waves[w], spec[w], 1))
    bluewaves = waves < 3420
    smoothedspec[bluewaves] = bluefit(waves[bluewaves])
     
    # Red side
    w = np.logical_and(waves > 8410, waves < 8800)
    redfit = np.poly1d(np.polyfit(waves[w], spec[w], 1))
    redwaves = waves > 8800
    smoothedspec[redwaves] = redfit(waves[redwaves])
    smoothedspec[not_telluric] = spec[not_telluric]
    # Divide the original and the telluric corrected spectra to
    # get the correction factor
    correction = spec / smoothedspec

    airmass = float(hdr['AIRMASS'])
    correction = correction ** (airmass ** -0.55)
    # Save the correction
    dout = np.ones((2, len(waves)))
    dout[0] = waves
    dout[1] = correction
    np.savetxt('telcor.dat', dout.transpose())


def telluric(filename, outfile):

    # Get the standard to use for telluric correction
    stdfile = 'telcor.dat'
    
    hdu = fits.open(filename)
    spec = hdu[0].data.copy()
    hdr = hdu[0].header.copy()
    hdu.close()
    waves = fitshdr_to_wave(hdr)
    
    telwave, telspec = np.genfromtxt(stdfile).transpose()
    # Cross-correlate the standard star and the sci spectra
    # to find wavelength shift of standard star.
    w = np.logical_and(waves > 7550., waves < 8410.)
    tw = np.logical_and(telwave > 7550., telwave < 8410.)
    p = fitxcor(waves[w], spec[w], telwave[tw], telspec[tw])
    # shift and stretch standard star spectrum to match science
    # spectrum.
    telcorr = np.interp(waves, p[0] * telwave + p[1], telspec)

    # Correct for airmass
    airmass = float(hdr['AIRMASS'])
    telcorr = telcorr ** (airmass ** 0.55)

    # Divide science spectrum by transformed standard star sub-spectrum
    correct_spec = spec / telcorr

    # Copy telluric-corrected data to new file.
    tofits(outfile, correct_spec, hdr=hdr)

def ncor(x, y):
    """Calculate the normalized correlation of two arrays"""
    d = np.correlate(x, x) * np.correlate(y, y)

    return np.correlate(x, y) / d ** 0.5

def xcorfun(p, warr, farr, telwarr, telfarr):
    # Telluric wavelengths and flux
    # Observed wavelengths and flux
    # resample the telluric spectrum at the same wavelengths as the observed
    # spectrum
    # Make the artifical spectrum to cross correlate
    asfarr = np.interp(warr, p[0] * telwarr + p[1], telfarr, left=1.0, right=1.0)
    return np.abs(1.0 / ncor(farr, asfarr))

def fitxcor(warr, farr, telwarr, telfarr):
    """Maximize the normalized cross correlation coefficient for the telluric
    correction
    """
    res = optimize.minimize(xcorfun, [1.0, 0.0], method='Nelder-Mead',
                   args=(warr, farr, telwarr, telfarr))

    return res['x']

def sort():
    if not os.path.exists('raw'):
        iraf.mkdir('raw')
    fs = glob('*.fits')
    for f in fs:
        iraf.mv(f, 'raw/')
    
    sensfs = glob('raw/sens*.fits')
    if len(sensfs) != 0:
        for f in sensfs:
            iraf.mv(f, './')
    # Make a reduction directory
    if not os.path.exists('work'):
        iraf.mkdir('work')

    sensfs = glob('sens*.fits')
    if len(sensfs) != 0:
        for f in sensfs:
            iraf.cp(f, 'work/')
    
    if os.path.exists('telcor.dat'):
        iraf.cp('telcor.dat', 'work/')
    
    if os.path.exists('raw/bias.fits'):
        iraf.cp('raw/bias.fits', 'work/')

    fs = glob('raw/*.qe.fits')
    if len(fs) > 0:
        for f in fs:
            iraf.cp(f, 'work/')
    
    # make a list of the raw files
    fs = glob('raw/*.fits')
    # Add a ../ in front of all of the file names
    for i in range(len(fs)):
        fs[i] = '../' + fs[i]
    return np.array(fs)

def init_northsouth(fs, topdir, rawpath):
    # Get the correct directory for the standard star
    base_stddir = 'spec50cal/'
    extfile = iraf.osfn('gmisc$lib/onedstds/kpnoextinct.dat') 
    observatory = 'Gemini-North'

    global is_GS
    is_GS = fits.getval(fs[0], 'OBSERVAT') == 'Gemini-South'
    if 'Hamamatsu' in fits.getval(fs[0], 'DETECTOR'):
        global dooverscan
        dooverscan = True
        global do_qecorr
        do_qecorr = True
    if is_GS:
        if not os.path.exists(topdir + '/raw_fixhdr'):
            iraf.mkdir(topdir + '/raw_fixhdr')
        rawpath = '%s/raw_fixhdr/' % topdir
        os.system('gmoss_fix_headers.py --files="%s/raw/*.fits" --destination=%s' % (topdir, rawpath))
        base_stddir = 'ctionewcal/'
        observatory = 'Gemini-South'
        extfile = iraf.osfn('gmisc$lib/onedstds/ctioextinct.dat') 
    return extfile, observatory, base_stddir, rawpath


def getobstypes(fs):
    # get the type of observation for each file
    obstypes = []
    obsclasses = []
    for f in fs: 
        obstypes.append(fits.getval(f, 'OBSTYPE', ext=0))
        obsclasses.append(fits.getval(f, 'OBSCLASS', ext=0))
        
    obstypes = np.array(obstypes)
    obsclasses = np.array(obsclasses)
    return obstypes, obsclasses


def makebias(fs, obstypes, rawpath):
    for f in fs:
        if f[-10:] == '_bias.fits':
            iraf.cp(f, 'bias.fits')
        elif 'bias' in f:
            iraf.cp(f, './')

    if len(glob('bias*.fits')) == 0:
        bias_files = fs[obstypes == 'BIAS']
        binnings = [fits.getval(f, 'CCDSUM', 1).replace(' ', 'x') for f in bias_files]
        for binning in list(set(binnings)):
            # Make the master bias
            biastxtfile = open('bias{binning}.txt'.format(binning=binning), 'w')
            biasfs = bias_files[np.array(binnings) == binning]
            for f in biasfs:
                biastxtfile.writelines(f.split('/')[-1] + '\n')
            biastxtfile.close()
            iraf.gbias('@%s/bias{binning}.txt'.format(binning=binning) % os.getcwd(),
                       'bias{binning}'.format(binning=binning), rawpath=rawpath, fl_over=dooverscan,
                       fl_vardq=True)


def getobjname(fs, obstypes):
    objname = fits.getval(fs[obstypes == 'OBJECT'][0], 'OBJECT', ext=0).lower()
    
    # get rid of nonsense in the name (like the plus and whitespace
    objname = objname.replace('+', '')
    objname = ''.join(objname.split())

    # replace ltt with just l
    objname = objname.replace('ltt', 'l')
    return objname


def maketxtfiles(fs, obstypes, obsclasses, objname):
    # go through each of the files (Ignore bias and aquisition files)
    goodfiles = np.logical_and(obsclasses != 'acqCal', obsclasses != 'acq')
    goodfiles = np.logical_and(goodfiles, obstypes != 'BIAS')
    
    for f in fs[goodfiles]:
        # put the filename in the correct text file.
        obsstr = ''
        obstype = fits.getval(f, 'OBSTYPE', ext=0)
        if obstype != 'OBJECT':
            obsstr = '.' + obstype.lower()
            expnum = ''
        else:
            expnum = 1
        
        # Drop the raw/
        fname = f.split('/')[-1]
        # red or blue setting
        redblue = fits.getval(f, 'GRATING')[0].lower()
        # central wavelength
        lamcentral = fits.getval(f, 'CENTWAVE')

        txtname = '%s.%s%s%i%s.txt' % (objname, str(expnum), redblue, lamcentral, obsstr)
        # If more than one arc or flat, append to the text file 
        if os.path.exists(txtname):
            if obsstr == '.flat' or obsstr == '.arc':
                # write to a text file
                txtfile = open(txtname, 'a')
            else:
                # We need to increment the exposure number
                moreimages = True
                expnum += 1
                while moreimages:
                    txtname = '%s.%s%s%i%s.txt' % (objname, str(expnum), redblue, lamcentral, obsstr)
                    if not os.path.exists(txtname):
                        txtfile = open(txtname, 'w')
                        moreimages = False
                    else:
                        expnum += 1
        else:
            txtfile = open(txtname, 'w')
        
        txtfile.write(fname + '\n')
        txtfile.close()


def gettxtfiles(fs, objname):
       
    flatfiles = np.array(glob('*.flat.txt'))
    
    # reduce the CuAr arcfiles.  Not flat fielded, gaps are not fixpixed
    arcfiles = np.array(glob('*.arc.txt'))

    # reduce the science files
    scifiles = glob(objname + '*.txt')
    
    nonscifiles = []
    # remove the arcs and flats
    for f in scifiles:
        if 'arc' in f or 'flat' in f: nonscifiles.append(f)
        
    for f in nonscifiles: 
        scifiles.remove(f)
    scifiles = np.array(scifiles)

    return flatfiles, arcfiles, scifiles


def makemasterflat(flatfiles, rawpath, plot=True):
    # normalize the flat fields
    for f in flatfiles:
        binning = get_binning(f, rawpath)
        # Use IRAF to get put the data in the right format and subtract the
        # bias
        # This will currently break if multiple flats are used for a single setting
        iraf.unlearn(iraf.gsreduce)
        if dobias:
            biasfile = "bias{binning}".format(binning=binning)
        else:
            biasfile = ''
        iraf.gsreduce('@' + f, outimages = f[:-4]+'.mef.fits',rawpath=rawpath, fl_bias=dobias,
                      bias=biasfile, fl_over=dooverscan, fl_flat=False, fl_gmosaic=False,
                      fl_fixpix=False, fl_gsappwave=False, fl_cut=False, fl_title=False,
                      fl_oversize=False, fl_vardq=True)

        if do_qecorr:
            # Renormalize the chips to remove the discrete jump in the
            # sensitivity due to differences in the QE for different chips
            iraf.unlearn(iraf.gqecorr)

            iraf.gqecorr(f[:-4]+'.mef', outimages=f[:-4]+'.qe.fits', fl_keep=True, fl_correct=True,
                         refimages=f[:-4].replace('flat', 'arc.arc.fits'),
                         corrimages=f[:-9] +'.qe.fits', verbose=True, fl_vardq=True)

            iraf.unlearn(iraf.gmosaic)
            iraf.gmosaic(f[:-4]+'.qe.fits', outimages=f[:-4]+'.mos.fits', fl_vardq=True, fl_clean=False)
        else:
            iraf.unlearn(iraf.gmosaic)
            iraf.gmosaic(f[:-4]+'.mef.fits', outimages=f[:-4]+'.mos.fits', fl_vardq=True, fl_clean=False)

        flat_hdu = fits.open(f[:-4] + '.mos.fits')

        data = np.median(flat_hdu['SCI'].data, axis=0)
        chip_edges = get_chipedges(data)

        x = np.arange(len(data), dtype=np.float)
        x /= x.max()

        y = data / np.median(data)

        fitme_x = x[chip_edges[0][0]:chip_edges[0][1]]
        fitme_x = np.append(fitme_x, x[chip_edges[1][0]:chip_edges[1][1]])
        fitme_x = np.append(fitme_x, x[chip_edges[2][0]:chip_edges[2][1]])

        fitme_y = y[chip_edges[0][0]:chip_edges[0][1]]
        fitme_y = np.append(fitme_y, y[chip_edges[1][0]:chip_edges[1][1]])
        fitme_y = np.append(fitme_y, y[chip_edges[2][0]:chip_edges[2][1]])

        fit = pfm.pffit(fitme_x, fitme_y, 15, 7, robust=True,
                    M=sm.robust.norms.AndrewWave())
        if plot:
            pyplot.ion()
            pyplot.clf()
            pyplot.plot(x, y)
            pyplot.plot(x, pfm.pfcalc(fit, x))
            _junk = raw_input('Press enter to continue')
        flat_hdu['SCI'].data /= pfm.pfcalc(fit, x) * np.median(data)
        flat_hdu.writeto(f[:-4] + '.fits')


def wavesol(arcfiles, rawpath):
    for f in arcfiles:
        binning = get_binning(f, rawpath)
        iraf.unlearn(iraf.gsreduce)
        if dobias:
            bias_filename = "bias{binning}".format(binning=binning)
        else:
            bias_filename = ''
        iraf.gsreduce('@' + f, outimages=f[:-4], rawpath=rawpath,
                      fl_flat=False, bias=bias_filename, fl_bias=dobias,
                      fl_fixpix=False, fl_over=dooverscan, fl_cut=False, fl_gmosaic=True,
                      fl_gsappwave=True, fl_oversize=False, fl_vardq=True)


        # determine wavelength calibration -- 1d and 2d
        iraf.unlearn(iraf.gswavelength)
        iraf.gswavelength(f[:-4], fl_inter='yes', fl_addfeat=False, fwidth=15.0, low_reject=2.0,
                          high_reject=2.0, step=10, nsum=10, gsigma=2.0,
                          cradius=16.0, match=-6, order=7, fitcxord=7,
                          fitcyord=7)

        if do_qecorr:
            # Make an extra random copy so that gqecorr works. Stupid Gemini.
            iraf.cp(f[:-4]+'.fits', f[:-4]+'.arc.fits')
        # transform the CuAr spectrum, for checking that the transformation is OK
        # output spectrum has prefix t
        iraf.unlearn(iraf.gstransform)
        iraf.gstransform(f[:-4], wavtran=f[:-4])

def make_qecorrection(arcfiles):
    for f in arcfiles:
        if do_qecorr:
            #read in the arcfile name
            with open(f) as txtfile:
                arcimage = txtfile.readline()
                # Strip off the newline character
                arcimage = 'g' + arcimage.split('\n')[0]
            if not os.path.exists(f[:-8] +'.qe.fits'):
                iraf.gqecorr(arcimage, refimages=f[:-4]+'.arc.fits', fl_correct=False, fl_keep=True,
                             corrimages=f[:-8] +'.qe.fits', verbose=True, fl_vardq=True)

def getsetupname(f):
    # Get the setup base name by removing the exposure number
    return f.split('.')[0] + '.' + f.split('.')[1][1:]

def getredorblue(f):
    return f.split('.')[1][1]

def scireduce(scifiles, rawpath):
    for f in scifiles:
        binning = get_binning(f, rawpath)
        setupname = getsetupname(f)
        if dobias:
            bias_filename = "bias{binning}".format(binning=binning)
        else:
            bias_filename = ''
        # gsreduce subtracts bias and mosaics detectors
        iraf.unlearn(iraf.gsreduce)
        iraf.gsreduce('@' + f, outimages=f[:-4]+'.mef', rawpath=rawpath, bias=bias_filename, fl_bias=dobias,
                      fl_over=dooverscan, fl_fixpix='no', fl_flat=False, fl_gmosaic=False, fl_cut=False,
                      fl_gsappwave=False, fl_oversize=False, fl_vardq=True)

        if do_qecorr:
            # Renormalize the chips to remove the discrete jump in the
            # sensitivity due to differences in the QE for different chips
            iraf.unlearn(iraf.gqecorr)
            iraf.gqecorr(f[:-4]+'.mef', outimages=f[:-4]+'.qe.fits', fl_keep=True, fl_correct=True, fl_vardq=True,
                         refimages=setupname + '.arc.arc.fits', corrimages=setupname +'.qe.fits', verbose=True)

            iraf.unlearn(iraf.gmosaic)
            iraf.gmosaic(f[:-4]+'.qe.fits', outimages=f[:-4] +'.fits', fl_vardq=True, fl_clean=False)
        else:
            iraf.unlearn(iraf.gmosaic)
            iraf.gmosaic(f[:-4]+'.mef.fits', outimages=f[:-4] +'.fits', fl_vardq=True, fl_clean=False)

        # Flat field the image
        hdu = fits.open(f[:-4]+'.fits', mode='update')
        hdu['SCI'].data /= fits.getdata(setupname+'.flat.fits', extname='SCI')
        hdu.flush()
        hdu.close()

        # Transform the data based on the arc  wavelength solution
        iraf.unlearn(iraf.gstransform)
        iraf.gstransform(f[:-4], wavtran=setupname + '.arc', fl_vardq=True)


def skysub(scifiles, rawpath):
    for f in scifiles:
        # sky subtraction
        # output has an s prefixed on the front
        # This step is currently quite slow for Gemini-South data
        iraf.unlearn(iraf.gsskysub)
        iraf.gsskysub('t' + f[:-4], long_sample='*', fl_inter='no', fl_vardq=True,
                      naverage=-10, order=1, low_reject=2.0, high_reject=2.0,
                      niterate=10, mode='h')
    

def crreject(scifiles):
    for f in scifiles:
        # run lacosmicx
        hdu = fits.open('st' + f.replace('.txt', '.fits'))

        readnoise = 3.5
        # figure out what pssl should be approximately
        d = hdu[2].data.copy()
        dsort = np.sort(d.ravel())
        nd = dsort.shape[0]
        # Calculate the difference between the 16th and 84th percentiles to be 
        # robust against outliers
        dsig = (dsort[int(round(0.84 * nd))] - dsort[int(round(0.16 * nd))]) / 2.0
        pssl = (dsig * dsig - readnoise * readnoise)

        mask = d == 0.0
        mask = np.logical_or(mask,  hdu['DQ'].data)

        crmask, _cleanarr = detect_cosmics(d, inmask=mask, sigclip=4.0,
                                                objlim=1.0, sigfrac=0.05, gain=1.0,
                                                readnoise=readnoise, pssl=pssl)
        
        tofits(f[:-4] + '.lamask.fits', np.array(crmask, dtype=np.uint8), hdr=hdu['SCI'].header.copy())

def fixpix(scifiles):
    # Run fixpix to interpolate over cosmic rays
    for f in scifiles:
        # run fixpix
        iraf.unlearn(iraf.fixpix)
        iraf.fixpix('t' + f[:-4] + '.fits[2]', f[:-4] + '.lamask.fits', mode='h')
        
def extract(scifiles):
    for f in scifiles:    
        iraf.unlearn(iraf.gsextract)
        # Extract the specctrum
        iraf.gsextract('t' + f[:-4], fl_inter='yes', bfunction='legendre', fl_vardq=True,
                       border=2, bnaverage=-3, bniterate=2, blow_reject=2.0,
                       bhigh_reject=2.0, long_bsample='-100:-40,40:100',
                       background='fit', weights='variance',
                       lsigma=3.0, usigma=3.0, tnsum=50, tstep=50, mode='h')

        # Trim off below the blue side cut
        hdu = fits.open('et' + f[:-4] +'.fits', mode='update')
        lam = fitshdr_to_wave(hdu['SCI'].header)
        w = lam > bluecut
        trimmed_data =np.zeros((1, w.sum()))
        trimmed_data[0] = hdu['SCI'].data[0, w]
        hdu['SCI'].data = trimmed_data
        hdu['SCI'].header['NAXIS1'] = w.sum()
        hdu['SCI'].header['CRPIX1'] = 1
        hdu['SCI'].header['CRVAL1'] = lam[w][0]
        hdu.flush()

        hdu.close()


def rescale_chips(scifiles):
    for f in scifiles:
        hdu = fits.open('et'+ f[:-4]+'.fits', mode='update')
        chip_edges = get_chipedges(hdu['SCI'].data)
        lam = fitshdr_to_wave(hdu['SCI'].header)
        x = (lam - lam.min())/ (lam.max() - lam.min())
        y = hdu['SCI'].data[0] / np.median(hdu['SCI'].data[0])

        # Fit the left chip
        left_x = x[chip_edges[0][1] - 200: chip_edges[0][1]]
        left_x = np.append(left_x, x[chip_edges[1][0]: chip_edges[1][0] + 200])

        left_data = y[chip_edges[0][1] - 200: chip_edges[0][1]]
        left_data = np.append(left_data , y[chip_edges[1][0]: chip_edges[1][0] + 200])

        left_errors = float(hdu['SCI'].header['RDNOISE']) ** 2.0
        left_errors += left_data * np.median(hdu['SCI'].data[0])
        left_errors **= 0.5
        left_errors /= np.median(hdu['SCI'].data[0])

        left_model = offset_left_model(cutoff=x[chip_edges[0][1] + 25])
        left_model.cutoff.fixed = True

        left_fit = irls(left_x, left_data, left_errors, left_model, maxiter=100)

        # Fit the right chip
        right_x = x[chip_edges[1][1] - 300: chip_edges[1][1]]
        right_x = np.append(right_x, x[chip_edges[2][0]: chip_edges[2][0] + 300])

        right_data = y[chip_edges[1][1] - 300: chip_edges[1][1]]
        right_data = np.append(right_data , y[chip_edges[2][0]: chip_edges[2][0] + 300])

        right_errors = float(hdu['SCI'].header['RDNOISE']) ** 2.0
        right_errors += right_data * np.median(hdu['SCI'].data[0])
        right_errors **= 0.5
        right_errors /= np.median(hdu['SCI'].data[0])

        right_model = offset_right_model(cutoff=x[chip_edges[1][1] + 25])
        right_model.cutoff.fixed = True

        right_fit = irls(right_x, right_data, right_errors, right_model, maxiter=100)

        hdu['SCI'].data[0, : chip_edges[0][1] + 35] /= left_fit.scale
        hdu['SCI'].data[0, chip_edges[2][0] - 35 :] /= right_fit.scale
        hdu.flush()
        hdu.close()


def makesensfunc(scifiles, objname, base_stddir, extfile):
    #TODO use individual standard star observations in each setting, not just red and blue
    for f in scifiles:
        redorblue = getredorblue(f)
        # If this is a standard star, run standard
        # Standards will have an observation class of either progCal or partnerCal
        # Standards will have an observation class of either progCal or partnerCal
        obsclass = fits.getval(f[:-4] + '.fits', 'OBSCLASS')
        if obsclass == 'progCal' or obsclass == 'partnerCal':
            # Figure out which directory the stardard star is in
            stddir = iraf.osfn('gmisc$lib/onedstds/') + base_stddir
            
            # iraf.gsstandard('est' + f[:-4], 'std' + redorblue,
            #                'sens' + redorblue, starname=objname.lower(),
            #                caldir='gmisc$lib/onedstds/'+stddir, fl_inter=True)

            specsens('et' + f[:-4] + '.fits', 'sens' + redorblue + '.fits',
                     stddir + objname + '.dat' , extfile,
                     float(fits.getval(f[:-4] + '.fits', 'AIRMASS')),
                     float(fits.getval(f[:-4] + '.fits', 'EXPTIME')))


def calibrate(scifiles, extfile, observatory):
    for f in scifiles:
        redorblue = getredorblue(f)
        iraf.unlearn(iraf.gscalibrate)
        iraf.gscalibrate('et' + f[:-4] + '.fits',
                         sfunc='sens' + redorblue + '.fits', fl_ext=True, fl_vardq=True,
                         extinction=extfile, observatory=observatory)
        
        if os.path.exists('cet' + f[:-4] + '.fits'):
            iraf.unlearn(iraf.splot)
            iraf.splot('cet' + f.replace('.txt', '.fits') + '[sci]')  # just to check
            
         
def updatecomheader(extractedfiles, objname):
    airmasses = []
    exptimes = []
    for f in extractedfiles:
        airmasses.append(float(fits.getval(f, 'AIRMASS')))
        exptimes.append(float(fits.getval(f, 'EXPTIME')))
    
    fits.setval(objname + '_com.fits', 'AIRMASS', value=np.mean(airmasses))
    fits.setval(objname + '_com.fits', 'SLIT', value=fits.getval(extractedfiles[0], 'MASKNAME').replace('arcsec', ''))
    
    comhdu = fits.open(objname + '_com.fits', mode='update')
    
    extractedhdu = fits.open(extractedfiles[0])
    for k in extractedhdu[0].header.keys():
        if not k in comhdu[0].header.keys():
            extractedhdu[0].header.cards[k].verify('fix')
            comhdu[0].header.append(extractedhdu[0].header.cards[k])
    
    comhdu.flush(output_verify='fix')
    comhdu.close()
    extractedhdu.close()
    dateobs = fits.getval(objname + '_com.fits', 'DATE-OBS')
    dateobs += 'T' + fits.getval(objname + '_com.fits', 'TIME-OBS')
    fits.setval(objname + '_com.fits', 'DATE-OBS', value=dateobs)

def cleanfinal(filename):
    # Clean the data of infs and nans
    hdu = fits.open(filename, mode='update')
    hdu[0].data[np.isnan(hdu[0].data)] = 0.0
    hdu[0].data[np.isinf(hdu[0].data)] = 0.0
    hdu.flush()
    hdu.close()
    
    
def rescale1e15(filename):
    hdu = fits.open(filename, mode='update')
    hdu[0].data *= 1e-15
    hdu.flush()
    hdu.close()
    

def run():
    # copy over sensr.fits, sensb.fits files
    # before running this script
    
    # launch the image viewer
    # os.system('ds9 &')
    
    topdir = os.getcwd()
    # Get the raw directory
    rawpath = '%s/raw/' % topdir
    
    # Sort the files into the correct directories
    fs = sort()
    
    # Change into the reduction directory
    iraf.cd('work')

    # Initialize variables that depend on which site was used
    extfile, observatory, base_stddir, rawpath = init_northsouth(fs, topdir, rawpath)
       
    # Get the observation type
    obstypes, obsclasses = getobstypes(fs)

    if dobias:
        # Make the bias frame
        makebias(fs, obstypes, rawpath)
    
    # get the object name
    objname = getobjname(fs, obstypes)
    
    # Make the text files for the IRAF tasks
    maketxtfiles(fs, obstypes, obsclasses, objname)
                
    # remember not to put ".fits" on the end of filenames!
    flatfiles, arcfiles, scifiles = gettxtfiles(fs, objname)

    # Get the wavelength solution which is apparently needed for everything else
    wavesol(arcfiles, rawpath)

    # Make the QE correction
    make_qecorrection(arcfiles)

    # Make the master flat field image
    makemasterflat(flatfiles, rawpath)

    # Flat field and rectify the science images
    scireduce(scifiles, rawpath)

    # Run sky subtraction
    skysub(scifiles, rawpath)

    # Run LA Cosmic
    crreject(scifiles)

    # Fix the cosmic ray pixels
    fixpix(scifiles)

    # Extract the 1D spectrum
    extract(scifiles)

    # Rescale the chips based on the science image
    #rescale_chips(scifiles)

    # If standard star, make the sensitivity function
    makesensfunc(scifiles, objname, base_stddir, extfile)
    
    # Flux calibrate the spectrum
    calibrate(scifiles, extfile, observatory)

    extractedfiles = glob('cet*.fits')

    # Write the spectra to ascii
    for f in scifiles:
        if os.path.exists('cet' + f[:-4] + '.fits'):
            split1d('cet' + f[:-4] + '.fits')
            # Make the ascii file
            spectoascii('cet' + f[:-4] + '.fits',f[:-4] + '.dat')
            
    # Get all of the extracted files
    splitfiles = glob('cet*c[1-9].fits')
    
    # Combine the spectra
    speccombine(splitfiles, objname + '_com.fits')
    
    # write out the ascii file
    spectoascii(objname + '_com.fits', objname + '_com.dat')
    
    # Update the combined file with the necessary header keywords
    updatecomheader(extractedfiles, objname)
    
    # If a standard star, make a telluric correction
    obsclass = fits.getval(objname + '_com.fits', 'OBSCLASS')
    if obsclass == 'progCal' or obsclass == 'partnerCal':
        # Make telluric
        mktelluric(objname + '_com.fits')
        
    # Telluric Correct
    telluric(objname + '_com.fits', objname + '.fits')
    
    #Clean the data of nans and infs
    cleanfinal(objname + '.fits')
    
    # Write out the ascii file
    spectoascii(objname + '.fits', objname + '.dat')
    
    # Multiply by 1e-15 so the units are correct in SNEx:
    rescale1e15(objname + '.fits')
    
    # Change out of the reduction directory
    iraf.cd('..')


if __name__ == "__main__":
    run()
