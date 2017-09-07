#!/usr/bin/env python
'''
Created on Nov 7, 2014

@author: cmccully
'''

import os

import numpy as np
from astropy.io import fits
from pyraf import iraf
from scipy import optimize

from lcogtgemini.file_utils import getsetupname
from lcogtgemini.fits_utils import sanitizeheader, tofits, fitshdr_to_wave
from lcogtgemini.fitting import offset_left_model, offset_right_model, irls, fitxcor
from lcogtgemini.utils import magtoflux, fluxtomag, get_binning

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
dodq = False


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


def extract(scifiles):
    for f in scifiles:    
        iraf.unlearn(iraf.gsextract)
        # Extract the specctrum
        iraf.gsextract('t' + f[:-4], fl_inter='yes', bfunction='legendre', fl_vardq=dodq,
                       border=2, bnaverage=-3, bniterate=2, blow_reject=2.0,
                       bhigh_reject=2.0, long_bsample='-100:-40,40:100',
                       background='fit', weights='variance',
                       lsigma=3.0, usigma=3.0, tnsum=100, tstep=100, mode='h')

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


def rescale1e15(filename):
    hdu = fits.open(filename, mode='update')
    hdu[0].data *= 1e-15
    hdu.flush()
    hdu.close()
