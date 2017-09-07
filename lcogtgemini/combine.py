import os
import numpy as np
import lcogtgemini
from astropy.io import fits
from scipy import optimize
from lcogtgemini import fits_utils
from lcogtgemini import fitting

from pyraf import iraf


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
    lam = fits_utils.fitshdr_to_wave(hdu['SCI'].header)
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
        lam = fits_utils.fitshdr_to_wave(hdu[0].header.copy())

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
                  reject='avsigclip', lthreshold='INDEF', w1=lcogtgemini.bluecut)


def rescale_chips(scifiles):
    for f in scifiles:
        hdu = fits.open('et'+ f[:-4]+'.fits', mode='update')
        chip_edges = get_chipedges(hdu['SCI'].data)
        lam = fits_utils.fitshdr_to_wave(hdu['SCI'].header)
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

        left_model = fitting.offset_left_model(cutoff=x[chip_edges[0][1] + 25])
        left_model.cutoff.fixed = True

        left_fit = fitting.irls(left_x, left_data, left_errors, left_model, maxiter=100)

        # Fit the right chip
        right_x = x[chip_edges[1][1] - 300: chip_edges[1][1]]
        right_x = np.append(right_x, x[chip_edges[2][0]: chip_edges[2][0] + 300])

        right_data = y[chip_edges[1][1] - 300: chip_edges[1][1]]
        right_data = np.append(right_data , y[chip_edges[2][0]: chip_edges[2][0] + 300])

        right_errors = float(hdu['SCI'].header['RDNOISE']) ** 2.0
        right_errors += right_data * np.median(hdu['SCI'].data[0])
        right_errors **= 0.5
        right_errors /= np.median(hdu['SCI'].data[0])

        right_model = fitting.offset_right_model(cutoff=x[chip_edges[1][1] + 25])
        right_model.cutoff.fixed = True

        right_fit = fitting.irls(right_x, right_data, right_errors, right_model, maxiter=100)

        hdu['SCI'].data[0, : chip_edges[0][1] + 35] /= left_fit.scale
        hdu['SCI'].data[0, chip_edges[2][0] - 35 :] /= right_fit.scale
        hdu.flush()
        hdu.close()
