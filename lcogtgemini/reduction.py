import numpy as np

import lcogtgemini
from astropy.io import fits
from lcogtgemini.utils import get_binning
from lcogtgemini.file_utils import getsetupname
from pyraf import iraf
from lcogtgemini import fixpix
from lcogtgemini import fits_utils


def scireduce(scifiles, rawpath):
    for f in scifiles:
        binning = get_binning(f, rawpath)
        fixed_rawpath = fixpix.fixpix(f, rawpath, binning, lcogtgemini.namps)

        setupname = getsetupname(f)
        if lcogtgemini.dobias:
            bias_filename = "bias{binning}".format(binning=binning)
        else:
            bias_filename = ''
        # gsreduce subtracts bias and mosaics detectors
        iraf.unlearn(iraf.gsreduce)
        iraf.gsreduce('@' + f, outimages=f[:-4]+'.mef', rawpath=fixed_rawpath, bias=bias_filename,
                      fl_bias=lcogtgemini.dobias, fl_over=lcogtgemini.dooverscan, fl_fixpix='no',
                      fl_flat=False, fl_gmosaic=False, fl_cut=False, fl_gsappwave=False, fl_oversize=False,
                      fl_vardq=lcogtgemini.dodq)

        if lcogtgemini.do_qecorr:
            # Renormalize the chips to remove the discrete jump in the
            # sensitivity due to differences in the QE for different chips
            iraf.unlearn(iraf.gqecorr)
            iraf.gqecorr(f[:-4]+'.mef', outimages=f[:-4]+'.qe.fits', fl_keep=True, fl_correct=True,
                         fl_vardq=lcogtgemini.dodq, refimages=setupname + '.arc.arc.fits',
                         corrimages=setupname +'.qe.fits', verbose=True)
            unmosaiced_name = f[:-4]+'.qe.fits'
        else:
            unmosaiced_name = f[:-4]+'.mef.fits'

        # Flat field the image
        hdu = fits.open(unmosaiced_name, mode='update')
        flat_hdu = fits.open(setupname+'.flat.fits')
        for i in range(1, lcogtgemini.namps + 1):
            hdu[i].data /= flat_hdu[i].data
        hdu.flush()
        hdu.close()


        iraf.unlearn(iraf.gmosaic)
        iraf.gmosaic(unmosaiced_name, outimages=f[:-4] + '.fits', fl_vardq=lcogtgemini.dodq,
                     fl_clean=False)
        # Transform the data based on the arc  wavelength solution
        iraf.unlearn(iraf.gstransform)
        iraf.gstransform(f[:-4], wavtran=setupname + '.arc', fl_vardq=lcogtgemini.dodq)


def extract(scifiles):
    for f in scifiles:
        iraf.unlearn(iraf.gsextract)
        # Extract the specctrum
        iraf.gsextract('t' + f[:-4], fl_inter='yes', bfunction='legendre',
                       fl_vardq=lcogtgemini.dodq,
                       border=2, bnaverage=-3, bniterate=2, blow_reject=2.0,
                       bhigh_reject=2.0, long_bsample='-100:-40,40:100',
                       background='fit', weights='variance',
                       lsigma=3.0, usigma=3.0, tnsum=100, tstep=100, mode='h')

        # Trim off below the blue side cut
        hdu = fits.open('et' + f[:-4] +'.fits', mode='update')
        lam = fits_utils.fitshdr_to_wave(hdu['SCI'].header)
        w = lam > lcogtgemini.bluecut
        trimmed_data =np.zeros((1, w.sum()))
        trimmed_data[0] = hdu['SCI'].data[0, w]
        hdu['SCI'].data = trimmed_data
        hdu['SCI'].header['NAXIS1'] = w.sum()
        hdu['SCI'].header['CRPIX1'] = 1
        hdu['SCI'].header['CRVAL1'] = lam[w][0]
        hdu.flush()

        hdu.close()
