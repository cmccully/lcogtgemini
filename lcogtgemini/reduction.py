import lcogtgemini
from astropy.io import fits
from lcogtgemini.utils import get_binning
from lcogtgemini.file_utils import getsetupname
from pyraf import iraf


def scireduce(scifiles, rawpath):
    for f in scifiles:
        binning = get_binning(f, rawpath)
        setupname = getsetupname(f)
        if lcogtgemini.dobias:
            bias_filename = "bias{binning}".format(binning=binning)
        else:
            bias_filename = ''
        # gsreduce subtracts bias and mosaics detectors
        iraf.unlearn(iraf.gsreduce)
        iraf.gsreduce('@' + f, outimages=f[:-4]+'.mef', rawpath=rawpath, bias=bias_filename, fl_bias=lcogtgemini.dobias,
                      fl_over=lcogtgemini.dooverscan, fl_fixpix='no', fl_flat=False, fl_gmosaic=False, fl_cut=False,
                      fl_gsappwave=False, fl_oversize=False, fl_vardq=lcogtgemini.dodq)

        if lcogtgemini.do_qecorr:
            # Renormalize the chips to remove the discrete jump in the
            # sensitivity due to differences in the QE for different chips
            iraf.unlearn(iraf.gqecorr)
            iraf.gqecorr(f[:-4]+'.mef', outimages=f[:-4]+'.qe.fits', fl_keep=True, fl_correct=True, fl_vardq=dodq,
                         refimages=setupname + '.arc.arc.fits', corrimages=setupname +'.qe.fits', verbose=True)

            iraf.unlearn(iraf.gmosaic)
            iraf.gmosaic(f[:-4]+'.qe.fits', outimages=f[:-4] +'.fits', fl_vardq=lcogtgemini.dodq, fl_clean=False)
        else:
            iraf.unlearn(iraf.gmosaic)
            iraf.gmosaic(f[:-4]+'.mef.fits', outimages=f[:-4] +'.fits', fl_vardq=lcogtgemini.dodq, fl_clean=False)

        # Flat field the image
        hdu = fits.open(f[:-4]+'.fits', mode='update')
        hdu['SCI'].data /= fits.getdata(setupname+'.flat.fits', extname='SCI')
        hdu.flush()
        hdu.close()

        # Transform the data based on the arc  wavelength solution
        iraf.unlearn(iraf.gstransform)
        iraf.gstransform(f[:-4], wavtran=setupname + '.arc', fl_vardq=lcogtgemini.dodq)