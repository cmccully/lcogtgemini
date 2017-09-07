import lcogtgemini
from lcogtgemini.utils import get_binning
from lcogtgemini.file_utils import getsetupname
from lcogtgemini import fits_utils
from lcogtgemini import fixpix
from pyraf import iraf
from astropy.io import fits


def reduce_flat(flatfile, rawpath):

    fixed_rawpath = fixpix.fixpix(flatfile, rawpath)
    binning = get_binning(flatfile, rawpath)
    setupname = getsetupname(flatfile)
    # Use IRAF to get put the data in the right format and subtract the
    # bias
    # This will currently break if multiple flats are used for a single setting
    iraf.unlearn(iraf.gsreduce)
    if lcogtgemini.dobias:
        biasfile = "bias{binning}".format(binning=binning)
    else:
        biasfile = ''
    iraf.gsreduce('@' + flatfile, outimages=flatfile[:-4] + '.mef.fits', rawpath=fixed_rawpath, fl_bias=lcogtgemini.dobias,
                  bias=biasfile, fl_over=lcogtgemini.dooverscan, fl_flat=False, fl_gmosaic=False,
                  fl_fixpix=False, fl_gsappwave=False, fl_cut=False, fl_title=False,
                  fl_oversize=False, fl_vardq=lcogtgemini.dodq)

    if lcogtgemini.do_qecorr:
        # Renormalize the chips to remove the discrete jump in the
        # sensitivity due to differences in the QE for different chips
        iraf.unlearn(iraf.gqecorr)

        iraf.gqecorr(flatfile[:-4] + '.mef', outimages=flatfile[:-4] + '.qe.fits', fl_keep=True, fl_correct=True,
                     refimages=f[:-4].replace('flat', 'arc.arc.fits'),
                     corrimages=f[:-9] + '.qe.fits', verbose=True, fl_vardq=lcogtgemini.dodq)
        mosaic_input = flatfile[:-4] + '.qe.fits'
    else:
        mosaic_input = flatfile[:-4] + '.mef.fits'

    iraf.unlearn(iraf.gmosaic)
    iraf.gmosaic(mosaic_input, outimages=flatfile[:-4] + '.mos.fits', fl_vardq=lcogtgemini.dodq, fl_clean=False)
    iraf.unlearn(iraf.gstransform)
    iraf.gstransform(flatfile[:-4]+'.mos.fits', wavtran=setupname + '.arc', fl_vardq=lcogtgemini.dodq)


def makemasterflat(flatfiles, rawpath, plot=True):
    # normalize the flat fields
    for flatfile in flatfiles:
        reduce_flat(flatfile, rawpath)
        setupname = getsetupname(f)
        flat_hdu = fits.open('t'+ f[:-4] + '.mos.fits')

        data = np.median(flat_hdu['SCI'].data, axis=0)
        wavelengths = fits_utils.fitshdr_to_wave(flat_hdu['SCI'].header)
        best_fit = fit_pfm(wavelengths, data)

        # Open the unmoasiced (and optionally qe corrected flat file)
        if lcogtgemini.do_qecorr:
            unmosaiced_file = flatfile[:-4] + '.qe.fits'
        else:
            unmosaiced_file = flatfile[:-4] + '.mef.fits'

        unmosaiced_hdu = fits.open(unmosaiced_file)
        wavelengths_hdu = fits.open(setupname+'.wavelengths.fits')
        for i in range(13):
            unmosaiced_hdu[i].data /= eval_fit(best_fit, wavelengths_hdu[i].data)
        flat_hdu.writeto(flatfile[:-4] + '.fits')