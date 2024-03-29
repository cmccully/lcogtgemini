import lcogtgemini
from lcogtgemini.combine import speccombine
from lcogtgemini.cosmicrays import crreject
from lcogtgemini.sky import skysub
from lcogtgemini.reduction import scireduce, extract
from lcogtgemini.utils import get_binning, rescale1e15
from lcogtgemini.qe import make_qecorrection
from lcogtgemini.wavelengths import wavesol, calculate_wavelengths
from lcogtgemini.telluric import telluric_correct, mktelluric, telluric_correction_exists
from lcogtgemini.fits_utils import clean_nans, updatecomheader, spectoascii
from lcogtgemini.flats import makemasterflat
from lcogtgemini.flux_calibration import flux_calibrate, makesensfunc
from lcogtgemini.extinction import correct_for_extinction
from lcogtgemini.file_utils import getobstypes, getobjname, gettxtfiles, maketxtfiles
from lcogtgemini.sort import sort, init_northsouth
from lcogtgemini.bpm import get_bad_pixel_mask
from lcogtgemini.utils import get_y_roi
from lcogtgemini.bias import makebias
import os
from astropy.io import fits
from pyraf import iraf
from glob import glob
from matplotlib import pyplot
pyplot.ion()


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

    # get the object name
    objname = getobjname(fs, obstypes)

    # Make the text files for the IRAF tasks
    maketxtfiles(fs, obstypes, obsclasses, objname)

    # remember not to put ".fits" on the end of filenames!
    flatfiles, arcfiles, scifiles = gettxtfiles(fs, objname)

    binnings = set([get_binning(scifile, rawpath) for scifile in scifiles])
    yroi = get_y_roi(scifiles[0], rawpath)

    get_bad_pixel_mask(binnings, yroi)

    if lcogtgemini.dobias:
        # Make the bias frame
        makebias(fs, obstypes, rawpath)

    # Get the wavelength solution which is apparently needed for everything else
    wavesol(arcfiles, rawpath)

    if lcogtgemini.do_qecorr:
        # Make the QE correction
        make_qecorrection(arcfiles)

    # Calculate the wavelengths of the unmosaiced data
    calculate_wavelengths(arcfiles, rawpath)

    # Make the master flat field image
    makemasterflat(flatfiles, rawpath)

    # Flat field and rectify the science images
    scireduce(scifiles, rawpath)

    # Run sky subtraction
    skysub(scifiles, rawpath)

    # Run LA Cosmic
    crreject(scifiles)

    # Extract the 1D spectrum
    extract(scifiles)

    # Correct for extinction
    correct_for_extinction(scifiles, extfile)

    # If standard star, make the sensitivity function
    makesensfunc(scifiles, objname, base_stddir)

    # Flux calibrate the spectrum
    flux_calibrate(scifiles)

    extractedfiles = glob('cxet*.fits')

    # If a standard star, make the telluric correction
    obsclass = fits.getval(extractedfiles[0], 'OBSCLASS')
    if obsclass == 'progCal' or obsclass == 'partnerCal':
        speccombine(extractedfiles, objname+'.notel.fits')
        updatecomheader(extractedfiles, objname + '.notel.fits')
        mktelluric(objname + '.notel.fits', objname, base_stddir)

    if telluric_correction_exists():
        # Telluric Correct
        files_to_combine = telluric_correct(extractedfiles)
    else:
        files_to_combine = extractedfiles

    # Combine the spectra
    speccombine(files_to_combine, objname + '.fits')

    # Update the combined file with the necessary header keywords
    updatecomheader(extractedfiles, objname + '.fits')

    #Clean the data of nans and infs
    clean_nans(objname + '.fits')

    # Write out the ascii file
    spectoascii(objname + '.fits', objname + '.dat')

    # Multiply by 1e-15 so the units are correct in SNEx:
    rescale1e15(objname + '.fits')

    # Change out of the reduction directory
    iraf.cd('..')


if __name__ == "__main__":
    run()
