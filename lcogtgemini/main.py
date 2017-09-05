import lcogtgemini
from lcogtgemini import dobias, makebias, scireduce, skysub, crreject, extract, split1d, spectoascii, speccombine, updatecomheader, \
    rescale1e15
from lcogtgemini.utils import fixpix
from lcogtgemini.qe import make_qecorrection
from lcogtgemini.wavelengths import wavesol
from lcogtgemini.telluric import telluric, mktelluric
from lcogtgemini.fits_utils import clean_nans
from lcogtgemini.flats import makemasterflat
from lcogtgemini.flux_calibration import calibrate, makesensfunc
from lcogtgemini.file_utils import getobstypes, getobjname, gettxtfiles, maketxtfiles
from lcogtgemini.sort import sort, init_northsouth


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

    if lcogtgemini.do_qecorr:
        # Make the QE correction
        make_qecorrection(arcfiles)

    # Calculate the wavelengths of the unmosaiced data
    calculate_wavelengths(arcfiles)

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
    clean_nans(objname + '.fits')

    # Write out the ascii file
    spectoascii(objname + '.fits', objname + '.dat')

    # Multiply by 1e-15 so the units are correct in SNEx:
    rescale1e15(objname + '.fits')

    # Change out of the reduction directory
    iraf.cd('..')


if __name__ == "__main__":
    run()
