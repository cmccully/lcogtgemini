import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import fits, ascii
import os
from glob import glob
from pyraf import iraf


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


def getobjname(fs, obstypes):
    objname = fits.getval(fs[obstypes == 'OBJECT'][0], 'OBJECT', ext=0).lower()

    # get rid of nonsense in the name (like the plus and whitespace
    objname = objname.replace('+', '')
    objname = ''.join(objname.split())

    # replace ltt with just l
    objname = objname.replace('ltt', 'l')
    return objname


def getredorblue(f):
    return f.split('.')[1][1]


def getsetupname(f, calfile=False):
    if calfile:
        setupname = f.split('.')[0] + '.' + f.split('.')[1]
    else:
        # Get the setup base name by removing the exposure number
        setupname = f.split('.')[0] + '.' + f.split('.')[1][1:]

    return setupname


def gettxtfiles(objname):

    flatfiles = np.array(glob('*.flat.txt'))

    # reduce the CuAr arcfiles.  Not flat fielded, gaps are not fixpixed
    arcfiles = np.array(glob('*.arc.txt'))

    # reduce the science files
    scifiles = glob(objname + '*.txt')

    nonscifiles = []
    # remove the arcs and flats
    for f in scifiles:
        if 'arc' in f or 'flat' in f:
            nonscifiles.append(f)

    for f in nonscifiles:
        scifiles.remove(f)
    scifiles = np.array(scifiles)

    return flatfiles, arcfiles, scifiles


def get_base_name(f):
    objname = getobjname(np.array([f]), np.array(['OBJECT']))
    # red or blue setting
    redblue = fits.getval(f, 'GRATING')[0].lower()
    # central wavelength
    lamcentral = fits.getval(f, 'CENTWAVE')

    return '%s.%s%i' % (objname, redblue, lamcentral)


def maketxtfiles(fs, obstypes, obsclasses, objname):
    # go through each of the files (Ignore bias and aquisition files)
    goodfiles = np.logical_and(obsclasses != 'acqCal', obsclasses != 'acq')
    goodfiles = np.logical_and(goodfiles, obstypes != 'BIAS')
    goodfiles = np.logical_and(goodfiles, obstypes != 'BPM')
    goodfiles = np.logical_and(goodfiles, obsclasses != 'sensitivity')
    correct_names = np.logical_or([os.path.basename(f)[0] == 'S' for f in fs],
                                  [os.path.basename(f)[0] == 'N' for f in fs])
    goodfiles = np.logical_and(correct_names, goodfiles)

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


def get_images_from_txt_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


def get_standard_file(objname, base_stddir):
    if os.path.exists(objname+'.std.dat'):
        standard_file = objname+'.std.dat'
    else:
        standard_file = os.path.join(iraf.osfn('gmisc$lib/onedstds/'), base_stddir, objname + '.dat')
    return standard_file


def read_standard_file(filename, maskname):
    standard = ascii.read(filename)
    standard['col2'] = smooth(maskname, standard['col2'])
    return standard


def read_telluric_model(maskname):
    # Read in the telluric model
    telluric = ascii.read('telluric_model.dat')
    telluric['col2'] = smooth(maskname, telluric['col2'])
    return telluric


def smooth(maskname, data):
    # Smooth the spectrum to the size of the slit
    # I measured 5 angstroms FWHM for a 1 arcsecond slit
    # the 2.355 converts to sigma
    smoothing_scale = 5.0 * float(maskname.split('arc')[0]) / 2.355

    return convolve(data, Gaussian1DKernel(stddev=smoothing_scale))
