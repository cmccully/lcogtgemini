import lcogtgemini
import numpy as np
from pyraf import iraf
import os
from glob import glob
from astropy.io import fits

def sort():
    if not os.path.exists('raw'):
        iraf.mkdir('raw')
    fs = glob('*.fits')
    fs += glob('*.dat')
    for f in fs:
        iraf.mv(f, 'raw/')

    # Make a reduction directory
    if not os.path.exists('work'):
        iraf.mkdir('work')

    sensfs = glob('raw/sens*.fits')
    if len(sensfs) != 0:
        for f in sensfs:
            iraf.cp(f, 'work/')

    if os.path.exists('raw/telcor.dat'):
        iraf.cp('raw/telcor.dat', 'work/')

    if os.path.exists('raw/telluric_model.dat'):
        iraf.cp('raw/telluric_model.dat', 'work/')

    std_files = glob('raw/*.std.dat')
    if len(std_files) != 0:
        for f in std_files:
            iraf.cp(f, 'work/')

    if os.path.exists('raw/bias.fits'):
        iraf.cp('raw/bias.fits', 'work/')

    bpm_file_list = glob('raw/bpm_g?.fits')
    if len(bpm_file_list) != 0:
        iraf.cp(bpm_file_list[0], 'work/')

    fs = glob('raw/*.qe.fits')
    if len(fs) > 0:
        for f in fs:
            iraf.cp(f, 'work/')

    fs = glob('raw/*.wavelengths.fits')
    if len(fs) > 0:
        for f in fs:
            iraf.cp(f, 'work/')

    fs = glob('raw/*.flat.fits')
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
    lcogtgemini.is_GS = fits.getval(fs[0], 'OBSERVAT') == 'Gemini-South'

    if 'Hamamatsu' in fits.getval(fs[0], 'DETECTOR'):
        lcogtgemini.dooverscan = True
        lcogtgemini.do_qecorr = True
        lcogtgemini.detector = 'Hamamatsu'
        lcogtgemini.namps = 12
        if lcogtgemini.is_GS:
            lcogtgemini.xchip_shifts = [-1.2, 0.0, 0.0]
            lcogtgemini.ychip_shifts = [0.71, 0.0, -0.73]
            lcogtgemini.chip_rotations = [0.0, 0.0, 0.0]
            lcogtgemini.chip_gap_size = 61.0
    else:
        lcogtgemini.namps = 3
        lcogtgemini.dooverscan = True
        lcogtgemini.chip_gap_size = 37.0
        lcogtgemini.xchip_shifts = [-1.49, 0.0, 4.31]
        lcogtgemini.ychip_shifts = [-0.22, 0.0, 2.04]
        lcogtgemini.chip_rotations = [0.011, 0.0, 0.012]
        lcogtgemini.detector = 'EEV'

    if lcogtgemini.is_GS:
        base_stddir = 'ctionewcal/'
        observatory = 'Gemini-South'
        extfile = iraf.osfn('gmisc$lib/onedstds/ctioextinct.dat')
    else:
        base_stddir = 'spec50cal/'
        extfile = iraf.osfn('gmisc$lib/onedstds/kpnoextinct.dat')
        observatory = 'Gemini-North'
    return extfile, observatory, base_stddir, rawpath
