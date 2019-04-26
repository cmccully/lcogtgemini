import lcogtgemini
import numpy as np
from pyraf import iraf
import os
from glob import glob
from astropy.io import fits


def sort():
    if not os.path.exists('raw'):
        iraf.mkdir('raw')

    for f in glob('*.fits'):
        hdr = fits.getheader(f, ext=1)
        if hdr.get('NAXIS2') == 2112:
            lcogtgemini.fits_utils.cut_gs_image(f, os.path.join('raw', f), [812, 1324], 12)
            os.remove(f)
        else:
            iraf.mv(f, 'raw/')

    for f in glob('*.dat'):
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

    std_files = glob('raw/*.std.dat')
    if len(std_files) != 0:
        for f in std_files:
            iraf.cp(f, 'work/')

    if os.path.exists('raw/bias.fits'):
        iraf.cp('raw/bias.fits', 'work/')

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
    for f in sensfs:
        fs.remove(f)
    # Add a ../ in front of all of the file names
    for i in range(len(fs)):
        fs[i] = '../' + fs[i]
    return np.array(fs)


def init_northsouth(fs):
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
            lcogtgemini.xchip_shifts = [-0.95, 0.0, 0.48]
            lcogtgemini.ychip_shifts = [-0.21739, 0.0, 0.1727]
            lcogtgemini.chip_rotations = [-0.004, 0.0, -0.00537]
            lcogtgemini.chip_gap_size = 67.0
    elif not lcogtgemini.is_GS:
        lcogtgemini.namps = 6
        lcogtgemini.dooverscan = True
        lcogtgemini.chip_gap_size = 37.0
        lcogtgemini.xchip_shifts = [-2.7, 0.0, 2.8014]
        lcogtgemini.ychip_shifts = [-0.749, 0.0, 2.05]
        lcogtgemini.chip_rotations = [-0.009, 0.0, -0.003]
        lcogtgemini.detector = 'E2V DD'

    if lcogtgemini.is_GS:
        extfile = iraf.osfn('gmisc$lib/onedstds/ctioextinct.dat')
    else:
        extfile = iraf.osfn('gmisc$lib/onedstds/kpnoextinct.dat')
    return extfile
