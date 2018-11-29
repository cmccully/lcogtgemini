import lcogtgemini
from pyraf import iraf
import os
from glob import glob
from astropy.io import fits


def sort(workpath, rawpath):
    # Make a reduction directory
    if not os.path.exists(workpath):
        iraf.mkdir(workpath)

    sensfs = glob('sens*.fits')
    if len(sensfs) != 0:
        for f in sensfs:
            iraf.cp(f, workpath)

    if os.path.exists('telcor.dat'):
        iraf.cp('telcor.dat', workpath)

    if os.path.exists('telluric_model.dat'):
        iraf.cp('telluric_model.dat', workpath)

    std_files = glob('*.std.dat')
    if len(std_files) != 0:
        for f in std_files:
            iraf.cp(f, workpath)

    if os.path.exists('bias.fits'):
        iraf.cp('bias.fits', workpath)

    bpm_file_list = glob('bpm_g?.fits')
    for bpm_file in bpm_file_list:
        iraf.cp(bpm_file, workpath)

    fs = glob('*.qe.fits')
    if len(fs) > 0:
        for f in fs:
            iraf.cp(f, workpath)

    fs = glob('*.wavelengths.fits')
    if len(fs) > 0:
        for f in fs:
            iraf.cp(f, workpath)

    fs = glob('*.flat.fits')
    if len(fs) > 0:
        for f in fs:
            iraf.cp(f, workpath)

    if not os.path.exists(rawpath):
        iraf.mkdir(rawpath)
    for f in glob('*.fits') + glob('*.dat'):
        iraf.mv(f, rawpath)


def get_raw_files(rawpath):
    # make a list of the raw files
    fs = set(glob(os.path.join(rawpath, '*.fits')))
    fs -= set(glob(os.path.join(rawpath, 'sens*.fits')))
    fs -= set(glob(os.path.join(rawpath, 'bpm_g?.fits')))
    return fs


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
        base_stddir = 'ctionewcal/'
        extfile = iraf.osfn('gmisc$lib/onedstds/ctioextinct.dat')
    else:
        base_stddir = 'spec50cal/'
        extfile = iraf.osfn('gmisc$lib/onedstds/kpnoextinct.dat')
    return extfile, base_stddir
