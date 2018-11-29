import lcogtgemini
import os
from pyraf import iraf


def make_qecorrection(arcfiles):
    for f in arcfiles:
        # read in the arcfile name
        with open(f) as txtfile:
            arcimage = txtfile.readline()
            # Strip off the newline character
            arcimage = 'g' + arcimage.split('\n')[0]
        if not os.path.exists(f[:-8] + '.qe.fits'):
            iraf.gqecorr(arcimage, refimages=f[:-4]+'.arc.fits', fl_correct=False, fl_keep=True,
                         corrimages=f[:-8] + '.qe.fits', verbose=True, fl_vardq=lcogtgemini.dodq)
