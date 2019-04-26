import numpy as np
from astropy.io import fits
from astroscrappy.astroscrappy import detect_cosmics

import lcogtgemini
from lcogtgemini.fits_utils import tofits
from lcogtgemini import fixpix


def crreject(scifiles):
    for f in scifiles:
        # run lacosmicx
        hdu = fits.open('t' + f.replace('.txt', '.fits'))
        d = hdu[2].data.copy()
        readnoise = float(hdu[2].header['RDNOISE'])

        hdu_skysub = fits.open('st' + f.replace('.txt', '.fits'))
        bkg = d - hdu_skysub[2].data

        mask = d == 0.0
        mask = np.logical_or(mask, d > (50000.0 * float(hdu[0].header['GAINMULT'])))
        if lcogtgemini.dodq:
            mask = np.logical_or(mask,  hdu['DQ'].data)

        crmask, _cleanarr = detect_cosmics(d, inmask=mask, bkg=bkg, sigclip=5.0,
                                           objlim=10.0, sigfrac=0.2, gain=1.0, readnoise=readnoise)

        tofits(f[:-4] + '.lamask.fits', np.array(crmask, dtype=np.uint8), hdr=hdu['SCI'].header.copy(), clobber=True)
        fixpix.run_fixpix('t' + f[:-4] + '.fits[2]', f[:-4] + '.lamask.fits')
