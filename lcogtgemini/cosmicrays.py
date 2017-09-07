import numpy as np
from astropy.io import fits
from astroscrappy.astroscrappy import detect_cosmics

import lcogtgemini
from lcogtgemini.fits_utils import tofits
from lcogtgemini import fixpix


def crreject(scifiles):
    for f in scifiles:
        # run lacosmicx
        hdu = fits.open('st' + f.replace('.txt', '.fits'))

        readnoise = float(hdu[2].header['RDNOISE'])
        # figure out what pssl should be approximately
        d = hdu[2].data.copy()
        dsort = np.sort(d.ravel())
        nd = dsort.shape[0]
        # Calculate the difference between the 16th and 84th percentiles to be
        # robust against outliers
        dsig = (dsort[int(round(0.84 * nd))] - dsort[int(round(0.16 * nd))]) / 2.0
        pssl = (dsig * dsig - readnoise * readnoise)

        mask = d == 0.0
        mask = np.logical_or(mask, d > (50000.0 * float(hdu[0].header['GAINMULT'])))
        if lcogtgemini.dodq:
            mask = np.logical_or(mask,  hdu['DQ'].data)

        crmask, _cleanarr = detect_cosmics(d, inmask=mask, sigclip=4.0,
                                           objlim=1.0, sigfrac=0.05, gain=1.0,
                                           readnoise=readnoise, pssl=pssl)

        tofits(f[:-4] + '.lamask.fits', np.array(crmask, dtype=np.uint8), hdr=hdu['SCI'].header.copy())
        fixpix.run_fixpix('t' + f[:-4] + '.fits[2]', f[:-4] + '.lamask.fits')
