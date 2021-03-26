import lcogtgemini
from glob import glob
from astropy.io import fits
from pyraf import iraf
import numpy as np


def makebias(fs, obstypes, rawpath):
    for f in fs:
        if f[-10:] == '_bias.fits':
            iraf.cp(f, 'bias.fits')
        elif 'bias' in f:
            iraf.cp(f, './')

    if len(glob('bias*.fits')) == 0:
        bias_files = fs[obstypes == 'BIAS']
        binnings = [fits.getval(f, 'CCDSUM', 1).replace(' ', 'x') for f in bias_files]
        for binning in list(set(binnings)):
            # Make the master bias
            biastxtfile = open('bias{binning}.txt'.format(binning=binning), 'w')
            biasfs = bias_files[np.array(binnings) == binning]
            for f in biasfs:
                biastxtfile.writelines(f.split('/')[-1] + '\n')
            biastxtfile.close()
            iraf.gbias('@%s/bias{binning}.txt'.format(binning=binning) % os.getcwd(),
                       'bias{binning}'.format(binning=binning), rawpath=rawpath, fl_over=lcogtgemini.dooverscan,
                       fl_vardq=lcogtgemini.dodq)