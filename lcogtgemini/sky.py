import lcogtgemini
from pyraf import iraf

def skysub(scifiles, rawpath):
    for f in scifiles:
        # sky subtraction
        # output has an s prefixed on the front
        # This step is currently quite slow for Gemini-South data
        iraf.unlearn(iraf.gsskysub)
        iraf.gsskysub('t' + f[:-4], long_sample='*', fl_inter='no', fl_vardq=lcogtgemini.dodq,
                      naverage=-10, order=1, low_reject=2.0, high_reject=2.0,
                      niterate=10, mode='h')