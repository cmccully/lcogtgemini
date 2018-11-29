import os
from pyraf import iraf
from lcogtgemini import file_utils


def fixpix(txtfile, rawpath, binning, namps):
    images = file_utils.get_images_from_txt_file(txtfile)
    for image in images:
        if not os.path.exists('../raw_fixpix'):
            iraf.mkdir('../raw_fixpix')
        iraf.cp(os.path.join(rawpath, image), '../raw_fixpix/')
        for i in range(1, namps + 1):
            run_fixpix(os.path.join('../raw_fixpix', image) + '[{i}]'.format(i=i),
                       'bpm.{i}.{binning}'.format(i=i, binning=binning))
    return '../raw_fixpix'


def run_fixpix(filename, maskname):
    # Run fixpix to interpolate over cosmic rays and bad pixels
    iraf.unlearn(iraf.fixpix)
    iraf.fixpix(filename, maskname, mode='h')
