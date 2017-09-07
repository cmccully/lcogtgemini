import numpy as np
from astropy.io import fits


def sanitizeheader(hdr):
    # Remove the mandatory keywords from a header so it can be copied to a new
    # image.
    hdr = hdr.copy()

    # Let the new data decide what these values should be
    for i in ['SIMPLE', 'BITPIX', 'BSCALE', 'BZERO']:
        if i in hdr.keys():
            hdr.pop(i)

#    if hdr.has_key('NAXIS'):
    if 'NAXIS' in hdr.keys():
        naxis = hdr.pop('NAXIS')
        for i in range(naxis):
            hdr.pop('NAXIS%i' % (i + 1))

    return hdr


def tofits(filename, data, hdr=None, clobber=False):
    """simple pyfits wrapper to make saving fits files easier."""
    hdu = fits.PrimaryHDU(data)
    if not (hdr is None):
        hdu.header += hdr
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename, overwrite=clobber, output_verify='ignore')


def clean_nans(filename):
    # Clean the data of infs and nans
    hdu = fits.open(filename, mode='update')
    hdu[0].data[np.isnan(hdu[0].data)] = 0.0
    hdu[0].data[np.isinf(hdu[0].data)] = 0.0
    hdu.flush()
    hdu.close()


def fitshdr_to_wave(hdr):
    crval = float(hdr['CRVAL1'])
    crpix = float(hdr['CRPIX1'])
    # Convert crpix to be zero indexed
    crpix -= 1
    if 'CDELT1' in hdr.keys():
        cdelt = float(hdr['CDELT1'])
    else:
        cdelt = float(hdr['CD1_1'])
    npix = float(hdr['NAXIS1'])
    lam = np.arange(crval - cdelt * crpix ,
                    crval + cdelt * (npix - crpix) - 1e-4,
                    cdelt)
    return lam


def hdr_pixel_range(x0, x1, y0, y1):
    return '[{0:d}:{1:d},{2:d}:{3:d}]'.format(x0, x1, y0, y1)


def get_x_pixel_range(keyword_value):
    """
    Get the x part of a section keyword
    :param keyword_value: Header keyword string
    :return: list xrange: 2 element list with start and end 1-indexed pixel values
    """
    # Strip off the brackets and split the coordinates
    pixel_sections = keyword_value[1:-1].split(',')
    return pixel_sections[0].split(':')


def cut_gs_image(filename, output_filename, pixel_range):
    """

    :param filename:
    :param output_filename:
    :param pixel_range: array-like, The range of pixels to keep, python indexed,
                        given in binned pixels
    :return:
    """
    hdu = fits.open(filename, unit16=True)
    for i in range(1, 13):
        ccdsum = hdu[i].header['CCDSUM']
        ccdsum = np.array(ccdsum.split(), dtype=np.int)

        y_ccdsec = [(pixel_range[0]  * ccdsum[1]) + 1,
                    (pixel_range[1]) * ccdsum[1]]

        x_detector_section = get_x_pixel_range(hdu[i].header['DETSEC'])
        hdu[i].header['DETSEC'] = hdr_pixel_range(int(x_detector_section[0]), int(x_detector_section[1]), y_ccdsec[0], y_ccdsec[1])

        x_ccd_section = get_x_pixel_range(hdu[i].header['CCDSEC'])
        hdu[i].header['CCDSEC'] = hdr_pixel_range(int(x_ccd_section[0]), int(x_ccd_section[1]), y_ccdsec[0], y_ccdsec[1])


        numpix = pixel_range[1] - pixel_range[0]

        x_bias_section = get_x_pixel_range(hdu[i].header['BIASSEC'])
        hdu[i].header['BIASSEC'] = hdr_pixel_range(int(x_bias_section[0]), int(x_bias_section[1]), 1, numpix)
        x_data_section = get_x_pixel_range(hdu[i].header['DATASEC'])
        hdu[i].header['DATASEC'] = hdr_pixel_range(int(x_data_section[0]), int(x_data_section[1]), 1, numpix)

        hdu[i].data = hdu[i].data[pixel_range[0]:pixel_range[1], :]

    hdu.writeto(output_filename)
    hdu.close()


def updatecomheader(extractedfiles, objname):
    airmasses = []
    exptimes = []
    for f in extractedfiles:
        airmasses.append(float(fits.getval(f, 'AIRMASS')))
        exptimes.append(float(fits.getval(f, 'EXPTIME')))

    fits.setval(objname + '_com.fits', 'AIRMASS', value=np.mean(airmasses))
    fits.setval(objname + '_com.fits', 'SLIT', value=fits.getval(extractedfiles[0], 'MASKNAME').replace('arcsec', ''))

    comhdu = fits.open(objname + '_com.fits', mode='update')

    extractedhdu = fits.open(extractedfiles[0])
    for k in extractedhdu[0].header.keys():
        if not k in comhdu[0].header.keys():
            extractedhdu[0].header.cards[k].verify('fix')
            comhdu[0].header.append(extractedhdu[0].header.cards[k])

    comhdu.flush(output_verify='fix')
    comhdu.close()
    extractedhdu.close()
    dateobs = fits.getval(objname + '_com.fits', 'DATE-OBS')
    dateobs += 'T' + fits.getval(objname + '_com.fits', 'TIME-OBS')
    fits.setval(objname + '_com.fits', 'DATE-OBS', value=dateobs)


def spectoascii(infilename, outfilename):
    hdu = fits.open(infilename)
    try:
        lam = fitshdr_to_wave(hdu['SCI'].header.copy())
        flux = hdu['SCI'].data.copy()
    except:
        lam = fitshdr_to_wave(hdu[0].header.copy())
        flux = hdu[0].data.copy()
    hdu.close()
    d = np.zeros((2, len(lam)))
    d[0] = lam
    d[1] = flux
    np.savetxt(outfilename, d.transpose())