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
    hdu = PrimaryHDU(data)
    if not (hdr is None):
        hdu.header += hdr
    hdulist = HDUList([hdu])
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