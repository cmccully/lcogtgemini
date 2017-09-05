import lcogtgemini
from lcogtgemini import utils


def wavesol(arcfiles, rawpath):
    for f in arcfiles:
        images = get_images_from_txt_file(f)
        for image in images:
            for i in range(1, 13):
                utils.fixpix(os.path.join(rawpath, image)+'[{i}]'.format(i=i), lcogtgemini.bpm+'[{i}]'.format(i))

        binning = utils.get_binning(f, rawpath)
        iraf.unlearn(iraf.gsreduce)
        if lcogtgemini.dobias:
            bias_filename = "bias{binning}".format(binning=binning)
        else:
            bias_filename = ''
        iraf.gsreduce('@' + f, outimages=f[:-4], rawpath=rawpath,
                      fl_flat=False, bias=bias_filename, fl_bias=lcogtgemini.dobias,
                      fl_fixpix=False, fl_over=lcogtgemini.dooverscan, fl_cut=False, fl_gmosaic=True,
                      fl_gsappwave=True, fl_oversize=False, fl_vardq=lcogtgemini.dodq)


        # determine wavelength calibration -- 1d and 2d
        iraf.unlearn(iraf.gswavelength)
        iraf.gswavelength(f[:-4], fl_inter='yes', fl_addfeat=False, fwidth=15.0, low_reject=2.0,
                          high_reject=2.0, step=10, nsum=10, gsigma=2.0, cradius=16.0,
                          match=-6, order=7, fitcxord=7, fitcyord=7)

        if lcogtgemini.do_qecorr:
            # Make an extra random copy so that gqecorr works. Stupid Gemini.
            iraf.cp(f[:-4]+'.fits', f[:-4]+'.arc.fits')
        # transform the CuAr spectrum, for checking that the transformation is OK
        # output spectrum has prefix t
        iraf.unlearn(iraf.gstransform)
        iraf.gstransform(f[:-4], wavtran=f[:-4])


def calculate_wavelengths(arcfiles):
    for f in arcfiles:
        images = get_images_from_txt_file(f)
        setupname = get_setupname(f)
        for image in images:
            hdu = fits.open(image)
            for i in range(1, 13):
                mosaic_file = mosiac_coordinates(hdu, setupname)
                hdu[i].data = convert_pixel_list_to_array(mosaic_file)
            hdu.write(setupname + '.wavelengths.fits')


def mosiac_coordinates(hdu, setupname):
    # Fake mosaicing
    X, Y = np.meshgrid(np.arange(hdu[i].data.shape[1]) + 1, np.arange(hdu[i].data.shape[0]) + 1)
    # Rotate the frame about the center for the given transformation
    # Add in the X detsec
    # Add in the chip gaps * (i - 1) // 4
    # Add in the chip shifts
    # Write the coordinates to a text file

    # Then evaluate the fit coords transformation at each pixel
    iraf.fceval(pixel_list, output_wavelength_textfile, setupname + '.arc_001')
