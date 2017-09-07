import lcogtgemini
from lcogtgemini import utils, file_utils, fixpix
from pyraf import iraf
import numpy as np
from astropy.io import fits
import os

def wavesol(arcfiles, rawpath):
    for f in arcfiles:
        fixed_rawpath = fixpix.fixpix(f, rawpath)
        binning = utils.get_binning(f, rawpath)
        iraf.unlearn(iraf.gsreduce)
        if lcogtgemini.dobias:
            bias_filename = "bias{binning}".format(binning=binning)
        else:
            bias_filename = ''
        iraf.gsreduce('@' + f, outimages=f[:-4], rawpath=fixed_rawpath,
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


def calculate_wavelengths(scifiles, rawpath):
    for f in scifiles:
        images = file_utils.get_images_from_txt_file(f)
        setupname = file_utils.getsetupname(f)
        binning = [float(i) for  i in utils.get_binning(f, rawpath).split('x')]
        for image in images:
            hdu = fits.open(os.path.join(rawpath, image))
            for i in range(1, 13):
                print('Calculating wavelengths for {setup} for amplifier {amp}'.format(setup=setupname, amp=i))
                mosaic_file = mosiac_coordinates(hdu, i, setupname, binning)
                hdu[i].data = utils.convert_pixel_list_to_array(mosaic_file, hdu[i].data.shape[1], hdu[i].data.shape[0])
            hdu.writeto(setupname + '.wavelengths.fits', clobber=True)


def mosiac_coordinates(hdu, i, setupname, binning):
    # Fake mosaicing
    X, Y = np.meshgrid(np.arange(hdu[i].data.shape[1]) + 1, np.arange(hdu[i].data.shape[0]) + 1)
    X = X.astype(np.float)
    Y = Y.astype(np.float)
    # Rotate the frame about the center for the given transformation
    x_center = (hdu[i].data.shape[1] / 2.0) + 0.5
    y_center = (hdu[i].data.shape[0] / 2.0) + 0.5
    # Subtract off the center
    X -= x_center
    Y -= y_center

    chip_number = (i - 1) // 4
    rotation = lcogtgemini.chip_rotations[chip_number]

    X = np.cos(np.radians(rotation)) * X - np.sin(np.radians(rotation)) * Y
    Y = np.sin(np.radians(rotation)) * X - np.cos(np.radians(rotation)) * Y

    # Add back in the chip centers
    X += x_center
    Y += y_center

    # Add in the X detsec
    X += (float(hdu[i].header['DETSEC'][1]) - 1.0) / binning[0]

    # Add in the chip gaps * (i - 1) // 4
    X += chip_number * lcogtgemini.chip_gap_size / binning[0]

    # Add in the chip shifts
    X += lcogtgemini.xchip_shifts[chip_number] / binning[0]
    Y += lcogtgemini.ychip_shifts[chip_number] / binning[1]

    # Write the coordinates to a text file
    lines_to_write = []
    for x, y in zip(X.ravel(), Y.ravel()):
        lines_to_write.append('{x} {y}\n'.format(x=x, y=y))

    pixel_list = setupname+'.{i}.pix.dat'.format(i=i)
    with open(pixel_list, 'w') as file_to_write:
        file_to_write.writelines(lines_to_write)

    output_wavelength_textfile = setupname + '.waves.dat'
    # Then evaluate the fit coords transformation at each pixel
    iraf.fceval(pixel_list, output_wavelength_textfile, setupname + '.arc_001')

    return output_wavelength_textfile
