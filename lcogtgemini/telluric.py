from lcogtgemini import fitshdr_to_wave, fitxcor, tofits, get_chipedges, telluricWaves


def telluric(filename, outfile):

    # Get the standard to use for telluric correction
    stdfile = 'telcor.dat'

    hdu = fits.open(filename)
    spec = hdu[0].data.copy()
    hdr = hdu[0].header.copy()
    hdu.close()
    waves = fitshdr_to_wave(hdr)

    telwave, telspec = np.genfromtxt(stdfile).transpose()
    # Cross-correlate the standard star and the sci spectra
    # to find wavelength shift of standard star.
    w = np.logical_and(waves > 7550., waves < 8410.)
    tw = np.logical_and(telwave > 7550., telwave < 8410.)
    p = fitxcor(waves[w], spec[w], telwave[tw], telspec[tw])
    # shift and stretch standard star spectrum to match science
    # spectrum.
    telcorr = np.interp(waves, p[0] * telwave + p[1], telspec)

    # Correct for airmass
    airmass = float(hdr['AIRMASS'])
    telcorr = telcorr ** (airmass ** 0.55)

    # Divide science spectrum by transformed standard star sub-spectrum
    correct_spec = spec / telcorr

    # Copy telluric-corrected data to new file.
    tofits(outfile, correct_spec, hdr=hdr)


def mktelluric(filename):
    #TODO Try fitting a black body instead of interpolating.
    # if it is a standard star combined file
    # read in the spectrum and calculate the wavelengths of the pixels
    hdu = fits.open(filename)
    spec = hdu[0].data.copy()
    hdr = hdu[0].header.copy()
    hdu.close()
    waves = fitshdr_to_wave(hdr)

    # Start by interpolating over the chip gaps
    chip_edges = get_chipedges(spec)
    chip_gaps = np.ones(spec.size, dtype=np.bool)
    for edge in chip_edges:
        chip_gaps[edge[0]: edge[1]] = False

    template_spectrum = signal.savgol_filter(spec, 21, 3)
    noise = np.abs(spec - template_spectrum)
    noise = ndimage.filters.gaussian_filter1d(noise, 100.0)

    # Smooth the chip gaps
    intpr = interpolate.splrep(waves[np.logical_not(chip_gaps)],
                               spec[np.logical_not(chip_gaps)],
                               w=1 / noise[np.logical_not(chip_gaps)], k=2,
                               s=10 * np.logical_not(chip_gaps).sum())
    spec[chip_gaps] = interpolate.splev(waves[chip_gaps], intpr)

    not_telluric = telluric_mask(waves)
    # Smooth the spectrum so that the spline doesn't go as crazy
    # Use the Savitzky-Golay filter to presevere the edges of the
    # absorption features (both atomospheric and intrinsic to the star)
    sgspec = signal.savgol_filter(spec, 31, 3)
    # Get the number of data points to set the smoothing criteria for the
    # spline
    m = not_telluric.sum()
    intpr = interpolate.splrep(waves[not_telluric], sgspec[not_telluric],
                               w=1 / noise[not_telluric], k=2, s=20 * m)

    # Replace the telluric with the smoothed function
    smoothedspec = interpolate.splev(waves, intpr)

    # Extrapolate the ends linearly
    # Blue side
    w = np.logical_and(waves > 3420, waves < 3600)
    bluefit = np.poly1d(np.polyfit(waves[w], spec[w], 1))
    bluewaves = waves < 3420
    smoothedspec[bluewaves] = bluefit(waves[bluewaves])

    # Red side
    w = np.logical_and(waves > 8410, waves < 8800)
    redfit = np.poly1d(np.polyfit(waves[w], spec[w], 1))
    redwaves = waves > 8800
    smoothedspec[redwaves] = redfit(waves[redwaves])
    smoothedspec[not_telluric] = spec[not_telluric]
    # Divide the original and the telluric corrected spectra to
    # get the correction factor
    correction = spec / smoothedspec

    airmass = float(hdr['AIRMASS'])
    correction = correction ** (airmass ** -0.55)
    # Save the correction
    dout = np.ones((2, len(waves)))
    dout[0] = waves
    dout[1] = correction
    np.savetxt('telcor.dat', dout.transpose())


def telluric_mask(waves):
    # True where not telluric contaminated
    not_telluric = np.ones(waves.shape, dtype=np.bool)
    for wavereg in telluricWaves:
        in_telluric_region = np.logical_and(waves >= wavereg[0],
                                            waves <= wavereg[1])
        not_telluric = np.logical_and(not_telluric,
                                         np.logical_not(in_telluric_region))
    return not_telluric