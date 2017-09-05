from lcogtgemini import dodq, fitshdr_to_wave, bluecut, get_chipedges, magtoflux, fluxtomag, \
    sanitizeheader, do_qecorr, tofits
from lcogtgemini.utils import boxcar_smooth
from lcogtgemini.file_utils import getredorblue


def calibrate(scifiles, extfile, observatory):
    for f in scifiles:
        redorblue = getredorblue(f)
        iraf.unlearn(iraf.gscalibrate)
        iraf.gscalibrate('et' + f[:-4] + '.fits',
                         sfunc='sens' + redorblue + '.fits', fl_ext=True, fl_vardq=dodq,
                         extinction=extfile, observatory=observatory)

        if os.path.exists('cet' + f[:-4] + '.fits'):
            iraf.unlearn(iraf.splot)
            iraf.splot('cet' + f.replace('.txt', '.fits') + '[sci]')  # just to check


def makesensfunc(scifiles, objname, base_stddir, extfile):
    #TODO use individual standard star observations in each setting, not just red and blue
    for f in scifiles:
        redorblue = getredorblue(f)
        # If this is a standard star, run standard
        # Standards will have an observation class of either progCal or partnerCal
        # Standards will have an observation class of either progCal or partnerCal
        obsclass = fits.getval(f[:-4] + '.fits', 'OBSCLASS')
        if obsclass == 'progCal' or obsclass == 'partnerCal':
            # Figure out which directory the stardard star is in
            stddir = iraf.osfn('gmisc$lib/onedstds/') + base_stddir

            # iraf.gsstandard('est' + f[:-4], 'std' + redorblue,
            #                'sens' + redorblue, starname=objname.lower(),
            #                caldir='gmisc$lib/onedstds/'+stddir, fl_inter=True)

            specsens('et' + f[:-4] + '.fits', 'sens' + redorblue + '.fits',
                     stddir + objname + '.dat' , extfile,
                     float(fits.getval(f[:-4] + '.fits', 'AIRMASS')),
                     float(fits.getval(f[:-4] + '.fits', 'EXPTIME')))


def specsens(specfile, outfile, stdfile, extfile, airmass=None, exptime=None,
             stdzp=3.68e-20, thresh=8, clobber=True):

    # read in the specfile and create a spectrum object
    obs_hdu = fits.open(specfile)
    try:
        obs_flux = obs_hdu[2].data.copy()[0]
        obs_hdr = obs_hdu[2].header.copy()
    except:
        obs_flux = obs_hdu[0].data.copy()
        obs_hdr = obs_hdu[0].header.copy()
    obs_hdu.close()
    obs_wave = fitshdr_to_wave(obs_hdr)

    # Mask out everything below 3450 where there is no signal
    obs_flux = obs_flux[obs_wave >= bluecut]
    obs_wave = obs_wave[obs_wave >= bluecut]

    # Figure out where the chip gaps are
    chip_edges = get_chipedges(obs_flux)

    try:
        chip_gaps = np.ones(obs_flux.size, dtype=np.bool)
        for edge in chip_edges:
            chip_gaps[edge[0]: edge[1]] = False
    except:
        chip_gaps = np.zeros(obs_flux.size, dtype=np.bool)

    template_spectrum = signal.savgol_filter(obs_flux, 21, 3)
    noise = np.abs(obs_flux - template_spectrum)
    noise = ndimage.filters.gaussian_filter1d(noise, 100.0)

    if chip_gaps.sum() != len(chip_gaps):
        # Smooth the chip gaps
        intpr = interpolate.splrep(obs_wave[np.logical_not(chip_gaps)],
                                   obs_flux[np.logical_not(chip_gaps)],
                                   w=1 / noise[np.logical_not(chip_gaps)], k=2,
                                   s=20 * np.logical_not(chip_gaps).sum())
        obs_flux[chip_gaps] = interpolate.splev(obs_wave[chip_gaps], intpr)
    # smooth the observed spectrum
    # read in the std file and convert from magnitudes to fnu
    # then convert it to fwave (ergs/s/cm2/A)
    std_wave, std_mag, _stdbnd = np.genfromtxt(stdfile).transpose()
    std_flux = magtoflux(std_wave, std_mag, stdzp)

    # Get the typical bandpass of the standard star,
    std_bandpass = np.max([50.0, np.diff(std_wave).mean()])
    # Smooth the observed spectrum to that bandpass
    obs_flux = boxcar_smooth(obs_wave, obs_flux, std_bandpass)
    # read in the extinction file (leave in magnitudes)
    ext_wave, ext_mag = np.genfromtxt(extfile).transpose()

    # calculate the calibrated spectra
    cal_flux = cal_std(obs_wave, obs_flux, std_wave, std_flux, ext_wave,
                             ext_mag, airmass, exptime)

    # Normalize the fit variables so the fit is well behaved
    fitme_x = (obs_wave - obs_wave.min()) / (obs_wave.max() - obs_wave.min())
    fitme_y = cal_flux / np.median(cal_flux)
    coeffs = pfm.pffit(fitme_x, fitme_y, 5 , 7, robust=True,
                    M=sm.robust.norms.AndrewWave())

    fitted_flux = pfm.pfcalc(coeffs, fitme_x) * np.median(cal_flux)

    cal_mag = -1.0 * fluxtomag(fitted_flux)
    # write the spectra out
    cal_hdr = sanitizeheader(obs_hdr.copy())
    cal_hdr['OBJECT'] = 'Sensitivity function for all apertures'
    cal_hdr['CRVAL1'] = obs_wave.min()
    cal_hdr['CRPIX1'] = 1
    if do_qecorr:
        cal_hdr['QESTATE'] = True
    else:
        cal_hdr['QESTATE'] = False

    tofits(outfile, cal_mag, hdr=cal_hdr, clobber=True)


def cal_std(obs_wave, obs_flux, std_wave, std_flux, ext_wave, ext_mag, airmass, exptime):
    """Given an observe spectra, calculate the calibration curve for the
       spectra.  All data is interpolated to the binning of the obs_spectra.
       The calibrated spectra is then calculated from
       C =  F_obs/ F_std / 10**(-0.4*A*E)/T/dW
       where F_obs is the observed flux from the source,  F_std  is the
       standard spectra, A is the airmass, E is the
       extinction in mags, T is the exposure time and dW is the bandpass
    """

    # re-interpt the std_spectra over the same wavelength
    std_flux = np.interp(obs_wave, std_wave, std_flux)

    # re-interp the ext_spetra over the same wavelength
    ext_mag = np.interp(obs_wave, ext_wave, ext_mag)

    # create the calibration spectra
    # set up the bandpass
    bandpass = np.diff(obs_wave).mean()

    # correct for extinction
    cal_flux = obs_flux / 10 ** (-0.4 * airmass * ext_mag)

    # correct for the exposure time and calculation the sensitivity curve
    cal_flux = cal_flux / exptime / bandpass / std_flux

    return cal_flux