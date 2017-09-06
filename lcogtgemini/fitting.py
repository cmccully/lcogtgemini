import numpy as np
import astropy.modeling
from scipy import optimize
from statsmodels import robust
from lcogtgemini.utils import mad

def ncor(x, y):
    """Calculate the normalized correlation of two arrays"""
    d = np.correlate(x, x) * np.correlate(y, y)

    return np.correlate(x, y) / d ** 0.5


def xcorfun(p, warr, farr, telwarr, telfarr):
    # Telluric wavelengths and flux
    # Observed wavelengths and flux
    # resample the telluric spectrum at the same wavelengths as the observed
    # spectrum
    # Make the artifical spectrum to cross correlate
    asfarr = np.interp(warr, p[0] * telwarr + p[1], telfarr, left=1.0, right=1.0)
    return np.abs(1.0 / ncor(farr, asfarr))

def normalize_fitting_coordinate(x):
    xrange = x.max() - x.min()
    return (x - x.min()) / xrange


class offset_left_model(astropy.modeling.Fittable1DModel):

    cutoff = astropy.modeling.Parameter(default=0)
    scale = astropy.modeling.Parameter(default=1)

    c0 = astropy.modeling.Parameter(default=1)
    c1 = astropy.modeling.Parameter(default=0)
    c2 = astropy.modeling.Parameter(default=0)
    c3 = astropy.modeling.Parameter(default=0)


    @staticmethod
    def evaluate(x, cutoff, scale, c0, c1, c2, c3):
        y = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3
        y[x <= cutoff] *= scale

        return y


class offset_right_model(astropy.modeling.Fittable1DModel):

    cutoff = astropy.modeling.Parameter(default=0)
    scale = astropy.modeling.Parameter(default=1)

    c0 = astropy.modeling.Parameter(default=1)
    c1 = astropy.modeling.Parameter(default=0)
    c2 = astropy.modeling.Parameter(default=0)
    c3 = astropy.modeling.Parameter(default=0)


    @staticmethod
    def evaluate(x, cutoff, scale, c0, c1, c2, c3):
        y = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3
        y[x >= cutoff] *= scale

        return y


class blackbody_model(astropy.modeling.Fittable1DModel):

    temperature = astropy.modeling.Parameter(default=10000)
    normalization = astropy.modeling.Parameter(default=1)

    @staticmethod
    def evaluate(x, temperature=10000., normalization=1.0):
        # Note x needs to be in microns and temperature needs to be in K
        flam = normalization * x ** -5
        flam /= np.exp(14387.7696 / x / temperature) - 1
        return flam


# Iterative reweighting linear least squares
def irls(x, data, errors, model, tol=1e-6, M=robust.norms.AndrewWave(2.0), maxiter=10):
    fitter = fitting.LevMarLSQFitter()

    if x is None:
        # Make x and y arrays out of the indicies
        x = np.indices(data.shape, dtype=np.float)

        if len(data.shape) == 2:
            y, x = x
        else:
            x = x[0]

        #Normalize to make fitting easier
        x = normalize_fitting_coordinate(x)
        if len(data.shape) == 2:
            y = normalize_fitting_coordinate(y)

    scatter = errors
    # Do an initial fit of the model
    # Use 1 / sigma^2 as weights
    weights = (errors ** -2.0).flatten()

    if len(data.shape) == 2:
        fitted_model = fitter(model, x, y, data, weights=weights)
    else:
        fitted_model = fitter(model, x, data, weights=weights)

    notconverged=True
    last_chi = np.inf
    iter = 0
    # Until converged
    while notconverged:
        # Update the weights
        if len(data.shape) == 2:
            residuals = data - fitted_model(x, y)
        else:
            residuals = data - fitted_model(x)
        # Save the chi^2 to check for convergence
        chi = ((residuals / scatter) ** 2.0).sum()

        # update the scaling (the MAD of the residuals)
        scatter = mad(residuals)  * 1.4826 # To convert to standard deviation
        weights = M.weights(residuals / scatter).flatten()

        # refit
        if len(data.shape) == 2:
            fitted_model = fitter(model, x, y, data, weights=weights)
        else:
            fitted_model = fitter(model, x, data, weights=weights)
        # converged when the change in the chi^2 (or l2 norm or whatever) is
        # less than the tolerance. Hopefully this should converge quickly.
        if iter >= maxiter or np.abs(chi - last_chi) < tol:
            notconverged = False
        else:
            last_chi = chi
            iter += 1

    return fitted_model


def fitxcor(warr, farr, telwarr, telfarr):
    """Maximize the normalized cross correlation coefficient for the telluric
    correction
    """
    res = optimize.minimize(xcorfun, [1.0, 0.0], method='Nelder-Mead',
                   args=(warr, farr, telwarr, telfarr))

    return res['x']


def fit_pfm(x, y):
    x = np.arange(len(data), dtype=np.float)
    x /= x.max()

    y = data / np.median(data)

    fitme_x = x[chip_edges[0][0]:chip_edges[0][1]]
    fitme_x = np.append(fitme_x, x[chip_edges[1][0]:chip_edges[1][1]])
    fitme_x = np.append(fitme_x, x[chip_edges[2][0]:chip_edges[2][1]])

    fitme_y = y[chip_edges[0][0]:chip_edges[0][1]]
    fitme_y = np.append(fitme_y, y[chip_edges[1][0]:chip_edges[1][1]])
    fitme_y = np.append(fitme_y, y[chip_edges[2][0]:chip_edges[2][1]])

    best_fit = fit_pfm(x, y, 21, 7, 1.0, interactive=True)
    fit = pfm.pffit(fitme_x, fitme_y, 21, 7, robust=True,
                    M=robust.norms.AndrewWave())
    if plot:
        pyplot.ion()
        pyplot.clf()
        pyplot.plot(x, y)
        pyplot.plot(x, pfm.pfcalc(fit, x))
        _junk = raw_input('Press enter to continue')