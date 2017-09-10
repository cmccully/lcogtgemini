import numpy as np
import astropy.modeling
from scipy import optimize
from statsmodels import robust
from lcogtgemini.utils import mad
from matplotlib import pyplot


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
    x_range = x.max() - x.min()
    return (x - x.min()) / x_range, x.min(), x_range


# Iterative reweighting linear least squares
def irls(x, data, errors, model_function, initial_parameter_guess, tol=1e-6, weight_function=robust.norms.AndrewWave,
         weight_scale=2.0, maxiter=10):
    weights_calculator = weight_function(weight_scale)

    #Normalize to make fitting easier
    normalized_x, xmin, x_range = normalize_fitting_coordinate(x)

    y_scale =  np.median(data)
    y = data / y_scale
    scatter = errors / y_scale

    # Do an initial fit of the model
    # Use 1 / sigma^2 as weights
    best_parameters = optimize.curve_fit(model_function, normalized_x, y, p0=initial_parameter_guess, sigma=scatter)[0]


    notconverged=True
    last_chi2 = np.inf
    iter = 0
    # Until converged
    while notconverged:
        # Update the weights
        residuals = y - model_function(normalized_x, *best_parameters)
        # Save the chi^2 to check for convergence
        chi2 = ((residuals / scatter) ** 2.0).sum()

        # update the scaling (the MAD of the residuals)
        scatter = mad(residuals)  * 1.4826 # To convert to standard deviation
        weights = weights_calculator.weights(residuals / scatter).flatten()
        fit_errors = np.zeros(weights.shape)
        fit_errors[weights > 0] = weights[weights > 0] ** -2.0
        fit_errors[weights == 0] = np.inf
        # refit
        best_parameters = optimize.curve_fit(model_function, normalized_x, y,
                                             p0=best_parameters, sigma=fit_errors)[0]

        # converged when the change in the chi^2 (or l2 norm or whatever) is
        # less than the tolerance. Hopefully this should converge quickly.
        if iter >= maxiter or np.abs(chi2 - last_chi2) < tol:
            notconverged = False
        else:
            last_chi2 = chi2
            iter += 1

    return {'popt': best_parameters, 'y_scale': y_scale, 'xmin': xmin, 'x_range': x_range,
            'model_function': model_function}


def eval_fit(fit_dict, x):
    x_to_eval = (x - fit_dict['xmin']) / fit_dict['x_range']
    return fit_dict['model_function'](x_to_eval, *fit_dict['popt']) * fit_dict['y_scale']


def polynomial_fourier_model(n_poly, n_fourier):
    def model_to_optimize(x, *p):
        y = p[0]
        for i in range(1, n_poly + 1):
            y += p[i] * x ** i
        # Note this assumes that x is roughly normalized between 0 and 1
        omega_t = 2.0 * np.pi * x
        for i in range(1, n_fourier + 1):
            y += p[2 * i - 1 + n_poly] * np.sin(i * omega_t)
            y += p[2 * i + n_poly] * np.cos(i * omega_t)
        return y
    return model_to_optimize


def run_polynomal_fourier_fit(x, y, errors, n_poly, n_fourier, weight_scale):
    function_to_fit = polynomial_fourier_model(n_poly, n_fourier)
    p0 = np.zeros(1 + n_poly + 2 * n_fourier)
    p0[0] = 1.0

    best_fit = run_fit(x, y, errors, function_to_fit, p0, weight_scale)

    return best_fit

def run_fit(x, y, errors, function_to_fit, p0, weight_scale):
    # Run IRLS on the data given the input parameters
    best_fit = irls(x, y, errors, function_to_fit, p0, weight_scale=weight_scale)

    # Plot the best fit
    plot_best_fit(x, y, best_fit)

    return best_fit

def fit_polynomial_fourier_model(x, y, errors, n_poly, n_fourier, weight_scale=2.0):

    best_fit = run_polynomal_fourier_fit(x, y, errors, n_poly, n_fourier, weight_scale)
    # Go into a while loop
    while True:
        # Ask if the user is not happy with the fit,
        response = user_input('Does this fit look good? y or n:', ['y', 'n'], 'y')
        if response == 'y':
            break
        # If not have the user put in new values
        else:
            n_poly = int(user_input('Order of polynomial to fit:', [str(i) for i in range(100)], n_poly))
            n_fourier = int(user_input('Order of Fourier terms to fit:', [str(i) for i in range(100)], n_fourier))
            weight_scale = user_input('Scale for outlier rejection:', default=weight_scale, is_number=True)
            best_fit = run_polynomal_fourier_fit(x, y, errors, n_poly, n_fourier, weight_scale)

    return best_fit


def user_input(prompt, choices=None, default=None, is_number=False):
    while True:
        response = raw_input(prompt + ' [{i}]'.format(i=default))
        if len(response) == 0:
            response = default
            break
        if choices is not None and response in choices:
            break
        elif is_number:
            try:
                response = float(response)
                break
            except:
                print('Input could not be parsed into a number. Please try again.')
        else:
            print('Please select a valid response')
    return response


def plot_best_fit(x, y, best_fit):
    fig = pyplot.gcf()
    fig.clf()
    axes = fig.get_axes()
    if not axes:
        pyplot.subplot(211)
        pyplot.subplot(212)

    axes[0].plot(x, y, 'b')
    y_model = eval_fit(best_fit, x)
    axes[0].plot(x, y_model, 'r')
    axes[1].plot(x, y - y_model, 'o')


def fitxcor(warr, farr, telwarr, telfarr):
    """Maximize the normalized cross correlation coefficient for the telluric
    correction
    """
    res = optimize.minimize(xcorfun, [1.0, 0.0], method='Nelder-Mead',
                   args=(warr, farr, telwarr, telfarr))

    return res['x']
