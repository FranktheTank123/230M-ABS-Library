"""
Welcome to the ABS_library.

This library contains all generic functions
required for the MFE230M HWs.

7/17/2016 -- Initiation of the library
"""

import numpy as np
from numpy.linalg import inv
import scipy.stats


class OLS:
    """DIY OSL class."""

    def __init__(self, X, y):
        """
        Initialtion of the class.

        All parameters were calculated during initiation

        Parameters:
        x - np.ndarray
        y - np.array
        """
        self.X = X
        self.y = y
        self.beta = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.reg_y = np.dot(X, self.beta)


class hullWhite:
    """The Hull White model class."""

    def __init__(self, FwdRate=lambda T: T,
                 FwdRatePartial=lambda T: T,
                 kappa=0.1, sig=0.1):
        """Initialization of the class."""
        self.kappa = kappa
        self.sig = sig
        self.FwdRate = FwdRate
        self.FwdRatePartial = FwdRatePartial

    def setKappa(self, kappa):
        """Set Kappa."""
        self.kappa = kappa

    def setSigma(self, sig):
        """Set Sigma."""
        self.sig = sig

    def setFwdRate(self, FwdRate):
        """Set Fwd Rate."""
        self.FwdRate = FwdRate

    def setFwdRatePartial(self, FwdRatePartial):
        """Set Fwd Rate Partial."""
        self.FwdRatePartial = FwdRatePartial

    def getOptionVol(self, T_O, T_B, t=0):
        """
        Calculate the option volatility.

        Parameters:
        T_O: initiation of the option
        T_B: termination of the option
        t  : current time
        """
        _part1 = self.sig**2*(1-np.exp(-2*self.kappa*(T_O-t)))/2/self.kappa
        _part2 = np.power(1-np.exp(self.kappa*(T_B-T_O)), 2)/self.kappa**2

        return np.power(_part1 * _part2, 0.5)

    def getA(self, t, T):
        """Get A."""
        _term1 = -scipy.integrate.quad(lambda tau: self.getB(tau, T) *
                                       self.getTheta(tau), t, T)[0]
        _term2 = self.sig**2/2/self.kappa**2 * \
            (T-t+(1-np.exp(-2*self.kappa*(T-t)))/2/self.kappa -
             2 * self.getB(t, T))

        return _term1 + _term2

    def getB(self, t, T):
        """Get B."""
        return (1-np.exp(-self.kappa*(T-t)))/self.kappa

    def getPartialA(self, T):
        """Get Partial A."""
        _term1 = -scipy.integrate.quad(lambda t:
                                       self.getTheta(t) *
                                       np.exp(-self.kappa*(T-t)), 0, T)[0]
        _term2 = self.sig**2 * \
            scipy.integrate.quad(lambda t: self.getB(t, T) *
                                 np.exp(-self.kappa*(T-t)), 0, T)[0]

        return _term1 + _term2

    def getTheta(self, T):
        """Get Theta."""
        return self.FwdRatePartial(T) + self.kappa*self.FwdRate(T) + \
            self.sig**2/2./self.kappa * (1-np.exp(-2*self.kappa*T))

    def getMC(self, r_t, T, t=0):
        """
        Return a function for Monte Carlo simulations.

        parameters --
        r_t: initial starting short rate
        T:  terminal time
        t: staring time, by defult = 0
        """
        def __wrapper(_z):  # _z should be 1d array of normal random variable
            __dt = (T-t)/len(_z)
            __time_stpes = np.linspace(t, T, len(_z)+1)  # time space
            __r_s = np.repeat(r_t, len(_z)+1)  # r_t space

            for _i, _z_i in enumerate(_z):
                __r_s[_i+1] = __r_s[_i]+(self.getTheta(__time_stpes[_i+1]) -
                                         self.kappa*__r_s[_i]) * \
                        __dt + self.sig * __dt**0.5 * _z_i
            return __r_s

        return __wrapper


def basisExpansion(data, power):
    """
    Expand the basis into its powers.

    e.g. when power = 2, will return [data, data**2]
    """
    assert type(power) == int, "Oops, the power variable should be an integer"

    _n = len(data)
    results = np.empty(_n*power).reshape(_n, power)

    for i in range(power):
        results[:, i] = np.power(data, i+1)
    return results


def capletVolToPrice(sig, T, dt, getZ, r_k, N=1):
    """
    Get the caplet price from the implied forward vol using Black's formula.

    Parameters:
    sig : implied forward volatility
    T   : maturity in years
    dt  : forward perioed in years
    getZ: a function which return the discount rate1
    r_k : strike rate
    N   : notional, default to be 1
    """
    # get the forward rate
    _f = (getZ(T-dt) / getZ(T) - 1) / dt

    _d1 = np.log(_f / r_k)/sig/np.sqrt(T - dt) \
        + 0.5*sig*np.sqrt(T-dt)
    _d2 = _d1 - sig * np.sqrt(T-dt)

    return N*dt*getZ(T)*(_f*scipy.stats.norm.cdf(_d1) -
                         r_k*scipy.stats.norm.cdf(_d2))


def calCapletPrice(sig, T, dt, getZ, r_k, N=1):
    """
    Get the caplet price under a normal spot rate model.

    The model can be, say Ho-Lee or Hull-White or Vasicek
    the input sig should be pre-modified for options

    Parameters:
    sig : spot rate's option volatility
    T   : maturity in years
    dt  : forward perioed in years
    getZ: a function which return the discount rate1
    r_k : strike rate
    N   : notional, default to be 1
    """
    _M = N*(1 + r_k*dt)
    _K = 1./(1 + r_k*dt)

    _d1 = np.log(getZ(T)/_K/getZ(T-dt))/sig + sig/2.
    _d2 = _d1 - sig

    return _M*(_K*getZ(T-dt)*scipy.stats.norm.cdf(_d1) -
               getZ(T)*scipy.stats.norm.cdf(_d2))


def multiStepMC(z, price_evolution, anti=False,
                tracker=lambda S_ts: S_ts):
    """
    A very generic version of multi-stps Monte Carlo.

    Assumptions:
    - THE STEPS IS DETERMINED BY THE DIMENSION OF Z (which is a np.ndarray)
    - Equally spaced time stpes

    Parameters:
    z:               A m by n numpy matrix. We assume m is the number of
                     simulations and n is the time stpes

    price_evolution: A function that takes a 1d array of Z slice and
                     returns a 1d array (+1 size to include initial point)
                     of the evlotion of underlyings which based on the Zs

    tracker:         A function (takes an array of evolution of underlyings)
                     that keep track of features of the price evolution,
                     which could be max/min, or whether a boundary is hitted,

    anti:            whether you want the anti-variate version of the MC.
                     If ture, will ONLY RETURN the ANTI part.

    The function will return a tuple of both the price evolutions and the
    tracked values time series, which are both m by (n+1), which include the
    initial point of the price at t=0.
    """
    if anti:
        z = -z

    # generate the evolution of underlyings for all pathes
    _evolutions = np.apply_along_axis(price_evolution, 1, z)

    return _evolutions, np.apply_along_axis(tracker, 1, _evolutions)


def cashFlowMCSummary(CFs, Zs, Z_ups, Z_downs, shock):
    """
    A very tailored function for HW1 Q2.

    Parameters:
    CFs : undiscounted cash flow of a security
    Zs  : a m by n array of MC simulated discount factors
    Z_ups, Z_downs, shock: shocked Zs and shock

    Will return a list of:
    [Mean, SE, Effective Duration, Convexity]
    """
    # pathwise results
    _path_results = np.apply_along_axis(np.dot, 1, Zs, CFs)
    _path_results_up = np.apply_along_axis(np.dot, 1, Z_ups, CFs)
    _path_results_down = np.apply_along_axis(np.dot, 1, Z_downs, CFs)

    _mean = _path_results.mean()
    _se = scipy.stats.sem(_path_results)

    _duration = (_path_results_up.mean()-_path_results_down.mean()) / \
        (_mean*2*shock)

    _convexity = (_path_results_up.mean() +
                  _path_results_down.mean() - 2*_mean) / \
        (_mean*shock**2)

    return [_mean, _se, _duration, _convexity]
