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
