"""
Welcome to the ABS_library.

This library contains all generic functions
required for the MFE230M HWs.

8/27/2016 -- Finalized for the HW1 part.

8/17/2016 -- Initiation of the library.
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
                 kappa=0.1, sig=0.1, short_rate=0.05):
        """Initialization of the class."""
        self.kappa = kappa
        self.sig = sig
        self.FwdRate = FwdRate
        self.FwdRatePartial = FwdRatePartial
        self.short_rate = short_rate

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

    def setShortRate(self, short_rate):
        """Set short rate."""
        self.short_rate = short_rate

    def getOptionVol(self, T_O, T_B, t=0):
        """
        Calculate the option volatility.

        Parameters:
        T_O: initiation of the option
        T_B: termination of the option
        t  : current time
        """
        _part1 = self.sig**2*(1-np.exp(-2*self.kappa*(T_O-t)))/2/self.kappa
        _part2 = np.power(1-np.exp(-self.kappa*(T_B-T_O)), 2)/self.kappa**2
        return np.power(_part1 * _part2, 0.5)

    def getA(self, t, T):
        """Get A."""
        if(type(T) != float and type(T) != int):
            _term1 = np.array([-scipy.integrate.quad(
                lambda tau: self.getB(tau, _T) * self.getTheta(tau), t, _T)[0]
                for _T in T])
        else:
            _term1 = -scipy.integrate.quad(
                lambda tau: self.getB(tau, T) * self.getTheta(tau), t, T)[0]

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

    def getZ(self, t=0):
        """Return a function for discount Curve."""
        return lambda T: np.exp(self.getA(t, T) -
                                self.getB(t, T) * self.short_rate)

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

    def getSpotRate(self, t=0):
        """Return a function of LIBOR spot rate"""
        def __wrapper(_tau):
            __part1 = -self.getA(t, t+_tau)/_tau
            __part2 = (1-np.exp(-self.kappa*_tau))/_tau /\
                self.kappa*self.short_rate
            return __part1 + __part2
        return __wrapper


class simpleREMIC:
    """The simple REMIC class."""

    def __init__(self, wac, warm, wala=0, psa=100, init_principle=100):
        """Initiation of the object."""
        # sanity check
        assert type(warm) == int, \
            "Input WAC should be an integer, not {}!".format(wac)

        # dump in the variables
        self.wac = wac
        self.warm = warm
        self.wala = wala
        self.psa = psa
        self.init_principle = init_principle
        self.CPR = lambda i: self.psa/100*0.06*np.minimum(1, (i+self.wala)/30)
        self.SMM = lambda i: 1 - np.power(1-self.CPR(i), 1/12.)

    def setPSA(self, psa):
        """Set PSA."""
        self.psa = psa

    def setCPR(self, CPR):
        """Set CPR."""
        self.CPR = CPR

    def setSMM(self, SMM):
        """Set SMM."""
        self.SMM = SMM

    def getCashFlows(self, fills=-1):
        """
        Get cashflows for each month.

        Parameters:
        fells = size of the output array of each cashflows,
                default to the warm

        return 3 cashflows: interests, principals, and prepayments
        """
        if (fills == -1):
            fills = self.warm

        _pmts = np.zeros(fills)
        _ints = np.zeros(fills)
        _prins = np.zeros(fills)
        _prepays = np.zeros(fills)
        _balance = self.init_principle
        _c = self.wac/12

        for i in range(self.warm):
            _n = self.warm - i
            _pmts[i] = _balance*_c*(1+_c)**_n/((1+_c)**_n-1)
            _ints[i] = _c*_balance
            _prins[i] = _pmts[i] - _ints[i]
            _prepays[i] = (_balance - _prins[i])*self.SMM(i+1)
            _balance -= _prepays[i] + _prins[i]

        return _ints, _prins, _prepays


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
    # _f = (getZ(T-dt) / getZ(T) - 1) / dt
    _f = -np.log(getZ(T)/getZ(T-dt))/dt

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

    return _M*(_K*getZ(T-dt)*scipy.stats.norm.cdf(-_d2) -
               getZ(T)*scipy.stats.norm.cdf(-_d1))


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

    _mean = _path_results.mean()  # mean
    _se = scipy.stats.sem(_path_results)  # standard error

    # duration
    _duration = -(_path_results_up.mean()-_path_results_down.mean()) / \
        (_mean*2*shock)

    # convexity
    _convexity = (_path_results_up.mean() +
                  _path_results_down.mean() - 2*_mean) / \
        (_mean*shock**2)

    return [_mean, _se, _duration, _convexity]


def getOAS(CFs, Rs, dt, par):
    """
    Calculate the OAS.

    Parameters:
    CFs : undiscounted cash flow of a security
    Rs  : an array of short rates, same dimension
          as CFs
    par : par value of the CFs
    """
    _diff = lambda _oas: abs(np.exp(-np.cumsum(Rs+_oas)*dt).dot(CFs)-par)
    oas = scipy.optimize.fmin(_diff, 0, disp=False)
    return oas[0]


def getGroupTwoCFs(pool_cf, CA_init, CY_init, coupon):
    """Very Tailored function to get the CFs of group 2."""
    CA_b = np.maximum(CA_init - np.append(0, np.cumsum(pool_cf))[:-1],
                      np.zeros(len(pool_cf)))
    CA_p = np.minimum(CA_b, pool_cf)
    CA_i = CA_b*coupon

    CY_b = np.maximum(CY_init - np.append(0, np.cumsum(pool_cf - CA_p))[:-1],
                      np.zeros(len(pool_cf)))
    CY_p = np.minimum(pool_cf - CA_p, pool_cf)
    CY_i = CY_b*coupon

    return CA_p + CA_i, CY_p + CY_i  # return total cash flow of each asset


def getGroupOneCFs(pool_cf, CG_init, VE_init, CM_init, GZ_init, TC_init,
                   CZ_init, coupon, tranche_coupon):
    """Very Tailored function to get the CFs of group 1."""
    _n = len(pool_cf)
    CG_b = np.zeros(_n)
    VE_b = np.zeros(_n)
    CM_b = np.zeros(_n)
    GZ_b = np.zeros(_n)
    TC_b = np.zeros(_n)
    CZ_b = np.zeros(_n)
    CG_i = np.zeros(_n)
    VE_i = np.zeros(_n)
    CM_i = np.zeros(_n)
    GZ_i = np.zeros(_n)
    TC_i = np.zeros(_n)
    CZ_i = np.zeros(_n)
    CG_p = np.zeros(_n)
    VE_p = np.zeros(_n)
    CM_p = np.zeros(_n)
    GZ_p = np.zeros(_n)
    TC_p = np.zeros(_n)
    CZ_p = np.zeros(_n)
    GZ_ti = np.zeros(_n)  # tranche coupon
    CZ_ti = np.zeros(_n)  # tranche coupon
    GZ_a = np.zeros(_n)
    CZ_a = np.zeros(_n)

    # set the initial balance:
    CG_b[0] = CG_init
    VE_b[0] = VE_init
    CM_b[0] = CM_init
    GZ_b[0] = GZ_init
    TC_b[0] = TC_init
    CZ_b[0] = CZ_init

    for i in range(_n):
        # interest in from tranche coupon rate
        CZ_ti[i] = CZ_b[i]*tranche_coupon
        GZ_ti[i] = GZ_b[i]*tranche_coupon

        CG_p[i] = max(0, min(pool_cf[i] + CZ_ti[i], CG_b[i]))
        VE_p[i] = max(0, min(pool_cf[i] + GZ_ti[i] + CZ_ti[i] -
                             CG_p[i], VE_b[i]))
        CM_p[i] = max(0, min(pool_cf[i] + GZ_ti[i] + CZ_ti[i] -
                             CG_p[i] - VE_p[i], CM_b[i]))
        _CM_b_next = CM_b[i] - CM_p[i]

        GZ_a[i] = GZ_ti[i] if _CM_b_next > 0 else min(CM_p[i], GZ_ti[i])
        GZ_p[i] = max(0, min(pool_cf[i] + GZ_a[i] + CZ_ti[i] -
                             CG_p[i] - VE_p[i] - CM_p[i], GZ_b[i]))
        _GZ_b_next = GZ_b[i] + GZ_a[i] - GZ_p[i]

        TC_p[i] = 0 if _GZ_b_next > 0 \
            else min(pool_cf[i] + CZ_ti[i] - GZ_p[i], TC_b[i])
        _TC_b_next = TC_b[i] - TC_p[i]

        CZ_a[i] = CZ_ti[i] if _TC_b_next > 0 else min(TC_p[i], CZ_ti[i])
        CZ_p[i] = max(0, min(pool_cf[i] + CZ_a[i] - CG_p[i] - VE_p[i] -
                             CM_p[i] - GZ_p[i] - TC_p[i], CZ_b[i]))

        # interest batch
        CG_i[i] = CG_b[i]*coupon
        VE_i[i] = VE_b[i]*coupon
        CM_i[i] = CM_b[i]*coupon
        GZ_i[i] = GZ_ti[i] - GZ_a[i]
        TC_i[i] = TC_b[i]*coupon
        CZ_i[i] = CZ_ti[i] - CZ_a[i]

        if(i < _n-1):  # we don't update the last period's balance
            CG_b[i+1] = CG_b[i] - CG_p[i]
            VE_b[i+1] = VE_b[i] - VE_p[i]
            CM_b[i+1] = _CM_b_next
            GZ_b[i+1] = _GZ_b_next
            TC_b[i+1] = _TC_b_next
            CZ_b[i+1] = CZ_b[i] + CZ_a[i] - CZ_p[i]

    return CG_p+CG_i, VE_p+VE_i, CM_p+CM_i, GZ_p+GZ_i, TC_p+TC_i, CZ_p+CZ_i


""""
Below are the new functions for HW2.
"""


def hazardRateFactory(gamma, p, b1, b2, v1, v2):
    """Return a tailored Hazard rate function."""
    def __wrapper(_t):
        __part1 = gamma*p*np.power(gamma*_t, p-1)/(1+np.power(gamma*_t, p))
        __part2 = np.exp(b1*v1(_t) + b2*v2(_t))
        return __part1 * __part2
    return __wrapper
