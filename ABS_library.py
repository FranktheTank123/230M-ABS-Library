"""
Welcome to the ABS_library.

This library contains all generic functions
required for the MFE230M HWs.

9/12/2016 -- add HW3 part

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
        try:
            _term1 = -scipy.integrate.quad(
                lambda tau: self.getB(tau, T) * self.getTheta(tau), t, T)[0]
        except:  # array case
            try:
                _term1 = np.array([-scipy.integrate.quad(
                    lambda tau: self.getB(tau, _T) * self.getTheta(tau), t, _T)[0]
                    for _T in T])
            except: # both t and T are arrays
                _term1 = np.array([-scipy.integrate.quad(
                    lambda tau: self.getB(tau, _T) * self.getTheta(tau), _t, _T)[0]
                    for (_t, _T) in zip(t, T)])

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
        
    def getFutureRate(self, t, r_t):
        """Return a function of future LIBOR rate, given short rates"""
        def __wrapper(_tau):
            __part1 = -self.getA(t, t+_tau)/_tau
            __part2 = (1-np.exp(-self.kappa*_tau))/_tau /\
                self.kappa*r_t
            return __part1 + __part2
        return __wrapper


class simpleREMIC:
    """The simple REMIC class."""

    def __init__(self, wac, warm, wala=0, psa=100, init_principle=100, Npath=1):
        """Initiation of the object."""
        # sanity check
        assert type(warm) == int, \
            "Input WAC should be an integer, not {}!".format(wac)

        # dump in the variables
        self.wac = wac
        self.warm = warm
        self.wala = wala
        self.psa = psa
        self.init_principle = float(init_principle)
        self.CPR = lambda i: self.psa/100*0.06*np.minimum(1, (i+self.wala)/30)
        self.SMM = lambda i: 1 - np.power(1-self.CPR(i), 1/12.)
        self.Npath = Npath

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
        fills = size of the output array of each cashflows,
                default to the warm

        return 3 cashflows: interests, principals, and prepayments
        """
        if (fills == -1):
            fills = self.warm

        Npath = self.Npath
        _pmts = np.squeeze(np.zeros((Npath, fills)))
        _ints = np.squeeze(np.zeros((Npath, fills)))
        _prins = np.squeeze(np.zeros((Npath, fills)))
        _prepays = np.squeeze(np.zeros((Npath, fills)))
        _balance = np.repeat(self.init_principle, Npath)
        _c = self.wac/12

        for i in range(min(self.warm, fills)):
            _n = self.warm - i
            _pmts[:, i] = _balance*_c*(1+_c)**_n/((1+_c)**_n-1)
            _ints[:, i] = _c*_balance
            _prins[:, i] = _pmts[:, i] - _ints[:, i]
            _prepays[:, i] = (_balance - _prins[:, i])*self.SMM(i+1)
            _balance -= _prepays[:, i] + _prins[:, i]

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
    _x, _y = pool_cf.shape
    CA_b = np.maximum(0, CA_init - np.append(np.zeros((_x, 1)),
                                             np.cumsum(pool_cf, axis=1), axis=1)[:,:-1])
    CA_p = np.minimum(CA_b, pool_cf)
    CA_i = CA_b*coupon

    CY_b = np.maximum(0, CY_init - np.append(np.zeros((_x, 1)),
                                             np.cumsum(pool_cf - CA_p, axis=1), axis=1)[:,:-1])
    CY_p = np.minimum(pool_cf - CA_p, pool_cf)
    CY_i = CY_b*coupon

    return CA_p + CA_i, CY_p + CY_i  # return total cash flow of each asset


def getGroupOneCFs(pool_cf, CG_init, VE_init, CM_init, GZ_init, TC_init,
                   CZ_init, coupon, tranche_coupon):
    """Very Tailored function to get the CFs of group 1."""
    _Npath, _n = pool_cf.shape
    CG_b = np.zeros((_Npath, _n))
    VE_b = np.zeros((_Npath, _n))
    CM_b = np.zeros((_Npath, _n))
    GZ_b = np.zeros((_Npath, _n))
    TC_b = np.zeros((_Npath, _n))
    CZ_b = np.zeros((_Npath, _n))
    CG_i = np.zeros((_Npath, _n))
    VE_i = np.zeros((_Npath, _n))
    CM_i = np.zeros((_Npath, _n))
    GZ_i = np.zeros((_Npath, _n))
    TC_i = np.zeros((_Npath, _n))
    CZ_i = np.zeros((_Npath, _n))
    CG_p = np.zeros((_Npath, _n))
    VE_p = np.zeros((_Npath, _n))
    CM_p = np.zeros((_Npath, _n))
    GZ_p = np.zeros((_Npath, _n))
    TC_p = np.zeros((_Npath, _n))
    CZ_p = np.zeros((_Npath, _n))
    GZ_ti = np.zeros((_Npath, _n))  # tranche coupon
    CZ_ti = np.zeros((_Npath, _n))  # tranche coupon
    GZ_a = np.zeros((_Npath, _n))
    CZ_a = np.zeros((_Npath, _n))

    # set the initial balance:
    CG_b[:,0] = CG_init
    VE_b[:,0] = VE_init
    CM_b[:,0] = CM_init
    GZ_b[:,0] = GZ_init
    TC_b[:,0] = TC_init
    CZ_b[:,0] = CZ_init

    for i in range(_n):
        # interest in from tranche coupon rate
        CZ_ti[:, i] = CZ_b[:, i] * tranche_coupon
        GZ_ti[:, i] = GZ_b[:, i] * tranche_coupon

        CG_p[:, i] = np.maximum(0, np.minimum(pool_cf[:,i] + CZ_ti[:,i], CG_b[:,i]))
        VE_p[:, i] = np.maximum(0, np.minimum(pool_cf[:,i] + GZ_ti[:,i] + CZ_ti[:,i] - CG_p[:,i],
                                              VE_b[:,i]))
        CM_p[:,i] = np.maximum(0, np.minimum(pool_cf[:,i] + GZ_ti[:,i] + CZ_ti[:,i] - CG_p[:,i] - VE_p[:,i],
                                             CM_b[:,i]))

        # Binary choice
        _CM_b_next = CM_b[:,i] - CM_p[:,i]
        _b_CM_b_next = (_CM_b_next > 0).astype(float)
        GZ_a[:,i] =  _b_CM_b_next * GZ_ti[:,i] + (1-_b_CM_b_next) * np.minimum(CM_p[:,i], GZ_ti[:,i])

        GZ_p[:,i] = np.maximum(0, np.minimum(pool_cf[:,i] + GZ_a[:,i] + CZ_ti[:,i] - CG_p[:,i] - VE_p[:,i] - CM_p[:,i],
                                             GZ_b[:,i]))
        # Binary choice
        _GZ_b_next = GZ_b[:,i] + GZ_a[:,i] - GZ_p[:,i]
        _b_GZ_b_next = (_GZ_b_next > 0).astype(float)
        TC_p[:,i] = (1-_b_GZ_b_next) * \
                    np.minimum(pool_cf[:,i] + CZ_ti[:,i] - GZ_p[:,i], TC_b[:,i])

        # Binary choice
        _TC_b_next = TC_b[:,i] - TC_p[:,i]
        _b_TC_b_next = (_TC_b_next > 0).astype(float)
        CZ_a[:,i] = _b_TC_b_next * CZ_ti[:,i] + (1-_b_TC_b_next) * np.minimum(TC_p[:,i], CZ_ti[:,i])

        CZ_p[:,i] = np.maximum(0, np.minimum(pool_cf[:,i] + CZ_a[:,i] - CG_p[:,i] - VE_p[:,i] - CM_p[:,i] - GZ_p[:,i] - TC_p[:,i],
                                             CZ_b[:,i]))

        # interest batch
        CG_i[:,i] = CG_b[:,i]*coupon
        VE_i[:,i] = VE_b[:,i]*coupon
        CM_i[:,i] = CM_b[:,i]*coupon
        GZ_i[:,i] = GZ_ti[:,i] - GZ_a[:,i]
        TC_i[:,i] = TC_b[:,i]*coupon
        CZ_i[:,i] = CZ_ti[:,i] - CZ_a[:,i]

        if(i < _n-1):  # we don't update the last period's balance
            CG_b[:,i+1] = CG_b[:,i] - CG_p[:,i]
            VE_b[:,i+1] = VE_b[:,i] - VE_p[:,i]
            CM_b[:,i+1] = _CM_b_next
            GZ_b[:,i+1] = _GZ_b_next
            TC_b[:,i+1] = _TC_b_next
            CZ_b[:,i+1] = CZ_b[:,i] + CZ_a[:,i] - CZ_p[:,i]

    return CG_p+CG_i, VE_p+VE_i, CM_p+CM_i, GZ_p+GZ_i, TC_p+TC_i, CZ_p+CZ_i



""""
Below are the new functions for HW2.
"""

def hazardRateFactory(gamma, p, b1, b2, v1, v2):
    """Return a tailored Hazard rate function."""
    def __wrapper(_t):      
        try:  # when _t is array
            __part2 = np.array([np.exp(b1*v1(__t) + b2*v2(__t))  for __t in _t])
            __part1 = np.repeat(gamma*p*np.power(gamma*_t, p-1)/(1+np.power(gamma*_t, p)),
                                __part2.shape[1]).reshape((len(_t), -1))
            return (__part1 * __part2).T
        except:
            __part2 = np.exp(b1*v1(_t) + b2*v2(_t))
            __part1 = gamma*p*np.power(gamma*_t, p-1)/(1+np.power(gamma*_t, p))
            return __part1 * __part2
    return __wrapper


def hazardToSMM(hazard_func):
    """Convert from hazard rate to SMM"""
    def __wrapper(_t):
        try: # when _t is array
            _x = [np.linspace(__t, __t+1, num=100, endpoint=False) for __t in _t]
            return (1.0 - np.array([np.exp(-np.trapz(hazard_func(__x), __x)) for __x in _x])).T
        except:
            _x = np.linspace(_t, _t+1, num=100, endpoint=False)
            return 1.0 - np.exp(-np.trapz(hazard_func(_x), _x))
    return __wrapper


def cashFlowPriceMC(CFs, Zs, anti=False):
    """
    A very tailored function for HW2.

    Parameters:
    CFs : a m by n array of MC simulated undiscounted cash flow of a security
    Zs  : a m by n array of MC simulated discount factors

    Will return average price of all m paths
    """
    # pathwise results (one path in each row)
    _path_results = (CFs * Zs).sum(axis=1)

    if anti:
        # Average of anti-thetic paths (2nd half is the mirror paths)
        _n = int(len(_path_results) / 2)
        _path_results = (_path_results[:_n] + _path_results[_n:]) / 2
    
    _mean = _path_results.mean()  # mean
    _se = scipy.stats.sem(_path_results)  # standard error

    return _mean, _se


""""
Below are the new functions for HW3.
"""

class simpleBSABS:
    """The simple BSABS pool class."""

    def __init__(self, wac, warm, wala=0, psa=100, init_principle=100, Npath=1):
        """Initiation of the object."""
        # sanity check
        assert type(warm) == int, \
            "Input WAC should be an integer, not {}!".format(wac)

        # dump in the variables
        self.wac = wac  ## this should be a function!
        self.warm = warm
        self.wala = wala
        self.psa = psa
        self.init_principle = float(init_principle)
        self.CPR = lambda i: self.psa/100*0.06*np.minimum(1, (i+self.wala)/30)
        self.SMM = lambda i: 1 - np.power(1-self.CPR(i), 1/12.)
        self.DEF = lambda i: 1 - np.power(0)
        self.Npath = Npath
        self.principals = np.repeat(init_principle, Npath*warm).reshape(Npath, warm)

    def setPSA(self, psa):
        """Set PSA."""
        self.psa = psa

    def setCPR(self, CPR):
        """Set CPR."""
        self.CPR = CPR

    def setSMM(self, SMM):
        """Set SMM."""
        self.SMM = SMM

    def setDef(self, DEF):
        """Set default rate evolution"""
        self.DEF = DEF

    def getSMM(self):
        """Get SMM."""
        return self.SMM

    def getDef(self):
        """Get Def."""
        return self.DEF
        
    def getPrinciple(self, t):
        return self.principals[:,t]

    def getCashFlows(self, fills=-1):
        """
        Get cashflows for each month.

        Parameters:
        fills = size of the output array of each cashflows,
                default to the warm

        return 3 cashflows: interests, principals, and prepayments
        """
        if (fills == -1):
            fills = self.warm

        Npath = self.Npath
        _pmts = np.squeeze(np.zeros((Npath, fills)))
        _ints = np.squeeze(np.zeros((Npath, fills)))
        _prins = np.squeeze(np.zeros((Npath, fills)))
        _prepays = np.squeeze(np.zeros((Npath, fills)))
        _defaults = np.squeeze(np.zeros((Npath, fills)))
        #_balance = np.repeat(self.init_principle, Npath)
        _balance = self.getPrinciple(0)

        for i in range(min(self.warm, fills)):

            _c = self.wac(i)/12 ## this is time_varying
            _n = self.warm - i
            ## determin the defaults at t            
            _defaults[:, i] = _balance*self.DEF(i+1) # each row might be different
            _pmts[:, i] = (_balance-_defaults[:, i])*_c*(1+_c)**_n/((1+_c)**_n-1)

            _ints[:, i] = _c*(_balance-_defaults[:, i])
            _prins[:, i] = _pmts[:, i] - _ints[:, i]
            _prepays[:, i] = (_balance-_defaults[:, i] - _prins[:, i])*self.SMM(i+1)
            _balance -= (_prepays[:, i] + _prins[:, i] + _defaults[:, i])

            ## update principals
            if i < min(self.warm, fills)-1:
                self.principals[:,i+1] = _balance

        return _ints, _prins, _prepays, _defaults, self.principals


def getHousePrice(q, phi, dt, h_0=1.):
    """
    Return a function for Monte Carlo simulations for House Price.

    parameters --
    q: rental flow rate
    phi: volatility
    h_0: current house price
    """
    def __wrapper(rz):  # rz is a 1d array of lenth 2n, where the first n is r,
                        # and the last n is z
        _n = int(len(rz)/2)
        _r = rz[:_n]
        _z = rz[_n:]
        
        __h_s = np.repeat(h_0, len(_z)+1)  # r_t space
        for _i, (_r_i, _z_i) in enumerate(zip(_r, _z)):
            __temp = 1 + (_r_i - q) * dt + phi*np.sqrt(dt)*_z_i
            
            __h_s[_i+1] = __h_s[_i] * __temp

        return __h_s

    return __wrapper



def getBSABScashFlow(pool_pp, pool_int, pool_def, pool_balance,
                     init_princ_array, spreads, LIBOR, dt=1/12):
    """
    Get the CF of a BSABS.

    Parameter:
    pool_pp:  pool prepayment+principle, going all from top
    pool_int: pool interests
    pool_def: pool default
    init_princ_array: array of asset initial value
    spreads: assets' spread to libor in decimal, should be the same size
             with init_princ_array
    LIBOR:  market libor rate, same dimension as pool_*
    """
    _Npath, _n = pool_pp.shape
    # print(_Npath)
    _n_asset = len(init_princ_array)

    # 3d_array:  path x n_of_asset x payment length
    assets_cf = np.zeros((_Npath, _n_asset, _n))
    # Keep track of CDS default amounts
    def_val = np.zeros((_Npath, _n_asset, _n))
    # Keep track of tranche amounts
    princ_val = np.zeros((_Npath, _n_asset, _n))
    # this is path x n_of_asset
    _curr_princ = np.repeat([init_princ_array],_Npath,axis=0)

    prin_id_now = np.repeat(0, _Npath) # current asset under pp
    def_id_now = np.repeat(_n_asset-1, _Npath) # current asset under default payment

    ## below is just magic... don't touch! 
    for i in range(_n): # iterate through payment length
        ## Some variables to setup first
        # current remaining balance
        _curr_balance = pool_balance[:, i]
        ## get interest rate for each asset and path
        _int_rate = np.add.outer(LIBOR[:,i], spreads)
        # the theoretical interest accrued over last month (before deducting principals)
        _face_int_accu = _curr_princ * _int_rate * dt
#        print(_curr_princ.sum(axis=1).mean())
        
        ## We deal with default first ...
        # print(i, "def")
        curr_def = pool_def[:,i] ## defaut payment at time i
        
        # First, deduct default from Excess Return and Overcollateralization
        ex_spread = np.maximum(0, pool_int[:, i] - _face_int_accu.sum(axis=1))
        OCA = np.maximum(0, _curr_balance - _curr_princ.sum(axis=1))
        
        # The order of the following statements must be retained!
        cushion = OCA + ex_spread    # cache total cushion
#        print((cushion).mean())
        
        OCA = np.maximum(0, OCA - np.maximum(0, curr_def - ex_spread))
        ex_spread = np.maximum(0, ex_spread - curr_def)

        # Final amount of default
        curr_def = np.maximum(0, curr_def - cushion)

        while(any(curr_def>0)):
            # print(def_id_now)
            ## get current index of bottom asset with principle > 0
            bottom_princ_temp = _curr_princ[np.arange(_Npath), def_id_now]

            ## update the current principle
            _curr_princ[np.arange(_Npath), def_id_now] -= \
                np.minimum(curr_def, bottom_princ_temp)

            ## update the amount of default 
            def_val[np.arange(_Npath), def_id_now, np.repeat(i,_Npath)] += \
                np.minimum(curr_def, bottom_princ_temp)
            
            ## check whether this principle will survive
            _principle_survive = (bottom_princ_temp - curr_def) > 0
            
            ## get unpaid def at current asset class
            curr_def = np.maximum(0, curr_def - bottom_princ_temp)

            ## update the current asset under pp
            def_id_now -= 1 - _principle_survive * 1 

            ## if def_id_now for an asset is out of bound (0), but pp is still >0
            ## for that path, we terminate that path manuall by set that path's
            ## pp to 0
            curr_def[def_id_now < 0] = 0
            def_id_now[def_id_now < 0 ] = 0 ## avoid of OOB
        
        ## then we deal with principal payment...
        # print(i, "pp")
        curr_pp = pool_pp[:,i]  # pp at time i
        
        # Extra Principal Distribution Amount
        OCTA = np.maximum(0.031 * pool_balance[:, 0], 3967158.)
        ex_PDA = np.maximum(0, np.minimum(OCTA - OCA, ex_spread))
        curr_pp += ex_PDA

        while(any(curr_pp>0)):

            ## get current index of top asset with principle > 0
            # print(prin_id_now, curr_pp.sum())
            top_princ_temp = _curr_princ[np.arange(_Npath), prin_id_now]

            ## update the current principle
            _curr_princ[np.arange(_Npath), prin_id_now] -= \
                np.minimum(curr_pp, top_princ_temp)

            ## update the asset cash flow 
            assets_cf[np.arange(_Npath), prin_id_now, np.repeat(i,_Npath)] += \
                np.minimum(curr_pp, top_princ_temp)

            ## check whether this principle will survive
            _principle_survive = (top_princ_temp - curr_pp) > 0

            ## get unpaid pp at current asset class
            curr_pp = np.maximum(0, curr_pp - top_princ_temp)

            ## update the current asset under pp
            prin_id_now += 1 - _principle_survive * 1 

            ## if prin_id_now for an asset is out of bound, but pp is still >0
            ## for that path, we terminate that path manuall by set that path's
            ## pp to 0
            curr_pp[prin_id_now >= _n_asset] = 0
            prin_id_now[prin_id_now >= _n_asset] = _n_asset-1 ## avoid of OOB


        ## next we deal with interests
        curr_int = pool_int[:,i] ## interests payment at time i

        # set up target 
        _target = curr_int.reshape(_Npath,1)

        # track the results
        _real_int_payment = _face_int_accu.copy()
        _real_int_payment[np.greater_equal(np.cumsum(_face_int_accu,axis=1), _target)] = 0

        last_non_zero = np.less(np.cumsum(_face_int_accu,axis=1), _target).sum(axis=1)
        # Avoid OOB
        last_non_zero = np.minimum(last_non_zero, _n_asset-1)

        _real_int_payment[np.arange(len(_face_int_accu)), last_non_zero] = \
            np.minimum(_face_int_accu[np.arange(len(_face_int_accu)), last_non_zero], 
                       np.squeeze(_target) - _real_int_payment.sum(axis=1))

        assets_cf[:,:,i] += _real_int_payment


        ## Deal with interest shortfall as default (and cash flows)
        _int_short = _face_int_accu - _real_int_payment
        ## Deduct shorted interest from principal
        _curr_princ -= np.minimum(_curr_princ, _int_short)
        def_val[:,:,i] += np.minimum(_curr_princ, _int_short)    
 

        ## Update final balance amount
        princ_val[:,:,i] = _curr_princ
        
    return assets_cf, def_val, princ_val


def hazardRateFactory_1(gamma, p, b, v):
    """Return a tailored Hazard rate function, 3 factors."""
    def __wrapper(_t):      
        try:  # when _t is array
            __part2 = np.array([np.exp(b*v(__t))  for __t in _t])
            __part1 = np.repeat(gamma*p*np.power(gamma*_t, p-1)/(1+np.power(gamma*_t, p)),
                                __part2.shape[1]).reshape((len(_t), -1))
            return (__part1 * __part2).T
        except:
            __part2 = np.exp(b*v(_t))
            __part1 = gamma*p*np.power(gamma*_t, p-1)/(1+np.power(gamma*_t, p))
            return __part1 * __part2
    return __wrapper
