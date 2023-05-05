from numpy import *
from scipy import integrate

def Talbot(f, t, *args):
    """
    takes the inverse Laplace of f(s, *args)
    evaluated at t

    t can be a scalar/vector
    f can be a scalar/vector-valued function    
    """
    n = 32
    c1 = 0.5017
    c2 = 0.6407
    c3 = 0.6122
    c4 = 0.2645j
    shift = 0.0
    h = 2*pi/n
    anss = 0

    for kind in range(int(n/2)):
        theta = -pi+(kind+0.5)*h
        z = shift + n/t *(c1*theta / tan(c2*theta) - c3  + c4*theta)
        dz = n/t * (-c1*c2*theta / (sin(c2*theta))**2 +c1/tan(c2*theta) + c4)
        mt = f(z, *args)
        anss = anss + exp(z*t)*mt*dz

    return real(anss*h/pi/1j)    

def PowerProfile(peak_power, peak_time, extra_time, charging_time):
    """
    returns a function of dissipated power as a function of time during fast charging;
    power goes from 0 to peak_power between time t = 0 to t = peak_time

    The profile stays at peak power from time t = peak_time to t = peak_time + extra_time, then
    the power decreases down to zero from t = peak_time + extra_time to t = charging_time
    """
    def PowerDissipated(t):
        powers = zeros_like(t)

        for i in range(t.shape[0]):
            ti = t[i]
            if ti <= peak_time:
                powers[i] = peak_power/peak_time*ti
            elif peak_time <= ti <= peak_time + extra_time:
                powers[i] = peak_power
            else:
                powers[i] = peak_power/(peak_time+extra_time-charging_time)*(ti-charging_time)
        
        #powers = peak_power*ones_like(t)
        return powers

    return PowerDissipated

def FastChargingBatteryTemperature(t, yt, L, pc, power_profile, **kwargs):
    """
    temperature of the battery while charging assuming it dissipates power according to the function power_profile

    L = distance between the battery and the active cooling system
    pc = volumetric heat capacity of material encapsulating the battery
    yt = vector of values which return the sample conductivity yt(i) for a given measurement time t(i) of the battery encapsulation

    """
    offset = t[0] * yt[0]

    if "custom_integral" in kwargs.keys():
        intyt = kwargs["custom_integral"]
    else:
        intyt = integrate.cumtrapz(yt, t, initial=0) + offset

    def LaplaceFunc(S):
        return power_profile(t)/yt/S/sqrt(pc*S)*tanh(L*sqrt(pc*S))
    
    return Talbot(LaplaceFunc, intyt)