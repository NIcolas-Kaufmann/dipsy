import math
import numpy as np
from types import SimpleNamespace
import dustpy.constants as c




def log_uniform_sample(random_uniform, min_value, max_value):
    """
    Generates a random sample from a log-uniform distribution.

    Parameters:
        random_uniform (float): A random number uniformly distributed between 0 and 1.
        min_value (float): The minimum value of the log-uniform distribution.
        max_value (float): The maximum value of the log-uniform distribution.

    Returns:
        float: A random sample from the log-uniform distribution.
    """
    if min_value <= 0 or max_value <= 0:
        raise ValueError("min_value and max_value must be greater than 0.")
    if not (0 <= random_uniform <= 1):
        raise ValueError("random_uniform must be between 0 and 1.")

    log_min = math.log(min_value)
    log_max = math.log(max_value)
    log_sample = log_min + random_uniform * (log_max - log_min)
    return math.exp(log_sample)


def kroupa_imf_sample(random_uniform, m_min=0.01, m_max=150):
    """
    Sample from the Kroupa IMF using inverse transform sampling.
    
    Parameters:
        size (int): Number of samples.
        m_min (float): Minimum stellar mass.
        m_max (float): Maximum stellar mass.
    
    Returns:
        numpy.ndarray: Array of sampled masses.
    """
    # Define mass breakpoints
    m1, m2, m3 = 0.08, 0.5, 1.0  # Solar masses
    
    # Define power-law exponents
    alpha0, alpha1, alpha2, alpha3 = -0.3, -1.3, -2.3, -2.3
    
    # Normalization constants
    k0 = 1
    k1 = k0 * m1**(alpha0 - alpha1)
    k2 = k1 * m2**(alpha1 - alpha2)
    k3 = k2 * m3**(alpha2 - alpha3)
    
    # Compute CDF segments
    def cdf(m, alpha, k, m_min):
        if alpha != -1:
            return k / (alpha + 1) * (m**(alpha + 1) - m_min**(alpha + 1))
        else:
            return k * np.log(m / m_min)
    
    cdf0 = cdf(m1, alpha0, k0, m_min)
    cdf1 = cdf0 + cdf(m2, alpha1, k1, m1)
    cdf2 = cdf1 + cdf(m3, alpha2, k2, m2)
    cdf3 = cdf2 + cdf(m_max, alpha3, k3, m3)

    
    # Inverse transform sampling
    u = random_uniform*cdf3

    

    if u < cdf0:
        samples = ((u * (alpha0 + 1) / k0) + m_min**(alpha0 + 1))**(1 / (alpha0 + 1))
    elif u < cdf1:
        samples = (( (u - cdf0) * (alpha1 + 1) / k1) + m1**(alpha1 + 1))**(1 / (alpha1 + 1))
    elif u < cdf2:
        samples = (( (u - cdf1) * (alpha2 + 1) / k2) + m2**(alpha2 + 1))**(1 / (alpha2 + 1))
    else:
        samples = (( (u - cdf2) * (alpha3 + 1) / k3) + m3**(alpha3 + 1))**(1 / (alpha3 + 1))

    return samples

#write a funtion that map a random number form 0 1 uniformy to a different interval uniformly with min and max values
def map_uniform_to_interval(random_uniform, min_value, max_value):
    """
    Maps a random number uniformly distributed between 0 and 1 
    to a different interval [min_value, max_value] uniformly.

    Parameters:
        random_uniform (float): A random number uniformly distributed between 0 and 1.
        min_value (float): The minimum value of the target interval.
        max_value (float): The maximum value of the target interval.

    Returns:
        float: A random number uniformly distributed in the interval [min_value, max_value].
    """
    if not (0 <= random_uniform <= 1):
        raise ValueError("random_uniform must be between 0 and 1.")
    return min_value + random_uniform * (max_value - min_value)

#function that creates the inital disk parametes as described in Delusu et al. 2024 Table 2 
def disk_delussu(n,size = 8,random_seed = 42):

    np.random.seed(random_seed)
    for _ in range(n):  
        np.random.rand(size)
    rand = np.random.rand(size)

    disk  = SimpleNamespace()
    disk.alpha = log_uniform_sample(rand[0], 1e-4,1e-2)
    disk.mstar = kroupa_imf_sample(rand[1],m_min=0.2,m_max=2.)*c.M_sun
    disk.mdisk = log_uniform_sample(rand[2],1e-3,0.5) * disk.mstar
    disk.rc = log_uniform_sample(rand[3], 10, 230) * c.au
    disk.vfrag = map_uniform_to_interval(rand[4], 200, 2000)
    disk.rp = map_uniform_to_interval(rand[5],0.05,1.5)*disk.rc
    disk.mp = min(map_uniform_to_interval(rand[6],1,1050)*c.M_earth,disk.mdisk)
    disk.tp = map_uniform_to_interval(rand[7],0.1,0.4)*1e6*c.year
    disk.d2g = 1e-2 
    disk.rhos = 1.7 
    disk.gamma = 1 
    # Emsenhuber 2023 Table 6
    disk.Lx = 10** np.random.normal(loc=0.31, scale=0.54, size=1)[0] * (disk.mstar/c.M_sun)**1.52 * 1e30 # erg/s
    disk.Feux = 10 ** np.random.normal(loc=3.25, scale=0.93, size=1)[0]  #G0
    return disk


def disk_v2(n,size = 8,random_seed = 42):

    np.random.seed(random_seed)
    for _ in range(n):  
        np.random.rand(size)
    rand = np.random.rand(size)

    disk  = SimpleNamespace()
    disk.alpha = log_uniform_sample(rand[0], 1e-4,1e-2)
    disk.mstar = kroupa_imf_sample(rand[1],m_min=0.2,m_max=2.)*c.M_sun
    disk.mdisk = log_uniform_sample(rand[2],1e-3,0.5) * disk.mstar
    disk.rc = log_uniform_sample(rand[3], 10, 230) * c.au
    disk.vfrag = map_uniform_to_interval(rand[4], 200, 2000)
    disk.rp = max(4*c.au, map_uniform_to_interval(rand[5],0.05,1.5)*disk.rc) # ensure rp is at least 4 au
    disk.mp = min(map_uniform_to_interval(rand[6],1,1050)*c.M_earth,disk.mdisk)
    disk.tp = map_uniform_to_interval(rand[7],0.1,0.4)*1e6*c.year
    disk.d2g = 1e-2 
    disk.rhos = 1.7 
    disk.gamma = 1 
    # Emsenhuber 2023 Table 6
    disk.Lx = 10** np.random.normal(loc=0.31, scale=0.54, size=1)[0] * (disk.mstar/c.M_sun)**1.52 * 1e30 # erg/s
    disk.Feux = 10 ** max(-1, np.random.normal(loc=1, scale=0.5, size=1)[0])  #G0 Weder 
    return disk

def disk_v3(n,size = 8,random_seed = 42):

    np.random.seed(random_seed)
    for _ in range(n):  
        np.random.rand(size)
    rand = np.random.rand(size)

    disk  = SimpleNamespace()
    disk.alpha = log_uniform_sample(rand[0], 10**-3.5,10**-2.5)
    disk.mstar = kroupa_imf_sample(rand[1],m_max=2.)*c.M_sun
    while disk.mstar < 0.2*c.M_sun:
        disk.mstar = kroupa_imf_sample(np.random.rand(),m_max=2.)*c.M_sun
    disk.mdisk = log_uniform_sample(rand[2],10**-2.3,10**-0.5) * disk.mstar
    disk.rc = log_uniform_sample(rand[3], 10, 230) * c.au
    disk.vfrag = map_uniform_to_interval(rand[4], 500, 2000)
    disk.rp = max(4*c.au, map_uniform_to_interval(rand[5],0.05,0.75)*disk.rc) # ensure rp is at least 4 au
    disk.mp = min(map_uniform_to_interval(rand[6],150,1050)*c.M_earth,disk.mdisk)
    disk.tp = map_uniform_to_interval(rand[7],0.1,0.4)*1e6*c.year
    disk.d2g = 1e-2 
    disk.rhos = 1.7 
    disk.gamma = 1 
    # Emsenhuber 2023 Table 6
    disk.Lx = 10** np.random.normal(loc=0.31, scale=0.54, size=1)[0] * (disk.mstar/c.M_sun)**1.52 * 1e30 # erg/s
    disk.Feux = 10 ** max(-1, np.random.normal(loc=1, scale=0.5, size=1)[0])  #G0 Weder 
    return disk



def disk_extended(n,size = 8,random_seed = 42):

    np.random.seed(random_seed)
    for _ in range(n):  
        np.random.rand(size)
    rand = np.random.rand(size)

    disk  = SimpleNamespace()
    disk.alpha = log_uniform_sample(rand[0], 10**-4,10**-2)
    disk.mstar = kroupa_imf_sample(rand[1],m_max=2.)*c.M_sun
    while disk.mstar < 0.2*c.M_sun:
        disk.mstar = kroupa_imf_sample(np.random.rand(),m_max=2.)*c.M_sun
    disk.mdisk = log_uniform_sample(rand[2],10**-2.3,10**-0.5) * disk.mstar
    disk.rc = log_uniform_sample(rand[3], 10, 230) * c.au
    disk.vfrag = map_uniform_to_interval(rand[4], 50, 2000)
    disk.rp = max(4*c.au, map_uniform_to_interval(rand[5],0.05,1.5)*disk.rc) # ensure rp is at least 4 au
    disk.mp = min(map_uniform_to_interval(rand[6],150,1050)*c.M_earth,disk.mdisk)
    disk.tp = map_uniform_to_interval(rand[7],0.1,0.4)*1e6*c.year
    disk.d2g = 1e-2 
    disk.rhos = 1.7 
    disk.gamma = 1 
    # Emsenhuber 2023 Table 6
    disk.Lx = 10** np.random.normal(loc=0.31, scale=0.54, size=1)[0] * (disk.mstar/c.M_sun)**1.52 * 1e30 # erg/s
    disk.Feux = 10 ** max(-1, np.random.normal(loc=1, scale=0.5, size=1)[0])  #G0 Weder 
    return disk