"""
Functions to calculate observables or read data from simulations
"""
from pathlib import Path
from types import SimpleNamespace
import h5py
from dustpylib.radtrans.slab.slab import I_over_B_EB, bplanck

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from collections import namedtuple
import dustpy.constants as c
import astropy.constants as ac

from dipsy.tracks import M_sun
from .cgs_constants import year, jy_sas, c_light, pc
from .utils import bplanck
import pandas as pd
import re

import dsharp_opac

observables = namedtuple(
    'observables', ['rf', 'flux_t', 'tau', 'I_nu', 'a', 'sig_da'])
dustpy_result = namedtuple('dustpy_result', [
                           'r', 'a_max', 'a', 'a_mean', 'sig_d', 'sig_da', 'sig_g', 'time',"T", 'L_star',"M_disk"])
tripod_result = namedtuple('tripod_result', [
                           'r', 'a_max', 'a', 'a_mean', 'sig_d','q', 'sig_g', 'time', 'T', 'L_star',"M_disk"])

rosotti_result = namedtuple('rosotti_result', [
                            'a_max', 'time', 'T', 'sig_d', 'sig_g', 'd2g', 'r', 'L_star', 'M_star', 'T_star'])


def get_powerlaw_dust_distribution(sigma_d, a_max, q=3.5, na=10, a0=None, a1=None):
    """
    Makes a power-law size distribution up to a_max, normalized to the given surface density.

    Arguments:
    ----------

    sigma_d : array
        dust surface density array

    a_max : array
        maximum particle size array

    Keywords:
    ---------

    q : float | array
        particle size index, n(a) propto a**-q
        if array, it has to have the same length as sigma_d

    na : int
        number of particle size bins

    a0 : float
        minimum particle size

    a1 : float
        maximum particle size

    Returns:
    --------

    a : array
        particle size grid (centers)

    a_i : array
        particle size grid (interfaces)

    sig_da : array
        particle size distribution of size (len(sigma_d), na)
    """

    if a0 is None:
        a0 = a_max.min()

    if a1 is None:
        a1 = 2 * a_max.max()

    nr = len(sigma_d)
    sig_da = np.zeros([nr, na]) + 1e-100

    a_i = np.logspace(np.log10(a0), np.log10(a1), na + 1)
    a = 0.5 * (a_i[1:] + a_i[:-1])

    # we want to turn q into an array if it isn't one already
    q = q * np.ones(nr)

    for ir in range(nr):

        if a_max[ir] <= a0:
            sig_da[ir, 0] = 1
        else:
            i_up = np.where(a_i < a_max[ir])[0][-1]

            # filling all bins that are strictly below a_max

            if q[ir] == 4.0:
                for ia in range(i_up):
                    sig_da[ir, ia] = np.log(a_i[ia + 1] / a_i[ia])

                # filling the bin that contains a_max
                sig_da[ir, i_up] = np.log(a_max[ir] / a_i[i_up])
            else:
                for ia in range(i_up):
                    sig_da[ir, ia] = a_i[ia +
                                         1]**(4 - q[ir]) - a_i[ia]**(4 - q[ir])

                # filling the bin that contains a_max
                sig_da[ir, i_up] = a_max[ir]**(4 - q[ir]) - \
                    a_i[i_up]**(4 - q[ir])

        # normalize

        sig_da[ir, :] = sig_da[ir, :] / sig_da[ir, :].sum() * sigma_d[ir]

    return a, a_i, sig_da


def read_rosotti_data(fname):
    """
    Reads Giovanni Rosottis HDF5 file and returns
    a dict with numpy arrays, all in CGS.
    """
    from astropy import constants as c
    from astropy import units as u
    import numpy as np

    au = u.au.to('cm')
    M_sun = c.M_sun.cgs.value
    L_sun = c.L_sun.cgs.value

    with h5py.File(fname) as f:

        dset = f['pop_0']

        snap_indices = []
        for key in dset.keys():
            if key.startswith('time_'):
                snap_indices += [int(key.replace('time_', ''))]

        snap_indices.sort()

        r = dset[f'time_{snap_indices[0]}/yso_0/Disk/Rc'][()] * au
        n_t = len(snap_indices)
        n_r = len(r)

        amax = np.zeros([n_t, n_r])
        d2g = np.zeros([n_t, n_r])
        T = np.zeros([n_t, n_r])
        sig_d = np.zeros([n_t, n_r])
        sig_g = np.zeros([n_t, n_r])
        age = np.zeros([n_t])
        L_star = np.zeros([n_t])
        M_star = np.zeros([n_t])
        T_star = np.zeros([n_t])

        for idx in snap_indices:
            sig = dset[f'time_{idx}/yso_0/Disk/sigma'][()]
            d2g = dset[f'time_{idx}/yso_0/Disk/dust_frac'][()].sum(0)

            sig_g[idx, :] = sig / (1 + d2g)
            sig_d[idx, :] = sig / (1 + d2g) * d2g
            amax[idx, :] = dset[f'time_{idx}/yso_0/Disk/amax'][()]
            T[idx, :] = dset[f'time_{idx}/yso_0/Disk/T'][()]

            age[idx] = dset[f'time_{idx}/yso_0/evolution_time'][()] / \
                (2 * np.pi) * year
            L_star[idx] = dset[f'time_{idx}/yso_0/Star/llum'][()] * L_sun
            M_star[idx] = dset[f'time_{idx}/yso_0/Star/mass'][()] * M_sun
            T_star[idx] = dset[f'time_{idx}/yso_0/Star/teff'][()]

        return rosotti_result(amax, age, T, sig_d, sig_g, d2g, r, L_star, M_star, T_star)

def read_tazzari_data(fname=None):
    if fname is None:
        fname = Path(__file__).parent / "datasets/Tazzari2021.txt"
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    # Get column names from header (line 8)
    header_line = lines[8].strip()
    columns = [col.strip() for col in header_line.split('|') if col.strip()]
    
    # Parse data starting from line 12
    data = []
    for line in lines[12:]:
        if line.strip():
            name = line[0:25].strip()  # First 25 chars
            rest = line[25:].strip().split()
            
            # Find where numbers start
            numeric_start = 0
            for i, part in enumerate(rest):
                try:
                    float(part)
                    numeric_start = i
                    break
                except:
                    continue
            
            other_name = ' '.join(rest[:numeric_start]) if numeric_start > 0 else ''
            numeric_vals = rest[numeric_start:numeric_start + 10]
            
            if len(numeric_vals) == 10:
                data.append([name, other_name] + numeric_vals)
    
    df = pd.DataFrame(data, columns=columns)
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col])
    
    return df


def read_feng_data(fname=None):
    if fname is None:
        fname = Path(__file__).parent / "datasets/feng_22.csv"
    with open(fname, 'r') as f:
        lines = f.readlines()

    # File is tab-separated. The header row is missing one label because two
    # columns share the "Continuum"/"CO" group header in the original table,
    # so we name columns manually based on the data pattern (col 9, "R_mm",
    # is the one uncertain label - rename if you can confirm it from the paper).
    columns = [
        'Idx', 'Name', 'Incl', 'PA',
        'Continuum_freq_GHz', 'Continuum_flux_mJy',
        'CO_val', 'Line', 'R_mm_arcsec',
        'CO_flux_mJy', 'R_CO_over_R_mm', 'References'
    ]

    data = []
    for line in lines[1:]:
        if line.strip():
            parts = [p.strip() for p in line.rstrip('\n').split('\t')]
            parts = [p for p in parts if p != '' or len(parts) <= len(columns)]
            parts = parts[:len(columns)]
            if len(parts) == len(columns):
                data.append(parts)

    df = pd.DataFrame(data, columns=columns)

    # Split "value +or- error" style columns into separate value/err columns
    for col in ['CO_val', 'CO_flux_mJy', 'R_CO_over_R_mm']:
        split = df[col].str.split(r'\+or-', expand=True)
        df[col + '_err'] = pd.to_numeric(split[1].str.strip(), errors='coerce')
        df[col] = pd.to_numeric(split[0].str.strip(), errors='coerce')

    # Numeric conversion for the remaining plain numeric columns
    for col in ['Idx', 'Incl', 'PA', 'Continuum_freq_GHz',
                'Continuum_flux_mJy', 'R_mm_arcsec']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def read_andrews_data(fname=None):
    if fname is None:
        fname = Path(__file__).parent / "datasets/ALLDISKS.summary.dat"

    # File is whitespace-separated with no header; astropy's ascii.read
    # names columns col1..colN (1-indexed), so we replicate that here.
    dat = pd.read_csv(fname, sep=r'\s+', header=None)
    dat.columns = [f'col{i+1}' for i in range(dat.shape[1])]
    ndisks = len(dat)

    xx = dat['col30'].to_numpy()
    xh = dat['col31'].to_numpy()
    xl = dat['col32'].to_numpy()
    y_f = dat['col36'].to_numpy()

    mstar = 10.**dat['col17'].to_numpy()

    yy = np.zeros(ndisks)
    yy[y_f == 0] = dat['col33'].to_numpy()[y_f == 0]
    yy[y_f == 1] = dat['col37'].to_numpy()[y_f == 1]

    yh = dat['col34'].to_numpy()
    yl = dat['col35'].to_numpy()

    cond = y_f == 0

    df = pd.DataFrame({
        'R_eff':   10.**yy[cond],
        'R_eff_l': (10.**yy - 10.**(yy - yl))[cond],
        'R_eff_h': (10.**(yy + yh) - 10.**yy)[cond],
        'L_mm':    10.**xx[cond],
        'L_mm_l':  (10.**xx - 10.**(xx - xl))[cond],
        'L_mm_h':  (10.**(xx + xh) - 10.**xx)[cond],
        'Mstar':   mstar[cond],
    })

    return df

def read_bito_data(fname = None):
    if fname is None:
        fname = Path(__file__).parent / "datasets/bito_22.csv"
    df = pd.read_csv(fname,sep=',',comment='#')

    return df 

def read_dustpy_data(data_path, time=None, amin=5e-10):
    """
    Read the dustpy files from the directory data_path, then interpolate the
    densities at the given time snapshots (or take the original time if None is
    passed).

    Arguments:
    ----------

    data_path : str
        path to the output directory where the hdf5 files are

    time : array | None
        if not None: interpolate at those times

    Output:
    -------
    returns dict with these keys:
    - r
    - a_max
    - sig_d
    - sig_g
    - time
    """
    import dustpy
    from scipy.interpolate import interp2d

    reader = dustpy.hdf5writer()
    reader.datadir = str(data_path)

    time_dp = reader.read.sequence("t")

    # Read the radial and mass grid
    r = reader.read.sequence("grid/r")[0]
    # rInt = reader.read.sequence("grid/rInt")
    # m = reader.read.sequence("grid/m")

    # Read the gas and dust densities
    sig_g = reader.read.sequence("gas/Sigma")
    sig_da = reader.read.sequence("dust/Sigma")
    sig_d = sig_da.sum(-1)
    A = reader.read.sequence("grid/A")
    M_disk = (sig_g * A).sum(-1)

    # Read the stokes number and dust size
    # St = reader.read.sequence("dust/St")
    a = reader.read.sequence("dust/a")[0, 0, :]

    i_min = np.argmin(np.abs(a - amin))
    sig_da=sig_da[:,:,i_min:]


    # Read the star mass and radius
    # M_star = reader.read.sequence("star/M", files]
    # R_star = reader.read.sequence("star/R", files]
    L_star = reader.read.sequence("star/L")

    # Obtain the dust to gas ratio
    # d2g = sig_d / sig_g

    # Read the Gas and Dust scale height
    # Hg = reader.read.sequence("gas/Hp")
    # Hd = reader.read.sequence("dust/h")

    # Read the gas (viscous) and dust velocities
    # Vel_g = reader.read.sequence("gas/v/visc")
    Vel_d = reader.read.sequence("dust/v/rad")

    # Read the alpha parameter and the orbital angular velocity
    # Alpha  = reader.read.sequence("gas/alpha")
    # OmegaK = reader.read.sequence("grid/OmegaK")

    # Read the gas midplane density, sound speed, and eta parameter
    # rho = reader.read.sequence("gas/rho")
    # cs = reader.read.sequence("gas/cs")
    # eta = reader.read.sequence("gas/eta")

    T = reader.read.sequence("gas/T")

    # Obtain the Accretion Rate of dust and gas
    # Acc_g = 2 * np.pi * r * Vel_g * sig_g
    # Acc_d = 2 * np.pi * r * (Vel_d * sig_d).sum(-1)

    # Obtain the alpha-viscosity
    # Visc =  Alpha * cs * cs / OmegaK

    a_mean = (a[i_min:] * sig_da * np.abs(Vel_d[:,:,i_min:])).sum(-1) / \
        (sig_da * np.abs(Vel_d[:,:,i_min:])).sum(-1)
    a_max = a[sig_da.argmax(-1)]
    a = a[i_min:]

    if time is None:
        time = time_dp
    else:
        f_Td = interp2d(np.log10(r), np.log10(time_dp + 1e-100), np.log10(T))
        f_sd = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(sig_d))
        f_sg = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(sig_g))
        f_ax = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(a_max))
        f_am = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(a_mean))
        f_Lstar = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(L_star))

        T = 10.**f_Td(np.log10(r), np.log10(time + 1e-100))
        sig_d = 10.**f_sd(np.log10(r), np.log10(time + 1e-100))
        sig_g = 10.**f_sg(np.log10(r), np.log10(time + 1e-100))
        a_max = 10.**f_ax(np.log10(r), np.log10(time + 1e-100))
        a_mean = 10.**f_am(np.log10(r), np.log10(time + 1e-100))
        L_star = 10.**f_Lstar(np.log10(r), np.log10(time + 1e-100))

        sig_da_new = np.zeros([len(time), len(r), len(a)])
        for ia in range(len(a)):
            f = interp2d(np.log10(r), np.log10(time_dp + 1e-100),
                         np.log10(sig_da[:, :, ia]))
            sig_da_new[:, :, ia] = 10.**f(np.log10(r), np.log10(time + 1e-100))

        sig_da = sig_da_new

    dp = dustpy_result(r, a_max, a, a_mean, sig_d, sig_da, sig_g, time, T,L_star,M_disk)

    return dp

def read_tripod_data(data_path, time=None, prefix = ""):
    """
    Read the dustpy files from the directory data_path, then interpolate the
    densities at the given time snapshots (or take the original time if None is
    passed).

    Arguments:
    ----------

    data_path : str
        path to the output directory where the hdf5 files are

    time : array | None
        if not None: interpolate at those times

    Output:
    -------
    returns dict with these keys:
    - r
    - a_max
    - sig_d
    - sig_g
    - time
    """
    import dustpy
    from scipy.interpolate import interp2d

    reader = dustpy.hdf5writer()
    reader.datadir = str(data_path)

    time_dp = reader.read.sequence(prefix+"t")

    # Read the radial and mass grid
    r = reader.read.sequence(prefix+"grid/r")[0]
    # rInt = reader.read.sequence("grid/rInt")
    # m = reader.read.sequence("grid/m")

    A = reader.read.sequence(prefix+"grid/A")
    # Read the gas and dust densities
    sig_g = reader.read.sequence(prefix+"gas/Sigma")
    sig_da = reader.read.sequence(prefix+"dust/Sigma")
    sig_d = sig_da.sum(-1)
    M_disk = (sig_g * A).sum(-1)
    # Read the stokes number and dust size
    # St = reader.read.sequence("dust/St")
    a = reader.read.sequence(prefix+"dust/a")
    q = reader.read.sequence(prefix+"dust/qrec")

    # Read the star mass and radius
    # M_star = reader.read.sequence("star/M", files]
    # R_star = reader.read.sequence("star/R", files]
    L_star = reader.read.sequence(prefix+"star/L")

    # Obtain the dust to gas ratio
    # d2g = sig_d / sig_g

    # Read the Gas and Dust scale height
    # Hg = reader.read.sequence("gas/Hp")
    # Hd = reader.read.sequence("dust/h")

    # Read the gas (viscous) and dust velocities
    # Vel_g = reader.read.sequence("gas/v/visc")
    Vel_d = reader.read.sequence(prefix+"dust/v/rad")

    # Read the alpha parameter and the orbital angular velocity
    # Alpha  = reader.read.sequence("gas/alpha")
    # OmegaK = reader.read.sequence("grid/OmegaK")

    # Read the gas midplane density, sound speed, and eta parameter
    # rho = reader.read.sequence("gas/rho")
    # cs = reader.read.sequence("gas/cs")
    # eta = reader.read.sequence("gas/eta")

    T = reader.read.sequence(prefix+"gas/T")

    # Obtain the Accretion Rate of dust and gas
    # Acc_g = 2 * np.pi * r * Vel_g * sig_g
    # Acc_d = 2 * np.pi * r * (Vel_d * sig_d).sum(-1)

    # Obtain the alpha-viscosity
    # Visc =  Alpha * cs * cs / OmegaK

    a_mean = reader.read.sequence(prefix+"dust/a")[:,:,2]
    
    a_max = reader.read.sequence(prefix+"dust/s/max")

    if time is None:
        time = time_dp
    else:
        f_Td = interp2d(np.log10(r), np.log10(time_dp + 1e-100), np.log10(T))
        f_sd = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(sig_d))
        f_sg = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(sig_g))
        f_ax = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(a_max))
        f_am = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(a_mean))
        f_q = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(q))
        f_Lstar = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(L_star))

        T = 10.**f_Td(np.log10(r), np.log10(time + 1e-100))
        sig_d = 10.**f_sd(np.log10(r), np.log10(time + 1e-100))
        sig_g = 10.**f_sg(np.log10(r), np.log10(time + 1e-100))
        a_max = 10.**f_ax(np.log10(r), np.log10(time + 1e-100))
        a_mean = 10.**f_am(np.log10(r), np.log10(time + 1e-100))
        q = 10.**f_q(np.log10(r), np.log10(time + 1e-100))
        L_star = 10.**f_Lstar(np.log10(r), np.log10(time + 1e-100))

        sig_da_new = np.zeros([len(time), len(r), len(a)])
        for ia in range(len(a)):
            f = interp2d(np.log10(r), np.log10(time_dp + 1e-100),
                         np.log10(sig_da[:, :, ia]))
            sig_da_new[:, :, ia] = 10.**f(np.log10(r), np.log10(time + 1e-100))

        sig_da = sig_da_new

    dp = tripod_result(r, a_max, a, a_mean, sig_d, q, sig_g, time, T,L_star,M_disk)

    return dp


def get_observables(r, sig_g, sig_d, a_max, T, opacity, lam, distance=140 * pc,
                    flux_fraction=0.68, a=None, q=3.5, na=50, a0=None, a1=None,L_star=ac.L_sun.cgs.value,mu=2.3*c.m_p, scattering=True,
                    inc=0.0):
    """ac.L_sun.cgs.value
    Calculates the radial profiles of the (vertical) optical depth and the intensity for a given simulation
    at a given time (using the closest simulation snapshot).

    Arguments:
    ----------

    r : array
        radial array [cm]

    sig_g : array
        gas surface density on r [g/cm^2]

    sig_d : 1d-array | 2d-array
        - 1d: dust surface density on r [g/cm^2]
        - 2d: dust surface density on r and a shape=(len(r), len(a)) [g/cm^2]

    a_max : array
        maximum particle size on r [cm]

    T : array
        temperature grid on r [K]

    opacity : instance of Opacity
        the opacity to use for the calculation

    lam : array
        the wavelengths at which to calculate the observables [cm]

    Keywords:
    ---------

    distance : float
        distance to source [cm]

    flux_fraction : float
        at which fraction of the total flux the effective radius is defined [-]

    a : None | float
        if size distribution information is known (= sig_d is 2D), pass the
        particle size array here

    q : float | array
        size exponent to use: n(a) ~ a^-q, so 3.5=MRN
        if array, it has to have the same length as sig_d

    na : int
        length of the particle size grid

    a0 : float
        minimum particle size to use for the dust size distribution [cm]

    a1 : float
        maximum particle size to use for the dust size distribution [cm]

    scattering : bool
        if True, use the scattering solution, else just absorption

    inc : float
        inclination, default is 0.0 = face-on. This is just treated as
        increasing the path length across the slab model.

    Output:
    -------

    rf : array
        effective radii for every wavelength [cm]

    flux_t : array
        integrated flux for every wavelength [Jy]

    tau,Inu : array
        optical depth and intensity profiles at every wavelength [-, Jy/arcsec**2]

    sig_da, : array
        reconstructed particle size distribution on grid (res.a, res.x)
    """
    from scipy.integrate import cumulative_trapezoid

    # get the size distribution
    if (a is not None and sig_d.ndim != 2) or (a is None and sig_d.ndim != 1):
        raise ValueError(
            'either a=None and sig_d.ndim=1 or a!=None and sig_d.ndim=2')

    if a is None:
        a, a_i, sig_da = get_powerlaw_dust_distribution(
            sig_d, a_max, q=q, na=na, a0=a0, a1=a1)
    else:
        sig_da = sig_d

    sig_d_tot = sig_da.sum(-1)

    lam = np.array(lam, ndmin=1)
    n_lam = len(lam)

    I_nu = np.zeros([n_lam, len(r)])
    tau = np.zeros([n_lam, len(r)])

    # get opacities at our wavelength and particle sizes

    if scattering:
        k_a, k_s = opacity.get_opacities(a, lam)
        g = opacity.get_g(a, lam).T
        k_a = k_a.T
        k_s = k_s.T

        k_se = (1.0 - g) * k_s
        k_ext = k_a + k_se
    else:
        k_ext = opacity.get_opacities(a, lam)[0].T

    for ilam, _lam in enumerate(lam):
        freq = c_light / _lam

        # Calculate intensity profile
        # 1. optical depth
        tau[ilam, :] = (sig_da * k_ext[ilam, :].T).sum(-1) / np.cos(inc)
        tau = np.minimum(100., tau)

        if scattering:
            # 2. a size averaged opacity and from that the averaged epsilon
            k_a_mean = (sig_da * k_a[ilam, :].T).sum(-1) / sig_d_tot
            k_s_mean = (sig_da * k_se[ilam, :].T).sum(-1) / sig_d_tot
            eps_avg = k_a_mean / (k_a_mean + k_s_mean)

            # 3. plug those into the solution
            I_nu[ilam, :] = bplanck(freq, T) * \
                I_over_B_EB(tau[ilam, :], eps_avg)
        else:
            dummy = np.where(tau[ilam, :] > 1e-15,
                             (1.0 - np.exp(-tau[ilam, :])),
                             tau[ilam, :])
            I_nu[ilam, :] = bplanck(freq, T) * dummy

    # calculate the fluxes

    flux = np.cos(inc) * distance**-2 * \
        cumulative_trapezoid(2 * np.pi * r * I_nu, x=r, axis=1, initial=0)
    # integrated flux density in Jy (sanity check: TW Hya @ 870 micron and 54 parsec is about 1.5 Jy)
    flux_t = flux[:, -1] / 1e-23

    # converted intensity to Jy/arcsec**2

    I_nu = I_nu / jy_sas

    # interpolate radius whithin which >=68% of the dust flux is

    rf = np.array([np.interp(flux_fraction, _f / _f[-1], r) for _f in flux])
    # calculate the gas mass from the surface denity and radial grid 
    Area = np.pi * (r[1:]**2 - r[:-1]**2)
    M_gas = (sig_g[:-1] * Area).sum()
    # TODO implement the gas radius 
    n_gas_crit = 10**(21.27-0.53*np.log10(L_star/ac.L_sun.cgs.value))*(M_gas/c.M_sun)**(0.3-0.08*np.log10(L_star/ac.L_sun.cgs.value))
    sig_gas_crit = 10*mu*n_gas_crit

    for i in range(len(r) - 1, 0, -1):
        if sig_g[i] > sig_gas_crit:
            break
    r_gas = r[i]
    
    return SimpleNamespace(
        rf=rf,
        flux_t=flux_t,
        tau=tau,
        I_nu=I_nu,
        a=a,
        sig_da=sig_da,
        r_CO = r_gas)


def get_all_observables(d, opac, lam, amax=True, q=3.5, flux_fraction=0.68, scattering=True, inc=0.0):
    """Calculate the radius and total flux for all snapshots of a simulation

    Arguments:
    ---------- 

    d : namedtuple
        the output of read_dustpy_data, read_rosotti_data, or run_bump_model

    opac : instance of Opacity
        which opacity to use

    lam : float | array
        wavelength(s) of the observations

    amax : bool
        if True, will always use a power-law distribution, even if size distribution
        is available.

    q : float | list of two elements
        size distribution exponent, either a single float to be used everywhere
        or a two-element list to specify q_f and q_d to be used in the fragmentation
        and drift limited regimes, respectively.

    flux_fraction : float
        at which fraction of the total flux the effective radius is defined [-]

    scattering : bool
        whether or not to include scattering

    inc : float
        inclination, default is 0.0 = face-on. This is just treated as
        increasing the path length across the slab model.

    Returns:
    --------
    rf : array
        e.g. 68% effective radii for all snapshots [cm]

    flux : array
        total flux for all snapshots [Jy]
    """
    rf = []
    flux = []
    tau = []
    I_nu = []
    a = []
    sig_da = []
    r_CO = []
    t_disk = -1 # default value, will be overwritten if available

    q_f, q_d = q * np.ones(2)

    if amax is False and hasattr(d, 'sig_da'):
        _a = d.a
        sig_d = d.sig_da
    else:
        _a = None
        sig_d = d.sig_d

    for it in range(len(d.time)):
        
        if t_disk == -1 and d.M_disk[it] < 1e-6*c.M_sun:
            t_disk = d.time[it]
        # assign the correct q
        if amax is True:
            if hasattr(d, 'q'):
                q_array = -d.q[it,:]
            else:
                q_array = np.where(d.a_max[it, :] > np.minimum(
                d.a_fr[it, :], d.a_df[it, :]), q_f, q_d)
        else:
            q_array = -3.5* np.ones(len(d.r))

        if hasattr(d, "L_star"):
            L_star = d.L_star[it]
        else:
            L_star = 1.0 * ac.L_sun.cgs.value
        if hasattr(d, "mu"):
            mu = d.mu[it]
        else:
            mu = 2.3 * c.m_p

        
        obs = get_observables(d.r, d.sig_g[it, :], sig_d[it], d.a_max[it, :], d.T[it, :], opac, lam,
                              q=q_array, a=_a, flux_fraction=flux_fraction,L_star=L_star,mu=mu ,scattering=scattering, inc=inc)
        rf += [obs.rf]
        flux += [obs.flux_t]
        tau += [obs.tau]
        I_nu += [obs.I_nu]
        a += [obs.a]
        sig_da += [obs.sig_da]
        r_CO += [obs.r_CO]


    rf = np.array(rf)
    flux = np.array(flux)
    tau = np.array(tau)
    I_nu = np.array(I_nu)
    a = np.array(a)
    sig_da = np.array(sig_da)
    r_CO = np.array(r_CO)

    return SimpleNamespace(
        t = d.time,
        rf=rf,
        flux=flux,
        tau=tau,
        I_nu=I_nu,
        a=a,
        sig_da=sig_da,
        r_CO=r_CO,
        t_disk = t_disk 
    )
