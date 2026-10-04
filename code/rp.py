"""Rock-physics core for the basalt CO2-mineralization sensitivity study.

All moduli in Pa, densities in kg m^-3, velocities in m s^-1.
Single parameter set used for every result in the manuscript.
"""
import numpy as np

# --- Mineral (grain) and fluids: Adam & Otheim (2013) ---------------------
K_GR, MU_GR, RHO_GR = 80.1e9, 31.0e9, 2800.0
K_BR, RHO_BR = 2.237e9, 1040.0          # brine
K_CO2, RHO_CO2 = 0.159e9, 832.0         # supercritical CO2
K_DIS, RHO_DIS = 2.0e9, 1000.0          # CO2-charged water (assumption)

# --- Fractured corridor (assumed, not calibrated) -------------------------
PHI_C, KD_C, MU_C = 0.045, 21.0e9, 11.0e9
# --- Facies calibrated to SAG-P2/P3 log medians (Navarro, 2021) -----------
# Massive core: phi 3.76 %, Vp 5.6 km/s; vesicular top: phi 17.5 %, Vp 3.8 km/s.
# Vs is not constrained by the logs used here: Vp/Vs = 1.80 and 1.90 assumed.
PHI_MAS, VP_MAS, VPVS_MAS = 0.0376, 5600.0, 1.80
PHI_VES, VP_VES, VPVS_VES = 0.175, 3800.0, 1.90


def inverse_gassmann(vp, vs, phi, kfl=K_BR, rfl=RHO_BR, k0=K_GR, r0=RHO_GR):
    """Dry-frame moduli from brine-saturated Vp, Vs and porosity."""
    rho = (1 - phi) * r0 + phi * rfl
    mu = rho * vs ** 2
    ks = rho * vp ** 2 - 4 / 3 * mu
    kd = (ks * (phi * k0 / kfl + 1 - phi) - k0) / (phi * k0 / kfl + ks / k0 - 1 - phi)
    return kd, mu


KD_MAS, MU_MAS = inverse_gassmann(VP_MAS, VP_MAS / VPVS_MAS, PHI_MAS)
KD_VES, MU_VES = inverse_gassmann(VP_VES, VP_VES / VPVS_VES, PHI_VES)
# --- Cemented end member = calibrated massive-basalt frame ----------------
KD_MAX, MU_MAX = KD_MAS, MU_MAS

# --- Scenario parameters ---------------------------------------------------
DISSOLVED = dict(finf=0.40, tau=7.5, S=1.0, Kinj=K_DIS, rinj=RHO_DIS)
FREE = dict(finf=0.04, tau=9.0, S=0.6, Kinj=K_CO2, rinj=RHO_CO2)


def gassmann_ksat(phi, kd, kfl, k0=K_GR):
    return kd + (1 - kd / k0) ** 2 / (phi / kfl + (1 - phi) / k0 - kd / k0 ** 2)


def bulk_density(phi, rfl, r0=RHO_GR):
    return (1 - phi) * r0 + phi * rfl


def velocities(phi, kd, mu, kfl, rfl):
    ks = gassmann_ksat(phi, kd, kfl)
    rho = bulk_density(phi, rfl)
    return np.sqrt((ks + 4 / 3 * mu) / rho), np.sqrt(mu / rho), rho


def reuss_fluid(S, kinj, rinj):
    """Uniform (Wood/Reuss) mixing of brine and injected fluid."""
    return 1 / (S / kinj + (1 - S) / K_BR), S * rinj + (1 - S) * RHO_BR


def cement_fraction(t, finf, tau):
    return finf * (1 - np.exp(-np.asarray(t, float) / tau))


def frame(f, phi0=PHI_C, kd0=KD_C, mu0=MU_C, kmax=KD_MAX, mumax=MU_MAX):
    """Porosity loss and linear frame stiffening for cemented fraction f."""
    phi = phi0 * (1 - f)
    return phi, kd0 + f * (kmax - kd0), mu0 + f * (mumax - mu0)


def state(t, finf, tau, S, Kinj, rinj, stiff=1.0):
    """Vp, Vs, rho, phi of the corridor cell at time t (months)."""
    f = cement_fraction(t, finf, tau)
    phi, kd, mu = frame(f, kd0=KD_C * stiff, mu0=MU_C * stiff)
    kfl, rfl = reuss_fluid(S, Kinj, rinj)
    vp, vs, rho = velocities(phi, kd, mu, kfl, rfl)
    return vp, vs, rho, phi


def reference(stiff=1.0):
    vp, vs, rho = velocities(PHI_C, KD_C * stiff, MU_C * stiff, K_BR, RHO_BR)
    return vp, vs, rho


def patchy_vp(S, kinj, rinj, phi=PHI_C, kd=KD_C, mu=MU_C):
    """Patchy (Hill) limit: harmonic average of P-wave moduli."""
    def pmod(kfl):
        return gassmann_ksat(phi, kd, kfl) + 4 / 3 * mu
    m = 1 / (S / pmod(kinj) + (1 - S) / pmod(K_BR))
    rho = bulk_density(phi, S * rinj + (1 - S) * RHO_BR)
    return np.sqrt(m / rho), np.sqrt(mu / rho)


def hs_upper_dry(phi, k0=K_GR, mu0=MU_GR):
    """Hashin-Shtrikman upper bound for a dry (empty-pore) mineral frame."""
    z = mu0 / 6 * (9 * k0 + 8 * mu0) / (k0 + 2 * mu0)
    k = 1 / ((1 - phi) / (k0 + 4 / 3 * mu0) + phi / (4 / 3 * mu0)) - 4 / 3 * mu0
    mu = 1 / ((1 - phi) / (mu0 + z) + phi / z) - z
    return k, mu
