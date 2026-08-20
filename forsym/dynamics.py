import math
import numpy as np


def beam_deflection_param(radius, length, noise_std=1.0, E=3e9):
    """
    Calculate the beam deflection parameter (Ks).

    Parameters:
    - E: Elastic modulus
    - radius: Radius of the beam
    - length: Length of the beam
    - noise_std: # Adjust this value to control the amount of noise

    Returns:
    - Kp, Kd: Beam deflection parameters
    """

    kp = (E * math.pi * (radius**4)) / (2 * length)
    # Generate & add Gaussian noise
    if noise_std > 0.0:
        gauss_noise = np.random.normal(0, noise_std)
        kp += gauss_noise

    kd = kp / 10
    return round(max(kp, 2.0), 2), round(max(kd, 1.0), 2)

def rud_deflection_param(branch_level, base_kp=100, noise_std=1., ):
    """
    Use a rudimentary method for calculating stiffness parameters
    For L1 use base_kp, every subsequent level, /=2


    Returns:
    - Kp, Kd: Beam deflection parameters
    """

    if branch_level > 5:
        base_kp = base_kp * (2 ** (branch_level - 5))

    if base_kp <= 100:
        assert 0 < branch_level < 6, "Parameters can get too low"

    kp = base_kp / (2 ** (branch_level - 1))

    if noise_std > 0.:
        gauss_noise = np.random.normal(0, noise_std)
        kp += gauss_noise

    kd = kp / 5
    return round(max(kp, 2.), 2), round(max(kd, 2.), 2)




