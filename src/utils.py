import numpy as np
from math import factorial
from scipy import constants
import miepy
import stoked
from topics.optical_matter.dynamics import (
    electrodynamics,
    cluster_get_orientation,
    cluster_get_anisotropic_drag,
)

ns = 1e-9
nm = 1e-9
kT = 300 * constants.k

A_0 = 578
A_1 = 1160


def get_position_2d(dist, theta):
    x = (dist / 2.0) * np.cos(theta)
    y = (dist / 2.0) * np.sin(theta)
    return np.array([[-x, -y, 0.0], [x, y, 0.0]])


def pickler(es, filename):
    open(filename, "wb").write(es.pickle_dumps())


def ms2steps(ms, dt=5000 * ns):
    return int(round((ms * 1e-3) / dt))


def anm_size(Nmax):
    return int((Nmax + 1) * (Nmax + 2) / 2) - 1


def radial_polynomial(n, m, rho):
    """Radial polynomial appearing in the definition of the Zernike polynomial"""
    if (n - m) % 2 == 1:
        return np.zeros_like(rho)

    R = np.zeros_like(rho)

    Nmax = int((n - m) / 2)
    for k in range(Nmax + 1):
        f = (
            (-1) ** k
            * factorial(n - k)
            / (factorial(k) * factorial((n + m) / 2 - k) * factorial((n - m) / 2 - k))
        )
        R += f * np.power(rho, n - 2 * k)

    return R


def zernike(n, m, rho, phi):
    """Zernike polynomial"""
    if m >= 0:
        return radial_polynomial(n, m, rho) * np.cos(m * phi)
    else:
        return radial_polynomial(n, -m, rho) * np.sin(-m * phi)


def zernike_index_gen(Nmax):
    """Generator for the (n,m) indices in Znm"""
    for n in range(1, Nmax + 1):
        for m in range(-n, n + 1, 2):
            yield n, m


# spatial-light-modulator source definition
def slm(theta, phi, anm, Nmax, rho_scale):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    rho = np.sqrt(x ** 2 + y ** 2)

    phase = np.zeros_like(theta)
    for i, (n, m) in enumerate(zernike_index_gen(Nmax)):
        phase += anm[i] * zernike(n, m, rho_scale * rho, phi)

    return phase


def stoked_from_cluster_2d(
    cluster,
    dt,
    temperature=300,
    viscosity=8e-4,
    hydrodynamic_coupling=False,
    unpolarized=False,
    force=None,
    torque=None,
    interface=False,
    integrator=None,
    inertia=None,
):
    if cluster.interface is not None:
        interface = stoked.interface()
    elif interface:
        interface = stoked.interface()
    else:
        interface = None

    if isinstance(cluster, miepy.cluster):
        radius = [p.enclosed_radius() for p in cluster.particles]

        interactions = [
            stoked.collisions_sphere(radius, 1),
            stoked.double_layer_sphere(
                radius=radius, potential=-77e-3, temperature=temperature, debye=1.0e-7
            ),
            electrodynamics(cluster, unpolarized),
        ]

        position = np.copy(cluster.position)
        orientation = cluster_get_orientation(cluster)
        drag = cluster_get_anisotropic_drag(cluster, viscosity)

    else:
        radius = cluster.radius

        interactions = [
            stoked.collisions_sphere(radius, 1),
            stoked.double_layer_sphere(
                radius=radius, potential=-77e-3, temperature=temperature, debye=1.0e-7
            ),
            electrodynamics(cluster, unpolarized),
        ]

        position = np.copy(cluster.position)
        orientation = None
        drag = stoked.drag_sphere(radius=radius, viscosity=viscosity)

    zval = cluster.position[0, 2]
    bd = stoked.stokesian_dynamics(
        position=position,
        orientation=orientation,
        drag=drag,
        temperature=temperature,
        dt=dt,
        constraint=stoked.constrain_position(z=zval),
        interactions=interactions,
        hydrodynamic_coupling=hydrodynamic_coupling,
        integrator=integrator,
        force=force,
        torque=torque,
        interface=interface,
        inertia=inertia,
    )

    return bd
