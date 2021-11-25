from math import factorial

import miepy
import numpy as np
from topics.optical_matter.dynamics import stoked_from_cluster_2d
from tqdm import tqdm

from .initial_factory import initial_factory


class Simulator:
    def __init__(
        self,
        width,
        power,
        init_anm,
        nmax,
        initial,
        polarization,
        rho_scale=3,
        dt=5000 * 1e-9,
    ):
        nm = 1e-9
        ns = 1e-9
        self.Ag = miepy.materials.Ag()
        self.water = miepy.materials.water()
        self.radius = 75 * nm
        self.width = width
        self.wavelength = 800 * nm
        self.polarization = polarization

        self.nmax = nmax  # max order in Zernike polynomial
        self.rho_scale = rho_scale  # scale the rho values in Znm
        self.anm_size = int((self.nmax + 1) * (self.nmax + 2) / 2) - 1
        if init_anm is None:
            self.anm = np.zeros([self.anm_size], dtype=float)  # coefficient array
        else:
            self.anm = init_anm

        self.e_field_sampling = 50  # angular resolution of the SLM

        # dependent variables (don't change)
        self.k = 2 * np.pi * 1.33 / self.wavelength
        self.lmax = 2

        self.source = miepy.sources.gaussian_beam(
            self.width, polarization=self.polarization, power=power
        )
        self.source = miepy.sources.phase_only_slm(self.source, self.slm)

        self.initial = initial_factory(initial)()

        self.cluster = miepy.sphere_cluster(
            position=self.initial,
            radius=self.radius,
            material=self.Ag,
            lmax=self.lmax,
            source=self.source,
            wavelength=self.wavelength,
            medium=self.water,
        )

        self.bd = stoked_from_cluster_2d(self.cluster, dt=dt)
        self.global_pos = None

    @property
    def xyz(self):
        return self.global_pos

    def sim(self, n_steps, use_tqdm=False):
        n_steps = int(n_steps)
        self.pos = np.empty((n_steps, self.initial.shape[0], 3))
        if use_tqdm:
            for i in tqdm(range(n_steps)):
                self.pos[i] = self.bd.position
                self.bd.step()

        else:
            for i in range(n_steps):
                self.pos[i] = self.bd.position
                self.bd.step()

        if self.global_pos is None:
            self.global_pos = self.pos
        else:
            self.global_pos = np.concatenate((self.global_pos, self.pos), axis=0)

    def setSource(self, anm, power=None):
        assert anm.shape == self.anm.shape
        self.anm = anm
        if power is not None:
            self.source.power = power
        self.refresh_source(self.source)

    def slm(self, theta, phi):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        rho = np.sqrt(x ** 2 + y ** 2)

        phase = np.zeros_like(theta)
        for i, (n, m) in enumerate(self.zernike_index_gen(self.nmax)):
            phase += self.anm[i] * self.zernike(n, m, self.rho_scale * rho, phi)

        return phase

    def radial_polynomial(self, n, m, rho):
        """Radial polynomial appearing in the definition of the Zernike polynomial"""
        if (n - m) % 2 == 1:
            return np.zeros_like(rho)

        R = np.zeros_like(rho)

        nmax = int((n - m) / 2)
        for k in range(nmax + 1):
            f = (
                (-1) ** k
                * factorial(n - k)
                / (
                    factorial(k)
                    * factorial((n + m) / 2 - k)
                    * factorial((n - m) / 2 - k)
                )
            )
            R += f * np.power(rho, n - 2 * k)

        return R

    def zernike(self, n, m, rho, phi):
        """Zernike polynomial"""
        if m >= 0:
            return self.radial_polynomial(n, m, rho) * np.cos(m * phi)
        else:
            return self.radial_polynomial(n, -m, rho) * np.sin(m * phi)

    def zernike_index_gen(self, nmax):
        """Generator for the (n,m) indices in Znm"""
        for n in range(1, nmax + 1):
            for m in range(-n, n + 1, 2):
                yield n, m

    def refresh_source(self, src):
        """refresh a source after the Zernike coefficients have been changed"""
        theta_c = src.theta_cutoff(self.k)
        src.p_src_func = miepy.vsh.decomposition.integral_project_source_far(
            src,
            self.k,
            self.lmax,
            theta_0=np.pi - theta_c,
            sampling=self.e_field_sampling,
        )
