import numpy as np
import miepy
from functools import partial
from scipy.integrate import cumtrapz
from tqdm import tqdm


from .utils import nm, get_position_2d, slm, stoked_from_cluster_2d


class DIMER:
    def __init__(
        self,
        width=400 * nm,
        polarization=[1, 1j],
        Nmax=4,
        rho_scale=1.0,
        power=0.25,
        radius=75 * nm,
        lmax=2,
        wavelength=770 * nm,
        e_field_sampling=50,
        **kwargs
    ):
        self.Ag = miepy.materials.Ag()
        self.water = miepy.materials.water()
        self.Nmax = Nmax
        self.rho_scale = rho_scale
        self.polarization = polarization
        self.radius = radius
        self.lmax = lmax
        self.width = width

        # beam parameters variables
        self.wavelength = wavelength

        # anm_size = int((Nmax + 1) * (Nmax + 2) / 2) - 1

        # convergence parameters
        self.e_field_sampling = e_field_sampling  # angular resolution of the SLM

        # dependent variables (don't change)
        self.k = 2 * np.pi * 1.33 / self.wavelength

    def _get_source(self, anm):
        source = miepy.sources.gaussian_beam(self.width, polarization=self.polarization)
        source = miepy.sources.phase_only_slm(
            source, partial(slm, anm=anm, Nmax=self.Nmax, rho_scale=self.rho_scale)
        )
        return source

    def _get_dimer(self, init_sep, init_theta, source):
        initial = get_position_2d(init_sep, init_theta)

        dimer = miepy.sphere_cluster(
            position=initial,
            radius=self.radius,
            material=self.Ag,
            source=source,
            wavelength=self.wavelength,
            medium=self.water,
            lmax=self.lmax,
        )
        return dimer

    def _get_bd(self, dimer, dt):
        bd = stoked_from_cluster_2d(dimer, dt)
        return bd

    def calc_radial_work(self, separation, theta, anm):

        source = self._get_source(anm)
        dimer = self._get_dimer(separation[0], theta, source)

        Fx = np.zeros([len(separation), 2])
        Fy = np.zeros([len(separation), 2])

        for i, sep in enumerate(separation):
            dimer.update_position(get_position_2d(sep, theta))
            Fx[i] = dimer.force()[:, 0]
            Fy[i] = dimer.force()[:, 1]

        # project force along theta4
        Ft0 = Fx[:, 0] * np.cos(theta) + Fy[:, 0] * np.sin(theta)
        Ft1 = Fx[:, 1] * np.cos(theta) + Fy[:, 1] * np.sin(theta)

        W = cumtrapz(Ft0 - Ft1, separation, initial=0)
        return W

    def calc_angular_work(self, thetas, sep, anm):

        source = self._get_source(anm)
        dimer = self._get_dimer(sep, thetas[0], source)

        Fx = np.zeros([len(thetas), 2])
        Fy = np.zeros([len(thetas), 2])

        for i, theta in enumerate(thetas):
            dimer.update_position(get_position_2d(sep, theta))
            Fx[i] = dimer.force()[:, 0]
            Fy[i] = dimer.force()[:, 1]

        # project force along theta
        Ft0 = -Fx[:, 0] * np.sin(thetas) + Fy[:, 0] * np.cos(thetas)
        Ft1 = -Fx[:, 1] * np.sin(thetas) + Fy[:, 1] * np.cos(thetas)

        W = sep * cumtrapz(Ft0 - Ft1, thetas, initial=0)
        return W

    def sim(self, init_sep, theta, n_steps, anm, dt=5000 * nm):
        source = self._get_source(anm)
        dimer = self._get_dimer(init_sep, theta, source)
        bd = self._get_bd(dimer, dt)

        n_steps = int(n_steps)
        pos = np.empty((n_steps, 2, 3))
        for i in tqdm(range(n_steps)):
            pos[i] = bd.position
            bd.step()
        return pos
