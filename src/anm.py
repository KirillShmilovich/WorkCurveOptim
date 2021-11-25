from abc import ABC, abstractmethod
import numpy as np

from .utils import anm_size


class BaseANM(ABC):
    @abstractmethod
    def get_x0(self):
        pass

    @abstractmethod
    def get_beam_profile(self, beam_profile):
        pass


class ANM(BaseANM):
    def __init__(self, ZeroCoeffs, Nmax=4, x_0_init=0, **kwargs):
        self.Nmax = Nmax
        self.ZeroCoeffs = np.array(ZeroCoeffs)
        self.x_0_init = x_0_init

        base_anm_size = anm_size(self.Nmax)
        if base_anm_size <= self.ZeroCoeffs.max():
            raise ValueError(
                f"Max ZeroCoeff: {self.ZeroCeoffs.max()}, while anm_size: {base_anm_size} (Nmax is {self.Nmax})"
            )

        self.anm_size = base_anm_size - self.ZeroCoeffs.size
        self.NonZeroIdxs = np.array(
            [i for i in range(base_anm_size) if i not in self.ZeroCoeffs]
        )
        self.base_anm_size = base_anm_size

    def get_x0(self):
        if isinstance(self.x_0_init, np.ndarray):
            return self.x_0_init
        elif isinstance(self.x_0_init, list):
            return np.array(self.x_0_init)
        else:
            if self.x_0_init > 0:
                return np.random.normal(scale=self.x_0_init, size=self.NonZeroIdxs.size)
            else:
                return np.zeros(self.NonZeroIdxs.size)

    def get_beam_profile(self, beam_profile):
        profile = np.zeros(self.base_anm_size)
        profile[self.NonZeroIdxs] = beam_profile
        return profile
