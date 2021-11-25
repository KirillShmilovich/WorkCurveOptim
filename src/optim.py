import numpy as np
from joblib import Parallel, delayed
import cma
import pickle
from pathlib import Path

from .utils import A_0, nm
from .losses import dist_loss, angle_loss_smooth, phase_loss


class Optimizer:
    def __init__(
        self,
        dimer,
        anm,
        target_dist,
        target_angle,
        w_dist_orient=0.5,
        w_reg=500.0,
        w_phase=0.0,
        cma_sigma=0.25,
        cma_opts={"popsize": 100, "tolfun": 1e-3, "tolx": 1e-3},
        min_sep_dist=200 * nm,
        max_sep_fac=2,
        save_path=None,
        load_path=None,
        save_freq=10,
        **kwargs
    ):

        self.dimer = dimer
        self.anm_gen = anm
        self.target_angle = target_angle
        self.target_dist = target_dist
        self.w_dist_orient = w_dist_orient
        self.w_reg = w_reg
        self.w_phase = w_phase
        self.min_sep_dist = min_sep_dist
        self.max_sep_fac = max_sep_fac

        self.cma_sigma = cma_sigma
        self.cma_opts = cma_opts

        self.separation = np.linspace(
            self.min_sep_dist, target_dist + self.max_sep_fac * A_0 * nm, 100
        )
        self.thetas = np.linspace(0, np.pi, 100)

        if save_path is not None:
            self.save_path = Path(save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
        else:
            self.save_path = save_path

        self.save_freq = save_freq
        self.load_path = load_path
        if self.load_path is not None:
            self.es = pickle.load(open(self.load_path, "rb"))

    def loss_func(self, x):
        anm = self.anm_gen.get_beam_profile(x)
        W = self.dimer.calc_radial_work(self.separation, self.target_angle, anm)
        W_angle = self.dimer.calc_angular_work(self.thetas, self.target_dist, anm)

        dist = dist_loss(W, self.separation, self.target_dist)
        orient = angle_loss_smooth(W_angle, self.thetas, self.target_angle)
        reg = np.sum(x ** 2)
        phase = phase_loss(anm, self.dimer.Nmax, self.dimer.rho_scale)

        total_loss = (
            self.w_dist_orient * dist
            + (1 - self.w_dist_orient) * orient
            + self.w_reg * reg
            + self.w_phase * phase
        )
        return total_loss

    def _save(self, step):
        if self.save_path is not None:
            if step % self.save_freq == 0:
                pickle.dump(self.es, open(self.save_path / "optim.pkl", "wb"))

    def optimize(self):
        x_0 = self.anm_gen.get_x0()
        self.es = cma.CMAEvolutionStrategy(x_0, self.cma_sigma, inopts=self.cma_opts)

        step = 0
        while not self.es.stop():
            X = self.es.ask()

            results = Parallel(n_jobs=-1)(delayed(self.loss_func)(x) for x in X)
            # results = [self.loss_func(x) for x in X]
            losses = results
            # losses = [r[0] for r in results]
            # sep_losses = [r[1][0] for r in results]
            # orientation_losses = [r[1][1] for r in results]
            # reg_losses = [r[1][2] for r in results]

            self.es.tell(X, losses)
            # es.manage_plateaus()

            self.es.logger.add()
            self.es.disp()
            # print(f'SEP:{min(sep_losses):.5g}| ORIENT:{min(orientation_losses):.5g}| REG:{min(reg_losses):.5g}')

            step += 1

            self._save(step)

        print("termination:", self.es.stop())
        cma.s.pprint(self.es.best.__dict__)
        self._save(step)
