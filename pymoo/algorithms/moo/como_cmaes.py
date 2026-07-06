"""COMO-CMA-ES — multi-objective CMA-ES via the Sofomore framework (comocma)."""

import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.docs import parse_doc_string
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.normalization import ZeroToOneNormalization


class COMOCMAES(Algorithm):
    """COMO-CMA-ES (Toure, Auger, Brockhoff & Hansen, GECCO 2019).

    A thin pymoo wrapper around the authors' ``comocma`` package. COMO-CMA-ES
    instantiates the Sofomore framework with a set of CMA-ES kernels that jointly
    maximize the Uncrowded Hypervolume Improvement (UHVI), giving strong anytime
    bi-objective performance — the CMA-ES family that is competitive with the
    documented bbob-biobj winners, which plain MO-CMA-ES is not at small budgets.

    Requires ``comocma`` (``pip install comocma``) and is **bi-objective only**.
    The search runs in a normalized ``[0, 1]`` decision space; the hypervolume
    reference point is estimated from the initial sample.
    """

    def __init__(self, pop_size=32, sigma=0.2, reference_point=None,
                 sampling=FloatRandomSampling(), output=MultiObjectiveOutput(), **kwargs):
        """Initialize COMO-CMA-ES.

        Args:
            pop_size: Number of CMA-ES kernels — roughly the number of Pareto-front
                points sought. Each kernel is a full CMA-ES.
            sigma: Initial step size in the normalized ``[0, 1]`` decision space.
            reference_point: Hypervolume reference point for the UHVI indicator.
                COMO-CMA-ES is sensitive to it; pass the problem's known nadir when
                available. If ``None`` it is estimated from the initial sample,
                which works when the objective ranges are stable but can be poor
                on problems whose scale collapses during convergence (e.g. ZDT).
            sampling: Sampling used for the initial population / reference estimate.
            output: Display output used to report progress during the run.
            **kwargs: Additional keyword arguments forwarded to ``Algorithm``.
        """
        super().__init__(output=output, **kwargs)
        self.pop_size = pop_size
        self.sigma0 = sigma
        self.reference_point = reference_point
        self.sampling = sampling
        self.norm = None
        self.moes = None
        self._sols = None

    def _setup(self, problem, **kwargs):
        if problem.n_obj != 2:
            raise ValueError("COMO-CMA-ES (comocma) supports only bi-objective problems.")
        self.norm = ZeroToOneNormalization(problem.xl, problem.xu)

    def _initialize_infill(self):
        return self.sampling(self.problem, self.pop_size, random_state=self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        try:
            import comocma
        except ImportError as exc:
            raise ImportError(
                "COMO-CMA-ES requires the comocma package — install it with "
                "`pip install comocma`."
            ) from exc

        n = self.problem.n_var
        rs = self.random_state

        if self.reference_point is not None:
            ref_point = list(self.reference_point)
        else:
            # estimate: nadir of the initial sample, pushed out by a margin
            F = infills.get("F")
            lo, hi = F.min(axis=0), F.max(axis=0)
            ref_point = (hi + 0.1 * (hi - lo)).tolist()

        x0 = [list(rs.random(n)) for _ in range(self.pop_size)]
        seed = int(rs.integers(0, 2**31 - 1))
        cmas = comocma.get_cmas(
            x0, self.sigma0,
            inopts={"bounds": [[0.0] * n, [1.0] * n], "verbose": -9, "seed": seed},
        )
        self.moes = comocma.Sofomore(cmas, reference_point=ref_point)
        self.pop = infills

    def _infill(self):
        self._sols = self.moes.ask()
        return Population.new(X=self.norm.backward(np.array(self._sols)))

    def _advance(self, infills=None, **kwargs):
        self.moes.tell(self._sols, infills.get("F").tolist())
        Fp = np.array(self.moes.pareto_front_cut, dtype=float)
        Xp = np.array(self.moes.pareto_set_cut, dtype=float)
        # the cut set and front can momentarily desync in length when kernels go
        # inactive — only update the population when they align.
        m = min(len(Fp), len(Xp))
        if m > 0:
            self.pop = Population.new(X=self.norm.backward(Xp[:m]), F=Fp[:m])


parse_doc_string(COMOCMAES.__init__)
