from __future__ import annotations

import math
import multiprocessing
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from multiprocessing.managers import ValueProxy
from time import time
from typing import Any, Callable, Literal, Optional, TypeVar, get_args

import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch import gen_candidates_scipy
from botorch.acquisition import AcquisitionFunction, qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.exceptions.errors import InfeasibilityError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.multitask import Standardize
from botorch.models.transforms.input import Normalize
from botorch.optim.initializers import sample_q_batches_from_polytope
from botorch.optim.optimize import optimize_acqf
from botorch.optim.utils import fix_features
from dataclasses_json import config, dataclass_json
from gpytorch.kernels import MaternKernel
from numpy.typing import NDArray
from torch import Tensor

from expr_parser import NonlinearInequalityConstraint, parse_nonlinear_constraint
from mylog import get_logger

logger = get_logger(__file__)
# logging.captureWarnings(True)


ConstraintPart = tuple[float, int]
Constraint = tuple[list[ConstraintPart], float]

OPT_TIMEOUT_SEC = 150.0
CATEGORY_SCORE_NAME = "%%combination score"


def todo(msg):
    print(f"TODO: {msg}")
    exit(1)


def cpu_cores_cgroups_v1() -> Optional[int]:
    try:
        with open("/sys/fs/cgroup/cpu/cfs_quota_us", "rb") as file:
            quota = int(file.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "rb") as file:
            period = int(file.read())
    except FileNotFoundError:
        return None

    if quota <= 0 or period <= 0:
        return None

    return math.ceil(quota / period)


def cpu_cores_cgroups_v2() -> Optional[int]:
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as file:
            line = file.read()
            try:
                max, period = line.split()
                max = int(max)
                period = int(period)
            except ValueError:
                # invalid format - just ignore
                return None

    except FileNotFoundError:
        return None

    if max == "max":
        return multiprocessing.cpu_count()
    if max <= 0 or period <= 0:
        return None

    return math.ceil(max / period)


def cpu_cores() -> int:
    # We don't know where this is going to be running. therefore try cgroups
    # v1 and v2, and fall back to number of cores available when both fail

    cores = cpu_cores_cgroups_v1()
    if cores is not None:
        return cores

    cores = cpu_cores_cgroups_v2()
    if cores is not None:
        return cores

    return multiprocessing.cpu_count()


def ensure_n_blank_of(
    input_indices: list[int], n: int
) -> Callable[[Tensor], float | Tensor]:
    def f(t: Tensor) -> Tensor | float:
        s = torch.randint(0, len(input_indices), size=[n])
        return -t[s].abs().sum(dim=0)

    return f


def parse_val_fac(vf: str) -> tuple[float, str] | str:
    """Parse a Value-factor pair of the form `fac * value_name` to a pair of float and string"""
    vals = vf.split("*")

    if len(vals) > 2:
        return f"Could not parse constraint part '{vf}'. Needs to be of the form 'factor * value' or just 'value' where value is a column name in your inputs."

    if len(vals) == 2:
        first, second = vals
        second = second.strip()
        first = first.strip()
    else:  # len(vals) == 1
        first, second = 1.0, vals[0].strip(' "')

    try:
        first = float(first)
    except ValueError as _:
        return f"Factors for the components need to be numbers. Got '{first}'"

    return first, second


def parse_constraint(c: str, columns: list[str]) -> tuple[Constraint, str] | str:
    if len(c) == 0:
        return "Empty constraint."

    try:
        values_factors, rhs = c.split("=")
    except ValueError:
        return "Equality constraint needs to have a left and a right hand side."

    try:
        rhs = float(rhs.strip())
    except ValueError:
        return f"Invalid equality right hand side: '{rhs}. Needs to be a number."

    def validate_parsed_vf(
        maybe_valid: list[tuple[float, str] | str],
    ) -> list[ConstraintPart] | str:
        valid = []
        for parsed in maybe_valid:
            if isinstance(parsed, str):
                return parsed

            fac, val = parsed
            try:
                val = columns.index(val)
            except ValueError:
                return f"Invalid constraint, could not find column '{val}'"

            valid.append((fac, val))

        return valid

    parsed_vf = list(map(parse_val_fac, values_factors.split(",")))

    validated_vf = validate_parsed_vf(parsed_vf)
    if isinstance(validated_vf, str):
        return validated_vf

    normalized_lhs = []
    for v in parsed_vf:
        assert not isinstance(v, str)
        fac, name = v

        normalized_lhs.append(f"{fac} * {name}")
    normalized_lhs = ", ".join(normalized_lhs)
    normalized = f"{normalized_lhs} = {rhs}"

    return (validated_vf, rhs), normalized


def botorch_compliant_constraint(c: Constraint) -> tuple[Tensor, Tensor, float]:
    coefs, indices = list(zip(*c[0]))
    rhs = c[1]
    indices = torch.tensor(indices)
    return (
        indices,
        torch.tensor(coefs).double(),
        rhs,
    )


def bounds_to_dict(bounds: pd.DataFrame):
    return bounds.reset_index().to_dict(orient="records", index=True)


@dataclass_json
@dataclass
class RawConstraint:
    normalized_constraint: str
    nonlinear: bool


T = TypeVar("T")


def find_in(values: list[T], t: T) -> int | None:
    for i, value in enumerate(values):
        if value == t:
            return i
    return None


def backwards_compatible_decode_fix_n_values(
    vals: list[tuple[str, float]] | dict[str, float],
):
    if isinstance(vals, dict):
        return vals

    return {k: v for k, v in vals}


@dataclass_json
@dataclass
class FixNConstraint:
    values: dict[str, float] = field(
        metadata=config(decoder=backwards_compatible_decode_fix_n_values)
    )
    n: int


TargetTreatment = Literal["maximize", "minimize", "ignore"]


@dataclass_json
@dataclass
class OptimizeConfig:
    input_names: list[str]
    target_names: list[str]
    constraints_raw: list[RawConstraint]

    parsed_constraints: list[Constraint | NonlinearInequalityConstraint] = field(
        default_factory=list, metadata=config(exclude=lambda _: True)
    )
    bounds: pd.DataFrame = field(
        default_factory=pd.DataFrame,
        metadata=config(
            decoder=partial(pd.DataFrame.from_records, index="bound"),
            encoder=bounds_to_dict,
        ),
    )
    target_treatment: dict[str, TargetTreatment] = field(default_factory=dict)
    fix_n: FixNConstraint = field(default_factory=lambda: FixNConstraint(dict(), -1))
    num_samples: int = 5
    fix_n_temperature: float = 25.0

    def __post_init__(self):
        possible_treatments = get_args(TargetTreatment)
        for target, treatment in self.target_treatment.items():
            if treatment not in possible_treatments:
                raise ValueError(
                    f"Treatment of target {target} is invalid. expected 'minimize', 'maximize' or ignore. got {treatment}"
                )

        for raw_c in self.constraints_raw:
            if raw_c.nonlinear:
                ret = parse_nonlinear_constraint(
                    raw_c.normalized_constraint, self.input_names
                )
                if isinstance(ret, str):
                    raise ValueError(
                        f"invalid nonlinear constraint in config: {raw_c.normalized_constraint}"
                    )
                normalized = ret.normalized_repr()
            else:
                ret = parse_constraint(raw_c.normalized_constraint, self.input_names)
                if isinstance(ret, str):
                    raise ValueError(
                        f"invalid linear equality constraint in config: {raw_c.normalized_constraint}"
                    )
                ret, normalized = ret
            self.parsed_constraints.append(ret)
            raw_c.normalized_constraint = normalized
            # new_raw_c.append(RawConstraint(normalized))
        # self.constraints_raw = new_raw_c

    def remove_constraint(self, idx: int, uuid: str):
        assert len(self.parsed_constraints) == len(self.constraints_raw)

        if idx > (len(self.constraints_raw) - 1):
            logger.error(
                "index larger than number of raw constraints", extra={"uuid": uuid}
            )
            raise IndexError()

        raw = self.constraints_raw.pop(idx)
        self.parsed_constraints.pop(idx)
        return raw.normalized_constraint

    def __eq__(self, other: object):
        if not isinstance(other, OptimizeConfig):
            return False
        return self.to_json() == other.to_json()  # type: ignore


def fit_callback(
    _,
    fun,
    uuid: str = "you-forgot-to-pass-the-uuid",
    ui_callback: Callable[[str], None] | None = None,
):
    msg = f"Fitting Model (step {fun.step}), objective fn={fun.fval:.3f}"
    logger.debug(
        msg,
        extra={"uuid": uuid},
    )
    if ui_callback:
        ui_callback(msg)


def fit_model(opt: Optimization, callback: Callable[[str], None]) -> dict[str, Any]:
    model = get_model(opt)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={
            "callback": partial(fit_callback, uuid=opt.uuid, ui_callback=callback),
            "options": {"maxiter": 500},
        },
    )
    return model.state_dict()


BotorchConstraint = tuple[Tensor, Tensor, float]


@dataclass
class Optimization:
    X: Tensor
    Y: Tensor
    cfg: OptimizeConfig
    uuid: str

    X_min: Tensor = field(init=False)
    X_max: Tensor = field(init=False)
    Y_opt: Tensor = field(init=False)

    opt_names: list[str] = field(init=False)

    bounds: Tensor = field(init=False)
    eq_constraints: list[BotorchConstraint] = field(init=False)
    nlics: list[tuple[NonlinearInequalityConstraint, bool]] = field(init=False)

    def __post_init__(self):
        keep = []
        opt_names = []
        for i, t in enumerate(self.cfg.target_names):
            try:
                target_treatment = self.cfg.target_treatment[t]
                if target_treatment == "ignore":
                    keep.append(False)
                elif target_treatment == "minimize":
                    self.Y[:, i] *= -1
                    keep.append(True)
                    opt_names.append(t)
                else:
                    keep.append(True)
                    opt_names.append(t)
            except KeyError:
                keep.append(False)

        self.Y_opt = self.Y[:, keep]
        self.opt_names = opt_names

        self.bounds = torch.tensor(
            self.cfg.bounds.loc[self.cfg.input_names].values.T
        ).double()

        self.eq_constraints = [
            botorch_compliant_constraint(x)
            for x in self.cfg.parsed_constraints
            if not isinstance(x, NonlinearInequalityConstraint)
        ]

        self.nlics = [
            (x, True)
            for x in self.cfg.parsed_constraints
            if isinstance(x, NonlinearInequalityConstraint)
        ]

        logger.info("Initialized Optimization", extra={"uuid": self.uuid})


def comb(N: int, P: int, L: int) -> list[int]:
    """Find a combination with P members from a maximum of N elements

    adapted from

    Buckles, Bill P., and Matthew Lybanon.
    "Algorithm 515: Generation of a vector from the lexicographical index [G6]."
    ACM Transactions on Mathematical Software (TOMS) 3.2 (1977): 180-182.

    https://doi.org/10.1145/355732.355739


    Args:
        N: Number of elements
        P: Number of elements in the combination
        L: lexicographical index of the combination

    Returns:

    """
    L += 1
    c = [0] * P
    K = 0
    P1 = P - 1
    for i in range(0, P1):
        c[i] = 0
        if i != 0:
            c[i] = c[i - 1]

        r = None
        while True:
            c[i] += 1
            n_c = N - c[i]
            r = math.comb(n_c, P - 1 - i)
            K = K + r
            if K >= L:
                break

        assert r is not None
        K = K - r

    c[P - 1] = c[P - 2] + L - K

    for i in range(len(c)):
        c[i] -= 1

    return c


def comb_idx(C: list[int] | Tensor | NDArray[np.int32], N: int):
    """get the lexicographical index of a combination

    adapted from

    Buckles, Bill P., and Matthew Lybanon.
    "Algorithm 515: Generation of a vector from the lexicographical index [G6]."
    ACM Transactions on Mathematical Software (TOMS) 3.2 (1977): 180-182.

    https://doi.org/10.1145/355732.355739


    Args:
        C: combination to get index for
        N: number of possibilities for each element

    Returns:

    """
    P = len(C)

    x = 0
    for j in range(P):
        if j == 0:
            cj1 = 0
        else:
            cj1 = C[j - 1] + 1

        cj = C[j] + 1

        if cj1 + 1 > cj - 1:
            continue

        for k in range(cj1 + 1, cj):
            x += math.comb(N - k, P - j - 1)

    return x


def empirical_ei(means: Tensor, stddevs: Tensor, best_y: Tensor):
    z = (means - best_y) / stddevs
    return stddevs * h(z)


def h(z: Tensor) -> Tensor:
    std_normal = torch.distributions.Normal(
        loc=torch.tensor(0.0), scale=torch.tensor(1.0)
    )
    return std_normal.log_prob(z).exp() + std_normal.cdf(z) * z


FixedSamples = list[tuple[int, str]]


def select_combinations(
    X: Tensor,
    Y: Tensor,
    num_samples: int,
    cfg: OptimizeConfig,
    eps: float = 1e-4,
    temperature: float = 50.0,
    output_probabilities: bool = True,
) -> tuple[list[tuple[FixedSamples, int, float]], Tensor]:
    """select combionations of input variables

    selects combinations of input variables depending on a heuristic based on
    the expected improvement

    Args:
        X: inputs
        Y: outputs
        num_samples: number of combinations to select
        cfg: optimization config
        eps: epsilon for determining if a constraint is met in the input
        temperature: temperature scaling factor
        output_probabilities: output probabilities instead of the expected improvement in sample scores

    Returns:
        the picked samples and category probabilities

    Raises:
        ValueError:
    """
    num_avail = len(cfg.fix_n.values)
    num_fixed = cfg.fix_n.n

    samples = torch.ones(X.shape[0], num_avail, dtype=torch.int32) * -1
    indices = torch.zeros(X.shape[0], dtype=torch.int32)
    # we rely on cfg.fix_n.values not being updated after this call.

    # Where more of the fix-n constraints are satisfied in the inputs than set
    # using cfg.fix_n.n, those should count towards all categories where
    # they are valid
    # TODO: can this become an issue somehow?
    for sample_idx, (name, v) in enumerate(cfg.fix_n.values.items()):
        idx = find_in(cfg.input_names, name)
        # here, we get the index from input names
        # they correspond to positions in the inputs (X) tensor
        # later on
        assert not isinstance(idx, str)

        fixed_in_samples = (X[:, idx] - v).abs() < eps
        fixed_indices = indices[fixed_in_samples]
        samples[fixed_in_samples, fixed_indices] = sample_idx
        indices[fixed_in_samples] += 1

    if not (indices >= num_fixed).all():
        (indices,) = list(np.where((indices < num_fixed).numpy()))
        indices = indices.tolist()
        raise ValueError(
            f"Invalid Optimization Config. Inputs must fulfill the Fix-N constraint. Constraint broken at positions: {indices}"
        )

    categories_by_sample = []
    for s in samples:
        if (s != -1).sum() > num_fixed:
            sample_cats = []
            valid_ones = s[s != -1]
            for c in torch.combinations(valid_ones, num_fixed):
                sample_cats.append(comb_idx(c, num_avail))
            categories_by_sample.append(valid_ones)
        else:
            categories_by_sample.append([comb_idx(s[:num_fixed], num_avail)])

    # categories = torch.tensor([comb_idx(c, n) for c in samples])
    n_categories = math.comb(num_avail, num_fixed)

    # initialize to overall mean and stddev
    means = torch.ones(n_categories, Y.shape[1]) * Y.mean(dim=0)
    stddevs = torch.ones(n_categories, Y.shape[1]) * Y.std(dim=0)
    best_y = Y.max(dim=0).values

    all_sample_categories = set()
    for categories in categories_by_sample:
        for category_id in categories:
            all_sample_categories.add(category_id)

    # TODO: find some smarter way to update the posterior for standard deviation
    # and mean rather than just replacing the overall mean and std dev
    for category_id in all_sample_categories:
        samples_with_category = torch.tensor(
            [category_id in cat for cat in categories_by_sample]
        )
        if samples_with_category.sum() > 0:
            means[category_id] = Y[samples_with_category].mean(dim=0)

        if samples_with_category.sum() > 1:
            stddevs[category_id] = Y[samples_with_category].std(dim=0)

    ei = empirical_ei(means, stddevs, best_y)
    if ei.shape[1] != 0:
        ei = ei.prod(dim=1)

    probabilities = (ei * temperature).softmax(dim=0)
    rng = np.random.default_rng(1234)

    choices = rng.choice(
        np.arange(n_categories), p=probabilities, replace=True, size=num_samples
    )

    names = list(cfg.fix_n.values.keys())

    collapsed = []
    for c in range(n_categories):
        found_n = int((choices == c).sum())
        if found_n > 0:
            combination = comb(num_avail, num_fixed, c)
            if output_probabilities:
                score = probabilities[c]
            else:
                score = ei[c]

            combination = [
                (find_in(cfg.input_names, names[c]), names[c]) for c in combination
            ]
            for c in combination:
                assert not isinstance(c[0], str)

            collapsed.append((combination, found_n, score))

    return collapsed, probabilities


def no_log(_: str):
    pass


class UserInterrupt(Exception):
    def __init__(self):
        pass


def wrap_gen_candidates(
    initial_conditions: Tensor,
    acqf: AcquisitionFunction,
    *args,
    uuid: str = "",
    ui_callback: Callable[[str], None] = no_log,
    sample_numbers: str | None = None,
    **kwargs,
):
    if sample_numbers is not None:
        of = f"for {sample_numbers}"
    else:
        of = ""
    logger.info(f"Optimizing initial conditions {of}", extra={"uuid": uuid})
    ui_callback(f"Optimizing initial conditions {of}")
    cc, ba = gen_candidates_scipy(initial_conditions, acqf, *args, **kwargs)

    return cc, ba


def check_equality_constraints_q_batch(
    data: Tensor, constraints: list[tuple[Tensor, Tensor, float]] | None = None
) -> Tensor:
    if constraints is None:
        return torch.ones([data.shape[0]], dtype=torch.bool)

    ok = torch.ones(data.shape[0], dtype=torch.bool)
    for indices, coefs, rhs in constraints:
        prod_sum = (data[:, indices] * coefs).sum(dim=1)
        is_close = prod_sum.isclose(torch.tensor(rhs).double())
        ok[~is_close] = False
    return ok


EqConstraint = tuple[Tensor, Tensor, float]


def fix_feature_in_equality_constraint(
    eq_constraint: EqConstraint, fix_index: int, value: float
) -> tuple[Tensor, Tensor, float]:
    """adjust the constraint such that it holds with the given feature fixed.

    For an equality constraint
    ```
    a * x + b * y + c * z = rhs
    ```
    and if feature x is fixed to value F,
    we compute a * x, and subtract it from the right hand side.
    subsequently, the bound on x is removed from the constraint.


    Args:
        eq_constraint: constraint to modify
        fix_index: index of fixed feature
        value: value the feature is fixed to

    Returns:
        modified constraint
    """
    indices, coefs, rhs = eq_constraint

    fix_coef = None
    del_idx = None
    for i, (coef, idx) in enumerate(zip(coefs, indices)):
        if idx == fix_index:
            fix_coef = coef
            del_idx = i
            break

    if fix_coef is None:
        # nothing to do if constraint doesnt care about the fixed feature
        return eq_constraint
    assert del_idx is not None

    rhs = rhs - fix_coef * value
    keep = torch.arange(indices.shape[0]) != del_idx
    indices = indices[keep]
    coefs = coefs[keep]

    return indices, coefs, float(rhs)


def fix_equality_constraints(
    equality_constraints: list[EqConstraint] | None = None,
    fixed_features: dict[int, float] | None = None,
) -> tuple[list[EqConstraint] | None, list[EqConstraint] | None]:
    old_constraints = None
    if fixed_features is not None and equality_constraints is not None:
        new_eq_constraints = []

        for constraint in equality_constraints:
            for idx, value in fixed_features.items():
                constraint = fix_feature_in_equality_constraint(constraint, idx, value)
            new_eq_constraints.append(constraint)

        old_constraints = equality_constraints
        equality_constraints = new_eq_constraints

    return old_constraints, equality_constraints


@dataclass
class RestartCandidateGen:
    bounds: Tensor
    uuid: str | None
    raw_samples: int
    q: int
    num_restarts: int
    nlics: list[tuple[Callable[[Tensor], Tensor | float], bool]]
    acq_function: AcquisitionFunction
    opt: Optimization
    model_state_dict: dict[str, Any]
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None
    check_constraints: list[tuple[Tensor, Tensor, float]]
    fixed_features: dict[int, float] | None

    def __call__(
        self,
        restart: int,
        rng: torch.Generator,
        ui_callback: Callable[[str], None] | None,
    ):
        ok = torch.zeros(self.bounds.shape[1]).bool()
        X = None
        i = 0
        points = []

        SIZES = [2, 4, 6, 8, 10]
        N_TRIES = len(SIZES)

        # acq_function = get_acqf(self.opt, self.model_state_dict)

        with torch.no_grad():
            for i, factor in enumerate(SIZES):
                logger.info(
                    f"restart {restart + 1}: Trying to create initial conditions, trial {i + 1}/{N_TRIES}",
                    extra={"uuid": self.uuid},
                )
                if ui_callback:
                    ui_callback(
                        f"restart {restart + 1}: Trying to create initial conditions, trial {i + 1}/{N_TRIES}"
                    )

                seed = int(torch.randint(10000000, [1], generator=rng).item())
                logger.info(
                    f"restart {restart + 1}: polytope sampling seed: {seed}",
                    extra={"uuid": self.uuid},
                )

                # sample until we have enough valid points
                X = sample_q_batches_from_polytope(
                    n=self.raw_samples * factor,
                    q=self.q,
                    bounds=self.bounds,
                    n_burnin=5_000,
                    n_thinning=32,
                    seed=seed,
                    equality_constraints=self.equality_constraints,
                    inequality_constraints=self.inequality_constraints,
                )
                logger.info(
                    f"restart {restart + 1}: sampled from polytope",
                    extra={"uuid": self.uuid},
                )

                if self.fixed_features is not None:
                    X = (
                        fix_features(X, fixed_features=self.fixed_features)
                        .cpu()
                        .detach()
                    )

                for qi in range(self.q):
                    # just to be sure :)
                    ok = check_equality_constraints_q_batch(
                        X[:, qi], self.check_constraints
                    )
                    assert ok.all(), "Error in constraint manipulation"

                try_again = False
                for qi in range(self.q):
                    q_batch = X[:, qi]
                    ok = torch.ones(q_batch.shape[0]).bool()
                    logger.info(
                        f"restart {restart + 1}: checking samples for q: {qi + 1}/{self.q}.",
                        extra={"uuid": self.uuid},
                    )
                    if ui_callback:
                        ui_callback(
                            f"restart {restart + 1}: checking candidate q: {qi + 1}/{self.q}.",
                        )

                    constraint_hardness = []
                    for constraint, intra_point in self.nlics:
                        if not intra_point:
                            logger.error(
                                "Inter-Point constraints are not implemented yet.",
                                extra={"uuid": self.uuid},
                            )
                            assert (
                                False
                            ), "Inter-Point constraints are not implemented yet."
                        else:
                            c_val = constraint(q_batch.T.squeeze())
                            assert not c_val.isnan().any()
                            constraint_invalid = c_val < 0
                            assert isinstance(constraint_invalid, Tensor)
                            ok[constraint_invalid] = False
                            constraint_hardness.append(constraint_invalid.sum().item())

                    batch_ok = q_batch[ok]
                    if batch_ok.shape[0] < self.raw_samples // 4:
                        # TODO: here, we reject the entire batch. should just accumulate the samples instead?
                        try_again = True
                        points.clear()
                        logger.info(
                            f"restart {restart + 1}: Sampling ICs for candidate {qi + 1}/{self.q}, trial {i+1}/{N_TRIES}. got {batch_ok.shape[0]} hits. Nonlinear constraint hardnesses: {constraint_hardness}. Retrying...",
                            extra={"uuid": self.uuid},
                        )
                        break

                    acq_computation_batch_size = 512
                    start = 0
                    qi_acq_vals_ = []
                    batch = 0
                    n_batches = int(
                        math.ceil(batch_ok.shape[0] / acq_computation_batch_size)
                    )
                    while start < batch_ok.shape[0]:
                        batch += 1
                        logger.info(
                            f"restart {restart + 1}: Sampling ICs for candidate {qi + 1}/{self.q}, trial {i+1}/{N_TRIES}. Computing acquisition function (batch {batch} / {n_batches}).",
                            extra={"uuid": self.uuid},
                        )
                        if ui_callback:
                            ui_callback(
                                f"restart {restart + 1}: Sampling ICs for candidate {qi + 1}/{self.q}, trial {i+1}/{N_TRIES}. Computing acquisition function (batch {batch} / {n_batches}).",
                            )

                        small_batch = batch_ok[
                            start : start + acq_computation_batch_size
                        ]
                        qi_acq_vals_batch: Tensor = self.acq_function(
                            small_batch.unsqueeze(1)
                        )
                        qi_acq_vals_.append(qi_acq_vals_batch)
                        start += acq_computation_batch_size

                    qi_acq_vals: Tensor = torch.concat(qi_acq_vals_, dim=0)

                    logger.info(
                        f"restart {restart + 1}: Sampling ICs for candidate {qi + 1}/{self.q}, trial {i+1}/{N_TRIES}. Done computing acquisition function.",
                        extra={"uuid": self.uuid},
                    )

                    top = batch_ok[qi_acq_vals.argmax()]
                    points.append(top)

                if i == N_TRIES - 1:
                    raise InfeasibilityError(
                        f"Could not sample a correct point in {N_TRIES} attempts. Check your constraints..."
                    )
                if not try_again:
                    break

        t = torch.stack(points, dim=0)
        logger.info(
            f"restart {restart + 1}: finished candidate generation",
            extra={"uuid": self.uuid},
        )
        if ui_callback:
            ui_callback(f"restart {restart + 1}: finished candidate generation")
        return t.requires_grad_(False)


def proxy_foo(foo, *args, **kwargs):
    ret = foo(*args, **kwargs)
    return ret


def negate_minimize_targets(opt: Optimization, y: Tensor):
    y = y.clone()
    for i, t in enumerate(opt.opt_names):
        try:
            if opt.cfg.target_treatment[t] == "minimize":
                y[:, i] *= -1
        except KeyError:
            continue
    return y


def nonlinear_ic_generator(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: dict[int, float] | None = None,
    options: dict[str, bool | float | int] | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    generator: Callable[[int, int, int | None], Tensor] | None = None,
    fixed_X_fantasies: Tensor | None = None,
    nlics: list[tuple[Callable[[Tensor], Tensor | float], bool]] = [],
    opt: Optimization | None = None,
    model_state_dict: dict[str, Any] | None = None,
    uuid: str | None = None,
    ui_callback: Optional[Callable[[str], None]] = None,
    rng: torch.Generator | None = None,
):
    logger.info("Calling nonlinear_ic_generator", extra={"uuid": uuid})
    if ui_callback:
        ui_callback("Generating initial conditions")

    if rng is None:
        rng = torch.Generator()

    # we need num_restarts * q * d
    # fix equality_constraints so that sample_q_batches_from_polytope will
    # produce samples which match the constraint after features are fixed
    assert (
        inequality_constraints is None
    ), "we don't support inequality_constraints for now"

    if equality_constraints is not None:
        equality_constraints = [
            (a.detach(), b.detach(), c) for a, b, c in equality_constraints
        ]

    old_constraints, equality_constraints = fix_equality_constraints(
        equality_constraints, fixed_features
    )
    check_constraints = old_constraints or equality_constraints

    assert model_state_dict is not None
    assert opt is not None

    gen = RestartCandidateGen(
        bounds=bounds.detach(),
        uuid=uuid,
        raw_samples=raw_samples,
        q=q,
        num_restarts=num_restarts,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        check_constraints=check_constraints,
        fixed_features=fixed_features,
        nlics=nlics,
        acq_function=acq_function,
        opt=opt,
        model_state_dict=model_state_dict,
    )

    base_seed = int(torch.randint(0, 12318237918273918, size=[1], generator=rng).item())

    samples = []
    try:
        for r in range(num_restarts):
            s = gen(r, torch.Generator().manual_seed(base_seed + r), ui_callback)
            samples.append(s)

    except UserInterrupt as e:
        raise e

    with torch.no_grad():
        ret = torch.stack(list(samples), dim=0)

    if fixed_features is not None:
        check_idx = []
        for idx in range(ret.shape[2]):
            if not idx in fixed_features:
                check_idx.append(idx)
        to_check = ret[:, :, check_idx]
        bounds = bounds[:, check_idx]
    else:
        to_check = ret

    # check bounds just to be safe
    if not ((to_check >= bounds[0]) & (to_check <= bounds[1])).all():
        logger.error(
            f"initial candidates generation produced samples out of bounds: {to_check}",
            extra={"uuid": uuid},
        )
        assert (
            False
        ), "Unreachable, initial candidates generation produced samples out of bounds"

    return ret.detach()


def callback_and_check_should_stop(
    args: str,
    abort: ValueProxy[bool],
    uuid: str,
    text_callback: Callable[[str], None] | None,
):
    if text_callback:
        text_callback(args)

    if abort.get():
        logger.info(
            f"User interrupted optimization. Throwing an Exception to cancel.",
            extra={"uuid": uuid},
        )

        raise UserInterrupt()


def get_acqf(opt: Optimization, state_dict: dict[str, Any]) -> AcquisitionFunction:
    model = get_model(opt, state_dict)
    logger.info("got model", extra={"uuid": opt.uuid})
    logger.info("loaded state dict for model", extra={"uuid": opt.uuid})

    if opt.Y_opt.shape[-1] == 1:
        logger.info(
            "Constructing acquisition function: qLogNoisyExpectedImprovement",
            extra={"uuid": opt.uuid},
        )
        acqf = qLogNoisyExpectedImprovement(
            model,
            X_baseline=opt.X,
            cache_root=False,
            prune_baseline=True,
        )
        logger.info(
            "Selected acquisition function: qLogNoisyExpectedImprovement",
            extra={"uuid": opt.uuid},
        )
    else:
        logger.info(
            "Constructing acquisition function: qLogNoisyExpectedHypervolumeImprovement",
            extra={"uuid": opt.uuid},
        )
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model,
            ref_point=torch.zeros(opt.Y_opt.shape[-1:], dtype=torch.float64),
            X_baseline=opt.X,
            prune_baseline=True,
            cache_root=False,
        )
        logger.info(
            "Selected acquisition function: qLogNoisyExpectedHypervolumeImprovement",
            extra={"uuid": opt.uuid},
        )

    return acqf


OptResults = namedtuple("OptResults", ["candidates", "mean", "stddev", "importance"])


def get_next_candidates(
    opt: Optimization,
    state_dict: dict[str, Any],
    text_callback: Callable[[str], None] | None,
    abort_opt: ValueProxy[bool],
) -> OptResults | str:
    """Given an Optimization and trained model, generate next candidates

    Args:
        opt: Optimization
        state_dict: trained model
        callback: user callback for ui logging

    Returns:
        set of candidates from bayesian optimziation and mean and standard deviation of model
        and the respective category scores (if applicable)
        optimization target model prediction or string message in case of error

    """
    model = get_model(opt, state_dict)
    acqf = get_acqf(opt, state_dict)

    gen = torch.Generator(device="cpu").manual_seed(int(time() * 12381238))
    logger.info(f"Optimizing acquisition function...", extra={"uuid": opt.uuid})

    cb = partial(
        callback_and_check_should_stop,
        text_callback=text_callback,
        uuid=opt.uuid,
        abort=abort_opt,
    )
    ic_gen_kwargs = {
        "nlics": opt.nlics,
        "uuid": opt.uuid,
        "ui_callback": cb,
        "rng": gen,
        "opt": opt,
        "model_state_dict": state_dict,
    }

    acqf_kwargs = {
        "options": {
            "maxiter": 256,
        },
        "ic_generator": nonlinear_ic_generator,
        "nonlinear_inequality_constraints": opt.nlics,
        "equality_constraints": opt.eq_constraints,
        "timeout_sec": OPT_TIMEOUT_SEC,
        "raw_samples": 512,
        "sequential": True,  # always use sequential optimization for memory reasons
        **ic_gen_kwargs,
    }
    RESTARTS = 10

    importances = None

    cb("Starting Optimization")

    try:
        if opt.cfg.fix_n.n > 0:
            combinations, _ = select_combinations(
                opt.X,
                opt.Y_opt,
                opt.cfg.num_samples,
                opt.cfg,
                temperature=opt.cfg.fix_n_temperature,
                output_probabilities=False,
            )
            # TODO: currently, this is treated as many different optimization problems
            # with different fixed features across the same underlying model
            #
            # select_combinations uses a heuristic based on expected improvement in
            # the category to select combinations for the fix-n constraint
            # there should be some nicer way to do this, for example using
            # hierarchical models, but that would require moving from botorch to
            # somthing like pyro

            candidate_batches = []
            importances = []

            n_done = 0
            for combination, n, cat_ei in combinations:
                if n == 1:
                    of = f"candidate {n_done + 1}"
                else:
                    of = f"candidates [{n_done + 1}-{n_done + n}]"

                cb(f"Starting generation of {of}/{opt.cfg.num_samples}")
                logger.info(
                    f"Starting generation of {of}/{opt.cfg.num_samples}",
                    extra={"uuid": opt.uuid},
                )

                bounds = opt.bounds.clone().cpu().detach()
                fixed_features = {}
                for i, name in combination:
                    val = opt.cfg.fix_n.values[name]
                    bounds[0, i] = val - 1
                    bounds[1, i] = val + 1
                    fixed_features[i] = val

                if len(candidate_batches) > 0:
                    acqf.set_X_pending(torch.cat(candidate_batches, dim=0))

                c, av = optimize_acqf(
                    acqf,
                    bounds,
                    n,
                    RESTARTS,
                    fixed_features=fixed_features,
                    gen_candidates=partial(
                        wrap_gen_candidates,
                        uuid=opt.uuid,
                        ui_callback=cb,
                        sample_numbers=of,
                    ),
                    **acqf_kwargs,
                )
                candidate_batches.append(c)
                importances.extend(cat_ei for _ in range(c.shape[0]))
                n_done += c.shape[0]

            candidates = torch.cat(candidate_batches, dim=0)
            importances = torch.tensor(importances)

        else:
            candidates, _ = optimize_acqf(
                acqf,
                opt.bounds,
                opt.cfg.num_samples,
                RESTARTS,
                gen_candidates=partial(
                    wrap_gen_candidates, uuid=opt.uuid, ui_callback=cb
                ),
                **acqf_kwargs,
            )

    except InfeasibilityError as e:
        logger.info(f"Error during optimization: {e}.", extra={"uuid": opt.uuid})
        cb(
            f"Could not find a feasible sample point. Constraints may be too restrictive."
        )
        return "Could not find a feasible sample point. Constraints may be too restrictive."
    except ValueError as e:
        logger.warning(f"Error during optimization: {e}.", extra={"uuid": opt.uuid})
        cb(f"Other Error during Optimization: {e}")
        return f"Other Error during Optimization: {e}"
    except KeyboardInterrupt:
        logger.error(
            f"Optimization interrupted by KeyboardInterrupt", extra={"uuid": opt.uuid}
        )
        cb("Optimization Interrupted from Keyboard")
        return "Optimization interrupted by KeyboardInterrupt"
    except UserInterrupt:
        logger.info("Optimization user interrupt successful", extra={"uuid": opt.uuid})
        cb("Optimization Interrupted")
        return "Optimization user interrupt successful"

    logger.info("Done optimizing acquisition function.", extra={"uuid": opt.uuid})
    cb("Done.")

    with torch.no_grad():
        pred = model.posterior(candidates)
    if opt.Y_opt.shape[1] == 1:
        mean = pred.mean
        stddev = pred.stddev
    else:
        mean = pred.mean
        stddev = pred.variance.sqrt()

    if len(mean.shape) == 1:
        # we have a single output variable
        mean = mean.unsqueeze(1)

    if len(stddev.shape) == 1:
        # we have a single output variable
        stddev = stddev.unsqueeze(1)

    mean = negate_minimize_targets(opt, mean)

    return OptResults(candidates, mean, stddev, importances)


def get_model(opt: Optimization, state_dict: dict[str, Any] | None = None) -> Model:
    logger.info(
        f"get_model with inputs of shape {tuple(opt.X.shape)} and targets of shape {tuple(opt.Y_opt.shape)}",
        extra={"uuid": opt.uuid},
    )
    X = opt.X
    Y = opt.Y_opt
    if len(X.shape) == 1:
        X = X.unsqueeze(1)

    model = SingleTaskGP(
        X,
        Y,
        covar_module=MaternKernel(ard_num_dims=X.shape[-1]),
        input_transform=Normalize(d=X.shape[-1]),
        outcome_transform=Standardize(m=Y.shape[-1]),
    )
    # logger.info(f"Selected SingleTaskGP", extra={"uuid": opt.uuid})

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model
