from dataclasses import dataclass
from typing import List
import numpy as np
from numpy.typing import ArrayLike

from .pydiffsol import Ode, MatrixType, ScalarType, SolverMethod, SolverType


ROOT_OPTION_FIELDS = [
    "method",
    "linear_solver",
    "rtol",
    "atol",
]

ODE_OPTION_FIELDS = [
    "max_nonlinear_solver_iterations",
    "max_error_test_failures",
    "update_jacobian_after_steps",
    "update_rhs_jacobian_after_steps",
    "threshold_to_update_jacobian",
    "threshold_to_update_rhs_jacobian",
    "min_timestep",
]

IC_OPTION_FIELDS = [
    "use_linesearch",
    "max_linesearch_iterations",
    "max_newton_iterations",
    "max_linear_solver_setups",
    "step_reduction_factor",
    "armijo_constant",
]


@dataclass
class MultiStepOdeConfig:
    """
    Configuration for multi-step ODE solvers, including the models to use,
    the number of steps, any mappers required for the solver, and optional durations.

    Attributes:
        models (List[str]):
            A list of strings representing the ODE models to be solved.
        steps (List[int]):
            A list of indices of `models` that specify the ODE model to use for
            each step.
        mappers (List[int]):
            A list of indices of `models` that specify the "mapper" model that
            maps the state from the previous step to the next step via the `F`
            tensor.
        duration (List[float | None]):
            A list of floats or None, indicating the max duration for each step.
            If None, the step uses the global final_time or t_eval.
    """
    models: List[str]
    steps: List[int]
    mappers: List[int]
    duration: List[float | None]


class MultiStepOde:
    """
    A multi-step ODE solver that can solve a sequence of ODE models with optional mappers and durations.
    
    This has a similar interface to `Ode` but manages multiple `Ode` instances internally and handles the logic of switching between them during a solve

    Arguments:
        config (MultiStepOdeConfig): The configuration for the multi-step ODE solver.
        matrix_type (MatrixType): The matrix type to use for all ODE solvers.
        scalar_type (ScalarType): The scalar type to use for all ODE solvers.
        method (SolverMethod): The solver method to use for all ODE solvers.
        linear_solver (SolverType): The linear solver to use for all ODE solvers.
    """
    def __init__(
        self,
        config: MultiStepOdeConfig,
        matrix_type: MatrixType = MatrixType.nalgebra_dense,
        scalar_type: ScalarType = ScalarType.f64,
        method: SolverMethod = SolverMethod.bdf,
        linear_solver: SolverType = SolverType.default,
    ):
        # check the validity of the config
        if len(config.steps) == 0:
            raise ValueError("The number of steps must be at least one.")
        if len(config.steps) != len(config.mappers) + 1:
            raise ValueError(
                f"The number of steps must be one more than the number of "
                f"mappers, found {len(config.steps)} steps and "
                f"{len(config.mappers)} mappers."
            )
        if any(
            step < 0 or step >= len(config.models) for step in config.steps
        ):
            raise ValueError(
                f"All step indices must be less than the number of models, "
                f"found step indices {config.steps} and "
                f"{len(config.models)} models."
            )
        if any(
            mapper < 0 or mapper >= len(config.models)
            for mapper in config.mappers
        ):
            raise ValueError(
                f"All mapper indices must be less than the number of models, "
                f"found mapper indices {config.mappers} and "
                f"{len(config.models)} models."
            )
        if len(config.duration) != len(config.steps):
            raise ValueError(
                f"The number of durations must match the number of steps, "
                f"found {len(config.duration)} durations and "
                f"{len(config.steps)} steps."
            )
        if any(
            duration is not None and duration <= 0.0
            for duration in config.duration
        ):
            raise ValueError(
                f"All durations must be positive or None, found durations "
                f"{config.duration}."
            )

        self._config = config
        self._odes = [
            Ode(
                model,
                matrix_type=matrix_type,
                scalar_type=scalar_type,
                method=method,
                linear_solver=linear_solver,
            )
            for model in config.models
        ]
        
        # all odes should have the same nparams
        nparams = self._odes[0].nparams
        for i, ode in enumerate(self._odes[1:], start=1):
            if ode.nparams != nparams:
                raise ValueError(
                    f"All ODE models must have the same number of parameters, "
                    f"found {nparams} for model 0 and {ode.nparams} for model {i}."
                )
        
        # all odes should have the same nout
        nout = self._odes[0].nout
        for i, ode in enumerate(self._odes[1:], start=1):
            if ode.nout != nout:
                raise ValueError(
                    f"All ODE models must have the same number of outputs, "
                    f"found {nout} for model 0 and {ode.nout} for model {i}."
                )
        
        # each step should have either a duration or have a root, both is ok but neither is not ok
        for i, (step_idx, duration) in enumerate(zip(config.steps, config.duration)):
            ode = self._odes[step_idx]
            if duration is None and not ode.has_stop():
                raise ValueError(
                    f"Each step must have either a duration or a root function, "
                    f"found step {i} with no duration and no root function."
                )

    @property
    def matrix_type(self):
        return self._odes[0].matrix_type

    @property
    def method(self):
        return self._odes[0].method

    @method.setter
    def method(self, value):
        for ode in self._odes:
            ode.method = value

    @property
    def linear_solver(self):
        return self._odes[0].linear_solver

    @linear_solver.setter
    def linear_solver(self, value):
        for ode in self._odes:
            ode.linear_solver = value

    @property
    def rtol(self):
        return self._odes[0].rtol

    @rtol.setter
    def rtol(self, value):
        for ode in self._odes:
            ode.rtol = value

    @property
    def atol(self):
        return self._odes[0].atol

    @atol.setter
    def atol(self, value):
        for ode in self._odes:
            ode.atol = value

    @property
    def ic_options(self):
        return self._odes[0].ic_options

    @property
    def options(self):
        return self._odes[0].options
    
    def _t_eval_for_step(self, current_time: float, step_idx: int, t_eval: np.ndarray):
        """
        Get the t_eval for the current step, which is the subset of t_eval that falls within the duration of the current step,
        including the current time and end time of the step if they are not already included.
        If the duration of the current step is None, then return all t_eval that are greater than or equal to the current time.
        """
        duration = self._config.duration[step_idx]
        if duration is None:
            step_t_evals = t_eval[t_eval >= current_time]
        else:
            step_end_time = current_time + duration
            step_t_evals = t_eval[(t_eval >= current_time) & (t_eval <= step_end_time)]
            # add step_end_time if it's not already included and is greater than current_time and is less than last t_eval
            if step_end_time > current_time and step_end_time not in step_t_evals and step_end_time <= t_eval[-1]:
                step_t_evals = np.append(step_t_evals, step_end_time)
        # add current time if it's not already included
        if current_time not in step_t_evals:
            step_t_evals = np.append(step_t_evals, current_time)
            step_t_evals.sort()
        return step_t_evals
    
    def _step_final_time(self, step_idx: int, final_time: float):
        step_duration = self._config.duration[step_idx]
        return final_time if step_duration is None else min(step_duration, final_time)

    def solve(self, params: ArrayLike, final_time: float, solution=None):
        self._sync_shared_config()
        params = np.asarray(params, dtype=np.float64)
        final_time = np.float64(final_time)

        # Step 0
        step_final_time = self._step_final_time(0, final_time)
        solution = self._odes[self._config.steps[0]].solve(
            params, step_final_time, solution
        )

        for i, (mapper_idx, step_idx) in enumerate(
            zip(self._config.mappers, self._config.steps[1:])
        ):
            current_t = solution.ts[-1]
            if current_t >= final_time:
                break
            solution = self._map_solution(params, mapper_idx, solution)
            step_final_time = self._step_final_time(i + 1, final_time)
            solution = self._odes[step_idx].solve(
                params, step_final_time, solution
            )
        return solution

    def solve_dense(self, params: ArrayLike, t_eval: ArrayLike, solution=None):
        self._sync_shared_config()
        params = np.asarray(params, dtype=np.float64)
        t_eval = np.asarray(t_eval, dtype=np.float64)

        # Step 0
        current_t = 0.0
        step_t_eval = self._t_eval_for_step(current_t, 0, t_eval)
        solution = self._odes[self._config.steps[0]].solve_dense(
            params, step_t_eval, solution
        )
        for i, (mapper_idx, step_idx) in enumerate(
            zip(self._config.mappers, self._config.steps[1:])
        ):
            current_t = solution.ts[-1]
            if current_t >= t_eval[-1]:
                break
            solution = self._map_solution(params, mapper_idx, solution)
            step_t_eval = self._t_eval_for_step(current_t, i + 1, t_eval)
            solution = self._odes[step_idx].solve_dense(
                params, step_t_eval, solution
            )
        return solution

    def solve_fwd_sens(
        self, params: ArrayLike, t_eval: ArrayLike, solution=None
    ):
        self._sync_shared_config()
        params = np.asarray(params, dtype=np.float64)
        t_eval = np.asarray(t_eval, dtype=np.float64)
        
        current_t = 0.0
        step_t_eval = self._t_eval_for_step(current_t, 0, t_eval)
        solution = self._odes[self._config.steps[0]].solve_fwd_sens(
            params, step_t_eval, solution
        )

        for i, (mapper_idx, step_idx) in enumerate(
            zip(self._config.mappers, self._config.steps[1:])
        ):
            current_t = solution.ts[-1]
            solution = self._map_solution(params, mapper_idx, solution)
            step_t_eval = self._t_eval_for_step(current_t, i + 1, t_eval)
            solution = self._odes[step_idx].solve_fwd_sens(
                params, step_t_eval, solution
            )
        return solution

    def _sync_shared_config(self):
        src = self._odes[0]
        src_options = src.options
        src_ic_options = src.ic_options
        for ode in self._odes[1:]:
            # Sync root-level fields
            for field in ROOT_OPTION_FIELDS:
                setattr(ode, field, getattr(src, field))
            # Sync options fields
            ode_options = ode.options
            ode_ic_options = ode.ic_options
            for field in ODE_OPTION_FIELDS:
                setattr(ode_options, field, getattr(src_options, field))
            for field in IC_OPTION_FIELDS:
                setattr(ode_ic_options, field, getattr(src_ic_options, field))

    def _map_solution(
        self, params: np.ndarray, mapper_idx: int, source_solution
    ):
        mapper = self._odes[mapper_idx]
        t_last = source_solution.ts[-1]
        y_last = source_solution.current_state
        mapped = mapper.rhs(params, t_last, y_last)
        source_solution.current_state = mapped
        return source_solution
