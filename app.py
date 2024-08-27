import asyncio
import enum
import math
import multiprocessing
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from functools import partial
from io import StringIO
from multiprocessing.context import BaseContext
from multiprocessing.managers import ListProxy, ValueProxy
from typing import Optional, get_args
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from nicegui import ui
from nicegui.elements.tabs import Tab
from nicegui.events import UploadEventArguments, ValueChangeEventArguments
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from expr_parser import parse_nonlinear_constraint
from mylog import get_logger
from opt import (
    FixNConstraint,
    Optimization,
    OptimizeConfig,
    RawConstraint,
    TargetTreatment,
    UserInterrupt,
    fit_model,
    get_model,
    get_next_candidates,
    negate_minimize_targets,
    parse_constraint,
    select_combinations,
)
from util import get_labels

SID_COL_NAME = "sample id"
COMBINATION_IMPORTANCE_NAME = "%%combination importance"
FAVICON_PATH = pathlib.Path("./assets/favicon.png").absolute()
EXAMPLE_PATH = pathlib.Path("./assets/example.zip").absolute()
logger = get_logger(__file__)

APP_VERSION = os.getenv("GPBO_WEBAPP_VERSION") or "development-version"


@dataclass
class UploadData:
    targets: Optional[pd.DataFrame] = None
    inputs: Optional[pd.DataFrame] = None

    def reset(self):
        self.inputs = None
        self.targets = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UploadData):
            return False

        def compare_opt_dfs(
            a: Optional[pd.DataFrame], b: Optional[pd.DataFrame]
        ) -> bool:
            if a is None:
                if b is not None:
                    return False
            else:
                if b is None:
                    return False
                else:
                    if (
                        not (a.values == b.values).all()
                        and (a.index == b.index).all()
                        and (a.columns == b.columns).all()
                    ):
                        return False
            return True

        if not compare_opt_dfs(self.inputs, other.inputs):
            return False
        return compare_opt_dfs(self.targets, other.targets)


class AppStep(enum.Enum):
    START = 0
    CFG = 1
    OPT = 2


def validate_csv(e: UploadEventArguments) -> pd.DataFrame | str:
    content = e.content.read().decode("utf-8")
    try:
        df = pd.read_csv(StringIO(content))
    except pd.errors.ParserError as err:
        return f"Could not parse file '{e.name}' as csv: '{err}'"
    if SID_COL_NAME not in df.columns:
        return f"No Sample ID column found in '{e.name}'. Please add a column '{SID_COL_NAME}' with the sample IDs"

    if not df[SID_COL_NAME].is_unique:
        duplicates = ", ".join(df[SID_COL_NAME][df[SID_COL_NAME].duplicated()])
        return f"Duplicate sample IDs in {e.name}: {duplicates}"

    for c in filter(lambda x: x != SID_COL_NAME, df.columns):
        if not pd.api.types.is_numeric_dtype(df[c]):
            # TODO: say which
            return f"Column '{c}' contains non-numeric values. We have no support for categorical values at the moment."

    return df


def add_opt_status(
    stats: ListProxy,
    candidate_id: ValueProxy[int],
    cur_candidate: ValueProxy[str],
    opt_start: ValueProxy[str],
    msg: str,
):
    now = time.localtime()
    t = time.strftime("%H:%M:%S", now)
    if "Generating initial conditions" in msg:
        n = candidate_id.get()
        candidate_id.set(n + 1)
        cur_candidate.set(f"{t} | Currently Optimizing Candidate {n + 1}")
        stats.append(f"{t} | {msg}")
    if "Starting Optimization" in msg:
        opt_start.set(f"Optimization Started at: {t}")
    stats.append(f"{t} | {msg}")


@dataclass
class CfgUi:
    bounds_form: ui.aggrid
    opt_dir: ui.aggrid
    opt_fixed: ui.aggrid
    n_fixed: ui.number


@dataclass
class ModelFit:
    Y_true: NDArray

    Y_means: NDArray
    Y_stddev: NDArray
    opt_names: list[str]


@dataclass
class App:
    uuid: str
    upload_data: UploadData
    valid_upload: bool = False
    cfg: OptimizeConfig | None = None
    next: Optional[pd.DataFrame] = None
    opt_running: bool = False
    opt_preparing: bool = False
    opt_status: ListProxy = None
    cancel_opt: ValueProxy[bool] = None
    cur_opt_candidate: ValueProxy[str] = None
    candidate_no: ValueProxy[int] = None
    opt_start: ValueProxy[str] = None
    model_fit: Optional[ModelFit] = None
    ctx: BaseContext = None

    plot_columns: tuple[str, str] = None

    candidate_ncols: int = 12

    cfg_ui: CfgUi | None = None

    step: AppStep = AppStep.START

    def __post_init__(self):
        logger.info(f"New client connected", extra={"uuid": self.uuid})

    def init_cfg(self):
        logger.info("Initializing Config", extra={"uuid": self.uuid})
        assert self.upload_data.inputs is not None
        assert self.upload_data.targets is not None
        self.cfg = OptimizeConfig(
            self.upload_data.inputs.columns.tolist(),
            self.upload_data.targets.columns.tolist(),
            [],
            bounds=self.upload_data.inputs.agg(["min", "max"]).T,
            num_samples=5,
        )

    def validate_upload(self):
        if self.upload_data.inputs is None or self.upload_data.targets is None:
            self.valid_upload = False
            logger.info(
                "Upload validation failed. Incomplete uploads.",
                extra={"uuid": self.uuid},
            )
            return

        for elem in self.upload_data.inputs.columns:
            if elem == COMBINATION_IMPORTANCE_NAME:
                self.valid_upload = False
                logger.warning(
                    f"Found reserved name {COMBINATION_IMPORTANCE_NAME} in input columns.",
                    extra={"uuid": self.uuid},
                )
                ui.notify(
                    f"Reserved name '{COMBINATION_IMPORTANCE_NAME}' found in input columns. Please choose another name.",
                    type="negative",
                )
                return

        merged = pd.merge(
            self.upload_data.inputs, self.upload_data.targets, how="outer"
        )
        no_matches = ", ".join(
            merged.reset_index()[merged.isna().any(axis=1)][SID_COL_NAME]
        )
        if len(no_matches) > 0:
            logger.warning(
                "Upload validation failed. Incomplete uploads.",
                extra={"uuid": self.uuid},
            )
            ui.notify(
                f"Missing values in either inputs or targets for the following samples: {no_matches}. These samples will not be used for optimization.",
                type="warning",
            )
        index = merged.dropna()[SID_COL_NAME]

        self.upload_data.inputs = self.upload_data.inputs.set_index(SID_COL_NAME).loc[
            index
        ]
        self.upload_data.targets = self.upload_data.targets.set_index(SID_COL_NAME).loc[
            index
        ]

        self.valid_upload = True
        if self.valid_upload:
            self.show_upload_plot.refresh()
        else:
            self.plot_columns = None
        logger.info("Upload validation succeeded.", extra={"uuid": self.uuid})

    def goto_cfg(self):
        logger.info("Open Config Tab", extra={"uuid": self.uuid})
        self.step = AppStep.CFG
        self.set_cancel_opt()
        if self.cfg is None:
            self.init_cfg()
        self.cfg_page.refresh()

    def goto_opt(self):
        self.ctx = multiprocessing.get_context("spawn")
        self.manager = self.ctx.Manager()
        self.opt_status = self.manager.list([])
        self.cancel_opt = self.manager.Value("b", False)
        self.cur_opt_candidate = self.manager.Value("c", "")
        self.candidate_no = self.manager.Value("i", 0)
        self.opt_start = self.manager.Value("c", "")

        logger.info("Open Optimization Tab", extra={"uuid": self.uuid})

        self.next = None
        self.step = AppStep.OPT

        self.opt_page.refresh()

    def add_constraint(self, c: str):
        logger.info("Adding Constraint...", extra={"uuid": self.uuid})
        if self.cfg is None:
            self.init_cfg()
        assert self.cfg is not None
        assert self.valid_upload
        assert self.upload_data.inputs is not None

        if "=" in c:
            constraint = parse_constraint(c, self.upload_data.inputs.columns.tolist())  # type: ignore
            if isinstance(constraint, str):
                logger.warning(
                    f"Could not parse constraint {c}: {constraint}.",
                    extra={"uuid": self.uuid},
                )
                ui.notify(f"Could not parse constraint: {constraint}", type="negative")
                return
            constraint, normalized = constraint

        else:
            constraint = parse_nonlinear_constraint(
                c, self.upload_data.inputs.columns.tolist()
            )
            if isinstance(constraint, str):
                logger.warning(
                    f"Could not parse constraint {c}: {constraint}.",
                    extra={"uuid": self.uuid},
                )
                ui.notify(f"Could not parse constraint: {constraint}", type="negative")
                return
            normalized = constraint.normalized_repr()

        raw_constraint = RawConstraint(normalized, False)

        if constraint in self.cfg.parsed_constraints:
            logger.warning(
                f"Could not add constraint because it already exists: {constraint}.",
                extra={"uuid": self.uuid},
            )
            ui.notify("Constraint already exists.", type="negative")
            return

        self.cfg.parsed_constraints.append(constraint)
        self.cfg.constraints_raw.append(raw_constraint)

        logger.info(
            f"Successfully added constraint {constraint}", extra={"uuid": self.uuid}
        )

    def validate_new_cfg_against_uploads(self, cfg: OptimizeConfig) -> Optional[str]:
        logger.info("Validating uploaded config", extra={"uuid": self.uuid})
        logger.info(f"Config: {cfg.to_json()}", extra={"uuid": self.uuid})  # type: ignore
        assert self.upload_data is not None

        def get_wrong_names(
            check: list[str], true_vals: list[str]
        ) -> tuple[list[str], list[str]]:
            too_many = []
            for elem in check:
                if elem not in true_vals:
                    too_many.append(elem)

            missing = []
            for elem in true_vals:
                if elem not in check:
                    missing.append(elem)

            return too_many, missing

        assert self.upload_data.inputs is not None
        assert self.upload_data.targets is not None
        for n in cfg.input_names:
            if n == COMBINATION_IMPORTANCE_NAME:
                logger.warning(
                    f"Found reserved name {COMBINATION_IMPORTANCE_NAME} in input columns.",
                    extra={"uuid": self.uuid},
                )
                return f"Reserved name '{COMBINATION_IMPORTANCE_NAME}' found in input columns. Please choose another name."

        tm, ms = get_wrong_names(cfg.input_names, self.upload_data.inputs.columns)  # type: ignore
        if len(tm) > 0:
            msg = f"Found unknown input column(s) in the config: [{', '.join(tm)}]"
            logger.warning(msg, extra={"uuid": self.uuid})
            return msg
        if len(ms) > 0:
            msg = f"Config is missing input column name(s): [{', '.join(ms)}]"
            logger.warning(msg, extra={"uuid": self.uuid})
            return msg

        tm, ms = get_wrong_names(cfg.target_names, self.upload_data.targets.columns)  # type: ignore
        if len(tm) > 0:
            msg = f"Found unknown target column(s) in the config: [{', '.join(tm)}]"
            logger.warning(msg, extra={"uuid": self.uuid})
            return msg
        if len(ms) > 0:
            msg = f"Config is missing target column name(s): [{', '.join(ms)}]"
            logger.warning(msg, extra={"uuid": self.uuid})
            return msg

        tm, ms = get_wrong_names(list(cfg.target_treatment.keys()), self.upload_data.targets.columns)  # type: ignore
        if len(tm) > 0:
            msg = f"Found unknown target column(s) in the config's target treatments: [{', '.join(tm)}]"
            logger.warning(msg, extra={"uuid": self.uuid})
            return msg
        if len(ms) > 0:
            msg = f"Config is missing target_treatment columns: [{', '.join(ms)}]"
            logger.warning(msg, extra={"uuid": self.uuid})
            return msg

        tm, ms = get_wrong_names(cfg.bounds.index, self.upload_data.inputs.columns)  # type: ignore
        if len(tm) > 0:
            msg = f"Found unknown target(s) in the config's bounds: [{', '.join(tm)}]"
            logger.warning(msg, extra={"uuid": self.uuid})
            return msg
        if len(ms) > 0:
            msg = f"Config is missing target bound(s): [{', '.join(ms)}]"
            logger.warning(msg, extra={"uuid": self.uuid})
            return msg

        if not "max" in cfg.bounds.columns:
            msg = f"Config is missing max bound"
            logger.warning(msg, extra={"uuid": self.uuid})
            return msg
        if not "min" in cfg.bounds.columns:
            msg = f"Config is missing min bound"
            logger.warning(msg, extra={"uuid": self.uuid})
            return msg

        cfg.parsed_constraints = []
        for rc in cfg.constraints_raw:
            if rc.nonlinear:
                c = parse_nonlinear_constraint(
                    rc.normalized_constraint, cfg.input_names
                )
                if isinstance(c, str):
                    msg = f"Invalid nonlinear inequality constraint in config: {c}"
                    logger.warning(msg, extra={"uuid": self.uuid})
                    return msg
            else:
                c = parse_constraint(rc.normalized_constraint, cfg.input_names)
                if isinstance(c, str):
                    msg = f"Invalid linear equality constraint in config: {c}"
                    logger.warning(msg, extra={"uuid": self.uuid})
                    return msg
                c, _ = c
            cfg.parsed_constraints.append(c)

        logger.info("Successfully validated uploaded config", extra={"uuid": self.uuid})
        return None

    def handle_target_upload(self, e: UploadEventArguments):
        target = validate_csv(e)

        if isinstance(target, str):
            logger.warning(
                f"Could not load target csv: {target}", extra={"uuid": self.uuid}
            )
            ui.notify(f"Could not load input csv: {target}", type="negative")
        else:
            self.upload_data.targets = target
            logger.info("Successfully loaded targets", extra={"uuid": self.uuid})
            ui.notify("Uploaded targets", type="positive")

        self.validate_upload()

    def handle_input_upload(self, e: UploadEventArguments):
        input = validate_csv(e)
        if isinstance(input, str):
            logger.warning(
                f"Could not load input csv: {input}", extra={"uuid": self.uuid}
            )
            ui.notify(f"Could not load input csv: {input}", type="negative")
        else:
            self.upload_data.inputs = input
            logger.info("Successfully loaded inputs", extra={"uuid": self.uuid})
            ui.notify("Uploaded inputs", type="positive")
        self.validate_upload()

    def remove_constraint(self, i: int):
        assert self.cfg is not None
        try:
            removed_constraint = self.cfg.remove_constraint(i, self.uuid)
        except IndexError:
            logger.error(
                f"Trying to remove non-existing constraint {i}. Encountered IndexError.",
                extra={"uuid": self.uuid},
            )
            ui.notify(
                "Could not remove constraint. Does not exist. This should never happen."
            )
            return

        logger.info(
            f"Removed constraint {i} ({removed_constraint})",
            extra={"uuid": self.uuid},
        )
        self.constraint_list.refresh()

    @ui.refreshable_method
    def constraint_list(self):
        assert self.cfg is not None
        for i, elem in enumerate(self.cfg.constraints_raw):
            with ui.row(wrap=False):
                ui.button(
                    "X",
                    color="red",
                    on_click=lambda _, idx=i: (self.remove_constraint(idx)),
                ).tooltip("Delete Constraint")
                ui.label(elem.normalized_constraint)

    @ui.refreshable_method
    def cfg_page(self, panels):
        ui.markdown("""#### Optimization Configuration """)

        if self.cfg is None:
            # do nothing here. this can only happen if we haven't had any uploads yet
            return

        def handle_cfg_upload(e: UploadEventArguments):
            content = e.content.read().decode("utf-8")
            try:
                cfg = OptimizeConfig.from_json(content)  # type: ignore
            except ValueError:
                logger.error(
                    "Could not load configuration file: invalid format",
                    extra={"uuid": self.uuid},
                )
                ui.notify(
                    "Could not load configuration file: invalid format", type="negative"
                )
                return
            except KeyError as error:
                logger.error(
                    f"Could not load configuration file. Missing key: {error}",
                    extra={"uuid": self.uuid},
                )
                ui.notify(
                    f"Could not load configuration file. Missing Key: {error}",
                    type="negative",
                )
                return

            logger.info(
                f"user uploaded config {cfg.to_json()}", extra={"uuid": self.uuid}
            )
            msg = self.validate_new_cfg_against_uploads(cfg)
            if msg is not None:
                logger.info(
                    f"Could not load configuration file, because it does not match uploaded data: {msg}",
                    extra={"uuid": self.uuid},
                )
                ui.notify(
                    f"Could not load configuration file, because it does not match uploaded data: {msg}",
                    type="negative",
                )
                return

            logger.info("Successfully set new config", extra={"uuid": self.uuid})
            ui.notify("Successfully uploaded config.", type="positive")
            self.cfg = cfg
            self.cfg_page.refresh()

        async def transition_opt():
            self.opt_preparing = True
            self.model_fit = None

            await asyncio.sleep(0.1)

            await self.update_cfg()
            assert self.cfg is not None
            logger.info(
                f"Moving to optimization tab with cfg: {self.cfg.to_json()}",
                extra={"uuid": self.uuid},
            )

            panels.set_value("Optimize")

        async def download_cfg():
            await self.update_cfg()
            assert self.cfg is not None
            cfg = self.cfg.to_json()  # type: ignore
            logger.info(f"User downloading config {cfg}.", extra={"uuid": self.uuid})
            ui.download(
                cfg.encode("utf-8"), filename="opt_cfg.json", media_type="text/json"
            )

        ui.markdown(
            "First, select the number of samples you want to acquire. Large numbers of samples may consume too much memory, especially for multi-target optimization and therefore the process may fail. For single-target optimization, 10 samples works fine, and 5 is a good number for two-target optimization. If you need more samples, the process will take longer, but 80 samples may take around 35 minutes."
        )

        with ui.row(wrap=False).classes("w-full"):
            ui.number("Number of Samples", min=1, value=4, precision=0).bind_value(
                self.cfg,
                "num_samples",
                forward=int,
            ).bind_enabled_from(self, "opt_running", backward=lambda x: not x)
            ui.button("Optimize", on_click=transition_opt, color="green")
            ui.button("Dowload Config", on_click=download_cfg)
            ui.upload(
                label="Load Saved Optimization Config",
                on_upload=handle_cfg_upload,
                auto_upload=True,
            ).props("accept=.json")

        with ui.row(align_items="stretch", wrap=False).classes("w-full"):
            ui.markdown("#### Parameter Ranges").classes("w-1/2")
            ui.markdown("#### Maximize/Minimize").classes("w-1/2")
        with ui.row(align_items="stretch", wrap=False).classes("w-full"):
            with ui.column().classes("w-1/2"):
                ui.markdown("""Set a minimum and maximum for all input parameters.""")
            with ui.column().classes("w-1/2"):
                ui.markdown("""Configure the optimization direction of the outputs.
A check-mark indicates, that the quantity will be maximized, no check-mark indicates minimization.
All targets that are present here are optimized at the same time.""")
        with ui.row(align_items="stretch", wrap=False).classes("w-full"):
            with ui.column().classes("w-1/2"):
                bounds_form = ui.aggrid.from_pandas(
                    self.cfg.bounds.reset_index(names="bound"),
                    options={
                        "columnDefs": [
                            {"field": "bound", "editable": False},
                            {"field": "min", "editable": True},
                            {"field": "max", "editable": True},
                        ],
                        "suppressMoveWhenColumnDragging": True,
                        "stopEditingWhenCellsLoseFocus": True,
                    },
                ).classes("h-stretch")
            with ui.column().classes("w-1/2"):
                df = pd.DataFrame.from_records(
                    [
                        {
                            "Output": c,
                            "Treatment": self.cfg.target_treatment.get(c, "maximize"),
                        }
                        for c in self.cfg.target_names
                    ],
                )

                df["Treatment"] = df["Treatment"].astype("category")
                opt_dir = ui.aggrid.from_pandas(
                    df,
                    options={
                        "columnDefs": [
                            {"field": "Output", "editable": False},
                            {
                                "field": "Treatment",
                                "cellEditor": "agSelectCellEditor",
                                "editable": True,
                                "cellEditorParams": {
                                    "values": get_args(TargetTreatment)
                                },
                            },
                        ],
                        "suppressMoveWhenColumnDragging": True,
                        "stopEditingWhenCellsLoseFocus": True,
                    },
                )

        with ui.row(align_items="stretch", wrap=False).classes("w-full"):
            ui.markdown("##### Input Constraints").classes("w-1/2")
            ui.markdown("##### Fixing N of K features").classes("w-1/2")

        with ui.row(align_items="stretch", wrap=False).classes("w-full"):
            with ui.column().classes("w-1/2"):
                ui.markdown(
                    """Equality constraints may be configured using a list of factors and column names in the inputs. The weighted sum of these columns is constrained to equal the output on the right hand side of the equality.
```
1.0 * v0, 4.0 * v2, -1.0 * v3 = 0.0
```
If you want constrained columns to contain spaces, you may enclose them in double quotes: like so `1.0 * "Value 1"`.
You may omit factors in front of column names. The factor will default to 1.0.

Inequality constraints may be specified using an inequality. For the inputs Fe, Cr and T:
```
(Fe * ln(Fe) + Cr * ln(Cr)) * T^2 < 10.0
```
Currently, only larger than and smaller than ('>' and '<') are supported.
Supported functions are 'ln' (natural logarithm), 'log' (base 10 logarithm), and 'exp' (exponential).
"""
                )
                self.constraint_list()
                constraint = ui.input()
                ui.button(
                    "Add Constraint",
                    on_click=lambda: (
                        self.add_constraint(constraint.value),
                        self.constraint_list.refresh(),
                    ),
                )
            with ui.column().classes("w-1/2"):
                ui.markdown(
                    """It is also possible to specify a set of K features of which N can be randomly
set to a specific value during an experiment.
For each maybe-fixed feature, it is possible to set the fixed value separately.
If a value should not be set, just leave it's entry in the table below blank.
This may be helpful if you are trying to optimize material compositions and want to
explore K materials in total, but only N at a time, and the other compositions should
be set to zero.

This constraint turns the optimization into a two-step process. First, the combination of fixed variables is selected, and then each combination is optimized.
To guide how strongly each combination's expected improvement influences it's importance,
we introduce the temperature parameter below.
Setting it to 0 will disregard how good each combination of inputs free inputs is,
and setting it to a high value (such as 1000) will cause it to strongly prefer only
the best combination of free variables.
"""
                )

                records = []
                names = set()
                for k, v in self.cfg.fix_n.values.items():
                    names.add(k)
                    records.append({"Bound": k, "Value": v})
                records.extend(
                    [
                        {"Bound": name, "Value": None}
                        for name in self.cfg.input_names
                        if name not in names
                    ]
                )
                df = pd.DataFrame.from_records(records)
                opt_fixed = ui.aggrid.from_pandas(
                    df,
                    options={
                        "columnDefs": [
                            {"field": "Bound", "editable": False},
                            {"field": "Value", "editable": True},
                        ],
                        "suppressMoveWhenColumnDragging": True,
                        "stopEditingWhenCellsLoseFocus": True,
                    },
                )

                with ui.row(align_items="stretch", wrap=False).classes("w-full"):
                    async def validate_n_fixed(v: ValueChangeEventArguments):
                        if v.value is None:
                            n_fixed.set_value(0)
                            return

                        fixed_data = await opt_fixed.get_client_data()
                        fixed = []
                        for t in fixed_data:
                            if t["Value"] is not None and t["Value"] != "":
                                try:
                                    fixed[t["Bound"]] = float(t["Value"])
                                except ValueError as e:
                                    logger.error(f"Value for {t['Bound']} must be a number. Got {t['Value']}")

                        if v.value > len(fixed):
                            logger.warning(
                                "User specified n_fixed > len(fixed). Clamping...",
                                extra={"uuid": self.uuid},
                            )
                            ui.notify(
                                "You specified more values to be fixed at a time than fixation values. Clamping to number of fixed values.",
                                type="warning",
                            )
                            n_fixed.set_value(len(fixed))

                        assert self.cfg is not None
                        logger.info("setting cfg.fix_n", extra={"uuid": self.uuid})
                        self.cfg.fix_n = FixNConstraint(
                            {k: float(v) for k, v in fixed}, int(v.value)
                        )
                        show_probabilities()

                    ui.label("Number of inputs fixed at a time:").classes("w-2/3")
                    n_fixed = ui.number(
                        min=0,
                        max=len(self.cfg.input_names),
                        precision=0,
                        value=self.cfg.fix_n.n if self.cfg.fix_n.n != -1 else 0,
                        on_change=validate_n_fixed,
                    ).classes("w-1/3")

                with ui.row(align_items="stretch", wrap=False).classes("w-full"):
                    probabilities_bar = ui.plotly({"layout": {"width": "w-full"}})
                    probabilities_bar.bind_visibility_from(
                        n_fixed, "value", lambda x: (x or 0) > 0
                    )

                    def show_probabilities():
                        assert self.upload_data.inputs is not None
                        assert self.upload_data.targets is not None
                        assert self.cfg is not None

                        X = torch.tensor(self.upload_data.inputs.values).double()
                        Y = torch.tensor(self.upload_data.targets.values).double()
                        if self.cfg.fix_n.n <= 0:
                            return
                        try:
                            _, probabilities = select_combinations(
                                X,
                                Y,
                                self.cfg.num_samples,
                                self.cfg,
                                temperature=self.cfg.fix_n_temperature,
                            )
                        except ValueError as e:
                            ui.notify(str(e), type="negative")
                            return
                        top_n = np.arange(probabilities.shape[0]) + 1
                        df = pd.DataFrame(
                            {
                                "cumulative probability / %": np.cumsum(
                                    sorted((probabilities * 100).tolist(), reverse=True)
                                ),
                                "Top N": top_n,
                            }
                        )
                        df["quartile"] = (df["cumulative probability / %"] / 25).astype(
                            int
                        ) + 1
                        df["quartile"] = "Q" + np.minimum(df["quartile"], 4).astype(str)
                        fig = px.bar(
                            df,
                            x="Top N",
                            y="cumulative probability / %",
                            orientation="v",
                            color="quartile",
                            title="Cumulative probability for top N combinations",
                        )
                        fig.update_layout(bargap=0.2)
                        probabilities_bar.update_figure(fig)

                    show_probabilities()

                    async def update_example_samples(v: ValueChangeEventArguments):
                        assert self.cfg is not None

                        if v.value is None:
                            return

                        try:
                            self.cfg.fix_n_temperature = float(v.value)
                        except ValueError:
                            return

                        show_probabilities()

                r = ui.row(align_items="stretch", wrap=False).classes("w-full")
                r.bind_visibility_from(n_fixed, "value", lambda x: (x or 0) > 0)
                with r:
                    try:
                        max_temp = (
                            math.comb(len(self.cfg.fix_n.values), self.cfg.fix_n.n) * 10
                        )
                    except ValueError:
                        max_temp = 100

                    ui.label("Temperature").classes("w-2/3")
                    self.temperature = ui.number(
                        min=0,
                        max=max_temp,
                        value=min(max_temp, 25),
                        on_change=update_example_samples,
                    ).classes("w-1/3")
                ui.markdown(
                    "The above plot visualizes the cumulative probabilities for combinations of fixed inputs sorted by importance. If you set the temperature to 0, you can see that the cumulative probability is a line, and all categories are equally important. If you choose a high value, categories which are more promising (have higher expected improvement) will be selected more, and less promising categories will be ignored."
                ).bind_visibility_from(n_fixed, "value", lambda x: (x or 0) > 0)

                self.cfg_ui = CfgUi(bounds_form, opt_dir, opt_fixed, n_fixed)

        with ui.card().bind_visibility_from(self, "opt_preparing").classes(
            "fixed z-20 m-auto inset-x-0 inset-y-0 justify-center w-fit h-fit bg-orange"
        ):
            with ui.row().classes("content-center"):
                ui.spinner("dots", size="2em")
                ui.label("Preparing Optimization...").classes("text-lg")

    async def update_cfg(self):
        if self.cfg is None:
            logger.error(
                "this should never be called in this app state",
                extra={"uuid": self.uuid},
            )
            return
        if self.cfg_ui is None:
            logger.error(
                "this should never be called in this app state",
                extra={"uuid": self.uuid},
            )
            return

        cfg_ui = self.cfg_ui

        logger.info("Updating config from user input...", extra={"uuid": self.uuid})
        bounds_data = await cfg_ui.bounds_form.get_client_data()
        self.cfg.bounds = pd.DataFrame(bounds_data).set_index("bound")

        opt_data = await cfg_ui.opt_dir.get_client_data()
        self.cfg.target_treatment = {t["Output"]: t["Treatment"] for t in opt_data}  # type: ignore

        fixed_data = await cfg_ui.opt_fixed.get_client_data()
        fixed: dict[str, float] = {}
        for t in fixed_data:
            if t["Value"] is not None and t["Value"] != "":
                try:
                    fixed[t["Bound"]] = float(t["Value"])
                except ValueError as e:
                    logger.warning(f"Erorr in fix-n bound specification. Value for {t['Bound']} must be a number. Got {t['Value']}")
                    ui.notify(f"Fix-N Bound: Value for {t['Bound']} must be a number. Got {t['Value']}", type="negative")

        if cfg_ui.n_fixed.value > len(fixed):
            logger.warning("User specified more inputs than ok")

        self.cfg.fix_n = FixNConstraint(fixed, int(cfg_ui.n_fixed.value))

        logger.info("Done updating config from user input.", extra={"uuid": self.uuid})

    @ui.refreshable_method
    def show_model_fit(self):
        if self.model_fit is None:
            return

        if len(self.model_fit.Y_means.shape) == 1:
            self.model_fit.Y_means = self.model_fit.Y_means[:, None]
        if len(self.model_fit.Y_true.shape) == 1:
            self.model_fit.Y_true = self.model_fit.Y_true[:, None]
        if len(self.model_fit.Y_stddev.shape) == 1:
            self.model_fit.Y_stddev = self.model_fit.Y_stddev[:, None]

        # TODO: Label points with sample ID
        num_plots = self.model_fit.Y_means.shape[1]
        columns = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / columns))

        labels = self.upload_data.inputs.index.values
        fig = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=[f"fit of {t}" for t in self.model_fit.opt_names],
        )
        for i in range(rows * columns):
            if i >= num_plots:
                continue

            t = self.model_fit.opt_names[i]
            x = self.model_fit.Y_true[:, i]

            row = i // columns + 1
            col = i % columns + 1
            dx = x.max() - x.min()
            fig.add_scatter(
                x=[x.min() - 0.1 * dx, x.max() + 0.1 * dx],
                y=[x.min() - 0.1 * dx, x.max() + 0.1 * dx],
                mode="lines",
                line=dict(color="black", width=3),
                showlegend=False,
                row=row,
                col=col,
            )
            fig.add_scatter(
                x=self.model_fit.Y_true[:, i],
                y=self.model_fit.Y_means[:, i],
                error_y=dict(
                    type="data", array=self.model_fit.Y_stddev[:, i], visible=True
                ),
                hovertext=labels,
                mode="markers",
                row=row,
                col=col,
                showlegend=False,
            )
            fig.update_xaxes(title=f"actual {t}", row=row, col=col)
            fig.update_yaxes(title=f"predicted {t}", row=row, col=col)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        ui.plotly(fig).classes("w-full h-fit")

    @ui.refreshable_method
    def opt_page(self):
        if self.cfg is None:
            return

        def is_int(x):
            try:
                x = int(x)
            except ValueError as _:
                return "Number of samples needs to be an integer"

        with ui.row():
            ui.button(
                "Start Optimizing", on_click=self.run_opt, color="orange"
            ).bind_visibility_from(
                self,
                "opt_running",
                backward=lambda x: not x,
            )

        def format_opt_stat(stats: ListProxy | None) -> str:
            if stats is None:
                return ""
            inner = "\n".join(stats[-10:])
            return f"```\n{inner}\n```"

        def set_candidate_ncols(n_columns):
            self.candidate_ncols = int(max(n_columns, 1))

        with ui.card(align_items="stretch").classes("w-full"):
            with ui.row():
                ui.spinner(size="4em", color="orange").bind_visibility_from(
                    self, "opt_running"
                )
                ui.markdown(f"#### Optimization Log")
            ui.markdown().bind_content_from(
                self,
                "opt_start",
                backward=lambda x: f"**{x.get()}**",
            )
            ui.markdown().bind_content_from(
                self,
                target_name="cur_opt_candidate",
                backward=lambda x: f"**{x.get()}**",
            )
            ui.markdown().bind_content_from(
                self, target_name="opt_status", backward=format_opt_stat
            )

        self.show_model_fit()

        with ui.row().bind_visibility_from(
            self, "next", backward=lambda x: x is not None
        ):
            ui.label("Column Size")
            n_columns = ui.number(min=1, max=50, step=1, value=12).on_value_change(
                lambda n: (set_candidate_ncols(n.value), self.show_next.refresh())
            )

        self.show_next()

    async def run_opt(self):
        self.model_fit = None
        self.opt_running = True

        assert self.upload_data.inputs is not None
        assert self.upload_data.targets is not None
        assert self.cfg is not None

        loop = asyncio.get_running_loop()

        X = torch.tensor(self.upload_data.inputs.values).double()
        Y = torch.tensor(self.upload_data.targets.values).double()

        opt = Optimization(X, Y, self.cfg, self.uuid)

        assert self.opt_status is not None
        update_stats = partial(
            add_opt_status,
            self.opt_status,
            self.candidate_no,
            self.cur_opt_candidate,
            self.opt_start,
        )

        with ProcessPoolExecutor(max_workers=1, mp_context=self.ctx) as pool:
            logger.info("Fitting model...", extra={"uuid": self.uuid})
            ui.notify("Fitting Model...", type="info")
            try:
                state_dict = await loop.run_in_executor(
                    pool,
                    fit_model,
                    opt,
                    update_stats,
                )
            except AttributeError as err:
                logger.error(
                    f"Error during model fitting: {err}", extra={"uuid": opt.uuid}
                )
                ui.notify(
                    f"Error during model fitting. Please report this to error message: {err}",
                    type="negative",
                )
                self.opt_running = False
                return
            except KeyboardInterrupt as e:
                self.opt_running = False
                raise e
            except Exception as err:
                logger.error(
                    f"Unknown error during model fitting: {err}",
                    extra={"uuid": opt.uuid},
                )
                ui.notify(
                    f"Unknown error during model fitting. Please report this to error message: {err}",
                    type="negative",
                )
                self.opt_running = False
                raise err

            logger.info(
                "Done fitting model. Optimizing Acquisition function...",
                extra={"uuid": self.uuid},
            )
            ui.notify(
                "Done Fitting Model. Optimizing acquisition function...", type="info"
            )

            model = get_model(opt, state_dict)

            with torch.no_grad():
                pred = model.posterior(opt.X)
            if opt.Y_opt.shape[1] == 1:
                mean = pred.mean
                stddev = pred.stddev
            else:
                mean = pred.mean
                stddev = pred.variance.sqrt()

            mean = negate_minimize_targets(opt, mean)
            y_true = negate_minimize_targets(opt, opt.Y_opt)
            self.model_fit = ModelFit(
                y_true.numpy(), mean.numpy(), stddev.numpy(), opt.opt_names
            )
            self.opt_page.refresh()

            # this minimizes the negative acquisition function
            try:
                opt_output = await loop.run_in_executor(
                    pool,
                    get_next_candidates,
                    opt,
                    state_dict,
                    update_stats,
                    self.cancel_opt,
                )
            except BrokenProcessPool as e:
                logger.error(
                    f"Unrecoverable error during optimization of acquisition function: {e}",
                    extra={"uuid": self.uuid},
                )
                ui.notify(
                    "Unrecoverable error during acquisition function optimization. Usually this means the application ran out of memory. Try reducing the number of samples.",
                    type="negative",
                )
                update_stats(
                    "Unrecoverable error during acquisition function optimization. Usually this means the application ran out of memory. Try reducing the number of samples."
                )
                self.opt_running = False
                raise e
            except UserInterrupt:
                self.cancel_opt.set(False)
                self.opt_running = False
                ui.notify("Optimization was canceled.")
                return

            if isinstance(opt_output, str):
                ui.notify(opt_output, type="negative")
                self.opt_running = False
                return

        candidates, mean, stddev, importances = opt_output
        logger.info(
            "Finished optimizing acquisition function.", extra={"uuid": self.uuid}
        )

        data = np.concatenate(
            [candidates.numpy(), mean.numpy(), stddev.numpy()], axis=1
        )

        mean_names = [
            f"mean {name}"
            for name in self.cfg.target_names
            if self.cfg.target_treatment[name] != "ignore"
        ]
        std_names = [
            f"stddev {name}"
            for name in self.cfg.target_names
            if self.cfg.target_treatment[name] != "ignore"
        ]
        self.next = pd.DataFrame(data, columns=[*self.cfg.input_names, *mean_names, *std_names])  # type: ignore
        if importances is not None:
            self.next[COMBINATION_IMPORTANCE_NAME] = importances.numpy()

        ui.notify("Done Optimizing!", type="positive")
        self.opt_running = False
        self.show_next.refresh()

    @ui.refreshable_method
    def show_upload_plot(self):
        v = [self.upload_data.inputs, self.upload_data.targets]
        v = [a for a in v if a is not None]
        if len(v) == 0:
            return
        elif len(v) == 1:
            v = v[0]
        else:
            assert len(v) == 2
            v = pd.merge(v[0], v[1], left_index=True, right_index=True)

        assert len(v.columns) > 1  # at least one input and one target

        if self.plot_columns is None:
            self.plot_columns = (v.columns[0], v.columns[1])

        v = v.reset_index()
        f = px.scatter(v, *self.plot_columns, hover_data=v.columns)
        f.update_layout(margin=dict(l=0, r=0, t=30, b=0))

        def set_plot_x(x: ValueChangeEventArguments):
            self.plot_columns = (x.value, self.plot_columns[1])
            self.show_upload_plot.refresh()

        def set_plot_y(x: ValueChangeEventArguments):
            self.plot_columns = (self.plot_columns[0], x.value)
            self.show_upload_plot.refresh()

        data_columns = [v for v in v.columns.tolist() if v != "sample id"]

        with ui.row():
            ui.plotly(f)

            with ui.column():
                with ui.row():
                    ui.label("X-Axis")
                    ui.select(data_columns, value=self.plot_columns[0]).on_value_change(
                        set_plot_x
                    )

                with ui.row():
                    ui.label("Y-Axis")
                    ui.select(data_columns, value=self.plot_columns[1]).on_value_change(
                        set_plot_y
                    )

    def upload_page(self, panels):
        with ui.row(align_items="stretch", wrap=False).classes("w-full p-0"):
            in_upload = (
                ui.upload(
                    label="Inputs",
                    on_upload=lambda x: self.handle_input_upload(x),
                    multiple=False,
                    auto_upload=True,
                )
                .props('accept=".csv"')
                .classes("w-1/2")
            )
            ui.space()
            tg_upload = (
                ui.upload(
                    label="Targets",
                    on_upload=lambda x: self.handle_target_upload(x),
                    multiple=False,
                    auto_upload=True,
                )
                .props('accept=".csv"')
                .classes("w-1/2")
            )

        def reset_uploads():
            in_upload.reset()
            tg_upload.reset()
            self.upload_data.reset()
            self.valid_upload = False

        with ui.row():
            ui.button(
                "Configure Optimization",
                on_click=lambda: panels.set_value("Configure"),
            ).bind_enabled(self, target_name="valid_upload")
            ui.button("Reset", on_click=reset_uploads, color="red-700")

        self.show_upload_plot()

    @ui.refreshable_method
    def show_next(self):
        if self.next is None or self.opt_running:
            return

        self.next[SID_COL_NAME] = get_labels(self.next.shape[0], self.candidate_ncols)
        self.next.set_index(SID_COL_NAME, inplace=True)

        io = StringIO()
        self.next.to_csv(io)
        io.seek(0)
        data_as_csv = io.read()

        with ui.row():
            ui.table.from_pandas(
                self.next.reset_index(),
                columns=[
                    {
                        "name": c,
                        "label": c,
                        "field": c,
                        "sortable": True,
                        **(
                            {":format": "value => value.toFixed(4)"}
                            if c != SID_COL_NAME
                            else {}
                        ),
                    }
                    for c in self.next.reset_index().columns
                ],
            )

            with ui.column():
                ui.button(
                    "Copy to Clipboard",
                    on_click=lambda: ui.clipboard.write(data_as_csv),
                )
                ui.button(
                    "Download",
                    on_click=lambda: ui.download(
                        data_as_csv.encode("utf-8"),
                        filename="next_iteration.csv",
                        media_type="text/csv",
                    ),
                    color="green",
                )

    def title(self):
        ui.page_title("Gaussian Process Bayesian Optimization")
        with ui.column(align_items="center"):
            with ui.row(wrap=False, align_items="baseline"):
                ui.image(FAVICON_PATH).classes("w-10")
                ui.markdown("""### Browser-Based GPBO""")
        with ui.expansion("How-To"):
            ui.markdown(
                """This app provides Gaussian Process Bayesian Optimization (GPBO). Sometimes, we want to optimize complex processes with a lot of input parameters. The naive solution of testing out a finite number of parameters for each input quickly becomes unfeasible with more and more inputs.

**GPBO** attempts to find optima in the input-output space while being sample efficient. This approach is iterative. We start with an initial set of experiments and use GPBO to generate the next set of parameters in each iteration. We stop the process when we have reached our goal.

```
|sample id | V0 | V1 | V2 |
|----------|----|----|----|
|        0 |    |    |    |
|        1 |    |    |    |
|        2 |    |    |    |
```
V0... signify the inputs or targets we want to optimize.

#### Workflow
1. Upload your inputs and targets. We expect these to be two '.csv' files. The first column in each file is the sample id, and all following columns are either input parameters or optimization targets (for input and target files, respectively).
2. Set in what direction you want to optimize each parameter, and what possible ranges are.
3. Hit Optimize, and generate new samples for you to test."""
            )
            ui.button(
                "Download Example Data",
                on_click=lambda: ui.download(EXAMPLE_PATH),
                color="green",
            )

    def set_cancel_opt(self):
        if self.cancel_opt is not None:
            self.cancel_opt.set(True)


@ui.page("/")
def main():
    app = App(uuid4().hex, UploadData())
    ui.dark_mode().enable()
    app.title()

    ui.on("beforeunload", handler=lambda _: app.set_cancel_opt())

    def update_pages(page: Tab | str, app: App):
        if isinstance(page, Tab):
            page_name = page._props["name"]
        else:
            page_name = page

        if page_name == "Upload":
            logger.info("Goto Upload Tab", extra={"uuid": app.uuid})
            pass
        elif page_name == "Configure":
            app.goto_cfg()
        elif page_name == "Optimize":
            app.goto_opt()
            app.opt_preparing = False
        else:
            logger.error(
                f"Trying to go to nonexisting page: {page_name}",
                extra={"uuid": app.uuid},
            )

    with ui.tabs(on_change=lambda x: update_pages(x.value, app)).classes(
        "w-full"
    ) as tabs:
        upload = ui.tab("Upload")
        cfg = ui.tab("Configure").bind_enabled_from(app, "valid_upload")
        opt = ui.tab("Optimize").bind_enabled_from(
            app, "step", backward=lambda x: x == AppStep.OPT
        )

    def panel_change(e: ValueChangeEventArguments):
        if app.cancel_opt is None:
            return

        if e.value == "Optimize":
            app.cancel_opt.set(False)
        else:
            app.cancel_opt.set(True)

    with ui.tab_panels(tabs, value=upload).classes("w-full") as panels:
        panels.on_value_change(panel_change)
        with ui.tab_panel(upload):
            app.upload_page(panels)
        with ui.tab_panel(cfg):
            app.cfg_page(panels)
        with ui.tab_panel(opt):
            app.opt_page()

    with ui.footer(value=True):
        ui.label(
            f"Browser-based GPBO {APP_VERSION} @ IAI - 2024, Hawo Höfer - Client Token: {app.uuid}"
        ).classes("text-xs")


if __name__ in ["__main__", "__mp_main__"]:
    main()
    ui.run(port=8080, favicon=FAVICON_PATH)
