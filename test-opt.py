import os

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as multiprocessing

import opt


def do_nothing(_):
    pass


def run_opt_from_path(base_path: str):
    with open(os.path.join(base_path, "opt_cfg.json"), "r") as file:
        cfg: opt.OptimizeConfig = opt.OptimizeConfig.from_json(file.read())

    x = torch.tensor(
        pd.read_csv(os.path.join(base_path, "example_inputs.csv"), header=0)
        .drop(["sample id"], axis=1)
        .values
    )

    y = torch.tensor(
        pd.read_csv(os.path.join(base_path, "example_targets.csv"), header=0)
        .drop(["sample id"], axis=1)
        .values
    )

    cfg.num_samples = 2
    optimization = opt.Optimization(x, y, cfg, "local-test")
    state_dict = opt.fit_model(optimization, print)

    ctx = multiprocessing.get_context("spawn")
    mgr = ctx.Manager()
    cancel_opt = mgr.Value("b", False)

    r = opt.get_next_candidates(optimization, state_dict, print, cancel_opt)

    if isinstance(r, str):
        print(f"Error: {r}")
        exit(1)


def test_single_target():
    base_path = os.path.join(os.path.dirname(__file__), "example_data_st")
    run_opt_from_path(base_path)


def test_multi_target():
    base_path = os.path.join(os.path.dirname(__file__), "example_data")
    run_opt_from_path(base_path)
