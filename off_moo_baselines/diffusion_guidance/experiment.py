import os
import sys
import wandb
import torch
import numpy as np
import pandas as pd
import datetime
import json
from copy import deepcopy

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(BASE_PATH)

import off_moo_bench as ob
from utils import (
    set_seed,
    get_quantile_solutions,
    spread,
    spacing,
    pairwise_euclidean_distances,
)
from off_moo_baselines.diffusion_guidance.ddpm_guidance import (
    train,
    Diffusion,
    train_preference,
)
from off_moo_baselines.diffusion_guidance.modules import (
    Preference_model,
    Model_unconditional,
    save_model,
    load_model,
)
from off_moo_baselines.data import tkwargs, get_dataloader, get_preference_rankings
from off_moo_bench.task_set import *
from off_moo_bench.evaluation.metrics import hv


def run(config: dict):
    if config["task"] in ALLTASKSDICT.keys():
        config["task"] = ALLTASKSDICT[config["task"]]
    results_dir = os.path.join(
        config["results_dir"],
        f"{config['model']}-{config['train_mode']}-{config['task']}",
    )
    config["results_dir"] = results_dir
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f"-ts-{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"
    run_name = f"{config['model']}-{config['train_mode']}-seed{config['seed']}-{config['task']}"

    logging_dir = os.path.join(config["results_dir"], run_name + ts_name)
    os.makedirs(logging_dir, exist_ok=True)

    if config["use_wandb"]:
        if "wandb_api" in config.keys():
            wandb.login(key=config["wandb_api"])

        wandb.init(
            project="Offline-MOO",
            name=run_name + ts_name,
            config=config,
            group=f"{config['model']}-{config['train_mode']}",
            job_type=config["run_type"],
            mode="online",
            dir=os.path.join(config["results_dir"], ".."),
        )

    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(config, f, indent=4)

    set_seed(config["seed"])

    task = ob.make(config["task"])

    X = task.x.copy()
    y = task.y.copy()
    if config["subsample"]:
        X_pref, y_pref = task.get_N_non_dominated_solutions(
            N=int(X.shape[0] * config["subsample_ratio"]), return_x=True, return_y=True
        )

    X_test = task.x_test.copy()
    y_test = task.y_test.copy()

    if config["to_logits"]:
        assert task.is_discrete
        task.map_to_logits()
        X = task.to_logits(X)
        X_test = task.to_logits(X_test)
    if config["normalize_xs"]:
        task.map_normalize_x()
        X = task.normalize_x(X)
        X_test = task.normalize_x(X_test)
    if config["normalize_ys"]:
        task.map_normalize_y()
        y = task.normalize_y(y)
        y_test = task.normalize_y(y_test)

    if config["to_logits"]:
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
        X_test = X_test.reshape(-1, n_dim * n_classes)
    else:
        data_size, n_dim = tuple(X.shape)
    n_obj = y.shape[1]
    hypervolumes = []
    for i in range(y.shape[0]):
        # hypervolumes.append(hv(task.normalize_y(task.nadir_point), y[i], config['task']))
        hypervolumes.append(1.0)
    ind_pareto_rank = None
    if config["use_diversity_metric"]:
        ind_pareto_rank = get_preference_rankings(
            y_pref if config["subsample"] else y,
            task.normalize_y(task.nadir_point),
            config["task"],
            config["diversity_metric"],
        )
    hypervolumes = np.array(hypervolumes)
    model_save_dir = config["model_save_dir"]
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(
        model_save_dir,
        f"{config['model']}-{config['train_mode']}-{config['task']}-{config['seed']}-0.pt",
    )
    preference_save_path = model_save_path.replace("-0.pt", "-preference.pt")

    (train_loader_pref, val_loader_pref, _, train_loader, _, _) = get_dataloader(
        X,
        y,
        X_test,
        y_test,
        X_pref=X_pref if config["subsample"] else None,
        y_pref=y_pref if config["subsample"] else None,
        val_ratio=0.9,
        batch_size=config["batch_size"],
        preference_loader=True,
        hypervolumes=hypervolumes,
        three_dim_out=config["three_dim_out"],
        use_diversity_metric=config["use_diversity_metric"],
        pareto_rankings=ind_pareto_rank,
        diversity_score_threshold=config["diversity_score_threshold"],
    )
    if os.path.exists(model_save_path):
        model_uncond = Model_unconditional(dim=n_dim)
        load_model(model_uncond, model_save_path, device=tkwargs["device"])
        diffusion = Diffusion(img_size=n_dim, device=tkwargs["device"])
    else:
        model_uncond, diffusion = train(train_loader)
        save_model(
            model=model_uncond, save_path=model_save_path, device=model_uncond.device
        )

    if os.path.exists(preference_save_path):
        preference_model = Preference_model(
            input_dim=train_loader.dataset[0][0].shape[-1],
            device=tkwargs["device"],
            three_dim_out=config["three_dim_out"],
        ).to(tkwargs["device"])
        load_model(preference_model, preference_save_path, device=tkwargs["device"])
    else:
        preference_model = train_preference(
            dataloader=train_loader_pref,
            diffusion=diffusion,
            val_loader=val_loader_pref,
            config=config,
            model_save_path=preference_save_path,
            three_dim_out=config["three_dim_out"],
        )

    X_d_best, d_best = task.get_N_non_dominated_solutions(
        N=256, return_x=True, return_y=True
    )
    try:
        res_y_pf_ideal = task.problem.get_pareto_front()
    except NotImplementedError:
        res_y_pf_ideal = None
    X_d_best = torch.tensor(X_d_best[-1]).unsqueeze(0).repeat(256, 1)
    samples = diffusion.sample_with_preference(
        model_uncond,
        256,
        preference_model,
        torch.tensor(X_d_best),
        cfg_scale=10.0,
        return_latents=False,
        ddim=False,
    )
    samples_20 = diffusion.sample_with_preference(
        model_uncond,
        256,
        preference_model,
        torch.tensor(X_d_best),
        cfg_scale=20.0,
        return_latents=False,
        ddim=False,
    )
    if config["normalize_xs"]:
        task.map_denormalize_x()
        samples = task.denormalize_x(samples.cpu().numpy())
        samples_20 = task.denormalize_x(samples_20.cpu().numpy())
        # latents = task.denormalize_x(latents.cpu().numpy())
    else:
        samples = samples.cpu().numpy()
        samples_20 = samples_20.cpu().numpy()
        # latents = latents.cpu().numpy()
    res_y = task.predict(samples)
    res_y_20 = task.predict(samples_20)
    res_y_75_percentile = get_quantile_solutions(res_y, 0.75)
    res_y_50_percentile = get_quantile_solutions(res_y, 0.5)
    res_y_20_75_percentile = get_quantile_solutions(res_y_20, 0.75)
    res_y_20_50_percentile = get_quantile_solutions(res_y_20, 0.5)
    np.save(os.path.join(logging_dir, "res_y.npy"), res_y)
    np.save(os.path.join(logging_dir, "res_x.npy"), samples)
    np.save(os.path.join(logging_dir, "res_y_20.npy"), res_y_20)
    np.save(os.path.join(logging_dir, "res_x_20.npy"), samples_20)
    hv_value = hv(
        task.normalize_y(task.nadir_point), task.normalize_y(res_y), config["task"]
    )
    if res_y_pf_ideal is not None:
        hv_pf_ideal = hv(
            task.normalize_y(task.nadir_point),
            task.normalize_y(res_y_pf_ideal),
            config["task"],
        )
    hv_d_best = hv(task.normalize_y(task.nadir_point), d_best, config["task"])
    hv_value_75_percentile = hv(
        task.normalize_y(task.nadir_point),
        task.normalize_y(res_y_75_percentile),
        config["task"],
    )
    hv_value_50_percentile = hv(
        task.normalize_y(task.nadir_point),
        task.normalize_y(res_y_50_percentile),
        config["task"],
    )
    hv_value_20 = hv(
        task.normalize_y(task.nadir_point), task.normalize_y(res_y_20), config["task"]
    )
    hv_value_20_75_percentile = hv(
        task.normalize_y(task.nadir_point),
        task.normalize_y(res_y_20_75_percentile),
        config["task"],
    )
    hv_value_20_50_percentile = hv(
        task.normalize_y(task.nadir_point),
        task.normalize_y(res_y_20_50_percentile),
        config["task"],
    )
    spread_value = spread(task.normalize_y(res_y))
    spacing_value = spacing(task.normalize_y(res_y))
    spread_value_50_percentile = spread(task.normalize_y(res_y_50_percentile))
    spacing_value_50_percentile = spacing(task.normalize_y(res_y_50_percentile))
    spread_value_75_percentile = spread(task.normalize_y(res_y_75_percentile))
    spacing_value_75_percentile = spacing(task.normalize_y(res_y_75_percentile))
    spread_value_20 = spread(task.normalize_y(res_y_20))
    spacing_value_20 = spacing(task.normalize_y(res_y_20))
    spread_value_20_50_percentile = spread(task.normalize_y(res_y_20_50_percentile))
    spacing_value_20_50_percentile = spacing(task.normalize_y(res_y_20_50_percentile))
    spread_value_20_75_percentile = spread(task.normalize_y(res_y_20_75_percentile))
    spacing_value_20_75_percentile = spacing(task.normalize_y(res_y_20_75_percentile))
    ped_20 = pairwise_euclidean_distances(task.normalize_y(res_y_20))
    ped_20_75 = pairwise_euclidean_distances(task.normalize_y(res_y_20_75_percentile))
    ped_20_50 = pairwise_euclidean_distances(task.normalize_y(res_y_20_50_percentile))
    ped = pairwise_euclidean_distances(task.normalize_y(res_y))
    ped_75 = pairwise_euclidean_distances(task.normalize_y(res_y_75_percentile))
    ped_50 = pairwise_euclidean_distances(task.normalize_y(res_y_50_percentile))
    print(f"HV value {hv_value} HV D best {hv_d_best}")
    hv_results = {
        "hypervolume_10/D(best)": hv_d_best,
        "hypervolume_10/100th": hv_value,
        "hypervolume_10/75th": hv_value_75_percentile,
        "hypervolume_10/50th": hv_value_50_percentile,
        "hypervolume_20/100th": hv_value_20,
        "hypervolume_20/75th": hv_value_20_75_percentile,
        "hypervolume_20/50th": hv_value_20_50_percentile,
        "Spread_10/100th": spread_value,
        "Spacing_10/100th": spacing_value,
        "Spread_10/75th": spread_value_75_percentile,
        "Spacing_10/75th": spacing_value_75_percentile,
        "Spread_20/100th": spread_value_20,
        "Spacing_20/100th": spacing_value_20,
        "Spread_20/75th": spread_value_20_75_percentile,
        "Spacing_20/75th": spacing_value_20_75_percentile,
        "Spread_20/50th": spread_value_20_50_percentile,
        "Spacing_20/50th": spacing_value_20_50_percentile,
        "Spread_10/50th": spread_value_50_percentile,
        "Spacing_10/50th": spacing_value_50_percentile,
        "ped_10/100th": ped,
        "ped_10/75th": ped_75,
        "ped_10/50th": ped_50,
        "ped_20/100th": ped_20,
        "ped_20/75th": ped_20_75,
        "ped_20/50th": ped_20_50,
        "evaluation_step": 1,
    }

    df = pd.DataFrame([hv_results])
    filename = os.path.join(logging_dir, "hv_results.csv")
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    from utils import process_args

    config = process_args(return_dict=True)

    save_dir = "./"
    results_dir = os.path.join(save_dir, "results")
    model_save_dir = os.path.join(save_dir, "model")

    config["results_dir"] = results_dir
    config["model_save_dir"] = model_save_dir
    run(config)
