import json
import os

import numpy as np
import off_moo_bench as ob
import torch
from gfmo_args import parse_args
from gfmo_nets import FlowMatching, VectorFieldNet
from gfmo_utils import (
    ALLTASKSDICT,
    DesignDataset,
    MultipleModels,
    SingleModel,
    SingleModelBaseTrainer,
    get_dataloader,
    tkwargs,
    training,
)
from off_moo_bench.evaluation.metrics import hv
from utils import get_quantile_solutions, set_seed
from torch.utils.data import DataLoader
import pandas as pd

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train_proxies(config, args, task):

    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)

    # For usual cases, we normalize the inputs and outputs with z-score normalization
    if config["normalize_xs"]:
        X = task.normalize_x(X, normalization_method="z-score")
    if config["normalize_ys"]:
        y = task.normalize_y(y, normalization_method="z-score")

    n_obj = y.shape[1]
    data_size, n_dim = tuple(X.shape)
    model_save_dir = config["model_save_dir"]
    os.makedirs(model_save_dir, exist_ok=True)
    if  os.path.exists(os.path.join(model_save_dir, f"ParetoFlow-Proxy-{config['task']}-{config['seed']}"+"-0.pt")):
        return
    model = MultipleModels(
        n_dim=n_dim,
        n_obj=n_obj,
        train_mode="Vallina",
        hidden_size=[2048, 2048],
        save_dir=model_save_dir,
        save_prefix=f"ParetoFlow-Proxy-{config['task']}-{config['seed']}",
    )
    model.set_kwargs(**tkwargs)

    trainer_func = SingleModelBaseTrainer

    for which_obj in range(n_obj):

        y0 = y[:, which_obj].copy().reshape(-1, 1)

        trainer = trainer_func(
            model=list(model.obj2model.values())[which_obj],
            which_obj=which_obj,
            args=args,
        )

        (train_loader, val_loader) = get_dataloader(
            X,
            y0,  # means 0.9 for training and 0.1 for validation
            batch_size=args.proxies_batch_size,
        )

        trainer.launch(train_loader, val_loader)


def train_flow_matching(config, args, task):
    # Set the seed
    set_seed(config["seed"])

    # Get the data
    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)

    # For usual cases, we normalize the inputs and outputs with z-score normalization
    if config["normalize_xs"]:
        X = task.normalize_x(X, normalization_method="z-score")
    if config["normalize_ys"]:
        y = task.normalize_y(y, normalization_method="z-score")

    # Use a subset of the data
    data_size = int(0.9*X.shape[0])
    X_test = X[data_size:]
    y_test = y[data_size:]
    X = X[:data_size]
    y = y[:data_size]

    # Obtain the number of objectives
    n_obj = y.shape[1]

    # Obtain the number of data points and the number of dimensions
    data_size, n_dim = tuple(X.shape)

    print(f"Data size: {data_size}")
    print(f"Number of objectives: {n_obj}")
    print(f"Number of dimensions: {n_dim}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Create datasets
    training_dataset = DesignDataset(X)
    val_dataset = DesignDataset(X_test)

    # Create dataloaders
    training_loader = DataLoader(
        training_dataset, batch_size=args.fm_batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.fm_batch_size, shuffle=False)

    # Create the model
    name = (config["model"]+"_"
        + config["task"]
        + "_"
        + str(config["seed"])
    )
    model_store_dir = config["model_save_dir"]
    if os.path.exists(os.path.join(model_store_dir,name+".model")):
        return
    if not (os.path.exists(model_store_dir)):
        os.makedirs(model_store_dir)

    net = VectorFieldNet(n_dim, args.fm_hidden_size)
    net = net.to(device)
    model = FlowMatching(
        net, args.fm_sigma, n_dim, args.fm_sampling_steps
    )
    model = model.to(device)

    # OPTIMIZER
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad == True], lr=args.fm_lr
    )

    # Training procedure
    nll_val = training(
        name=os.path.join(model_store_dir, name),
        max_patience=args.fm_patience,
        num_epochs=args.fm_epochs,
        model=model,
        optimizer=optimizer,
        training_loader=training_loader,
        val_loader=val_loader,
    )

    return nll_val


def sampling(config, args, task):
    # Set the seed
    set_seed(config["seed"])

    # Get the data
    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)
        data_size, dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)

    if config["normalize_xs"]:
        X = task.normalize_x(X, normalization_method="z-score")
    if config["normalize_ys"]:
        y = task.normalize_y(y, normalization_method="z-score")

    # Obtain the number of objectives
    n_obj = y.shape[1]

    # Set K to the number of objectives if args.K is 0
    if args.fm_K == 0:
        fm_K = n_obj + 1  # K = n_obj + 1
    else:
        fm_K = args.fm_K

    # Obtain the number of data points and the number of dimensions
    data_size, n_dim = tuple(X.shape)

    model_name = (config["model"]+"_"
        + config["task"]
        + "_"
        + str(config["seed"])
    )
    model_store_dir = config["model_save_dir"]

    # Load the best model
    net = VectorFieldNet(n_dim, args.fm_hidden_size)
    net = net.to(device)
    model_best = FlowMatching(
        net, args.fm_sigma, n_dim, args.fm_sampling_steps
    )
    model_best = model_best.to(device)
    model_best = torch.load(os.path.join(model_store_dir, model_name) + ".model")
    model_best = model_best.to(device)
    print(
        f"Succesfully loaded the model from {model_store_dir + model_name + '.model'}"
    )

    # Load the classifiers
    list_of_classifiers = []
    for i in range(n_obj):
        classifier = SingleModel(
            input_size=n_dim,
            which_obj=i,
            hidden_size=[2048, 2048],
            save_dir=config["model_save_dir"],
            save_prefix=f"ParetoFlow-Proxy-{config['task']}-{config['seed']}",
        )
        classifier.load()
        classifier = classifier.to(device)
        classifier.eval()
        list_of_classifiers.append(classifier)
    print(f"Loaded {len(list_of_classifiers)} classifiers successfully.")

    # Conditional sampling
    x_samples, hv_results = model_best.gfmo_sample(
        list_of_classifiers,
        T=args.fm_sampling_steps,
        O=args.fm_O,
        K=fm_K,
        num_solutions=args.fm_num_solutions,
        distance=args.fm_distance_metrics,
        init_method=args.fm_init_method,
        g_t=args.fm_gt,
        task=task,
        task_name=config["task"],
        t_threshold=args.fm_threshold,
        adaptive=args.fm_adaptive,
        gamma=args.fm_gamma,
    )

    # Denormalize the solutions
    res_x = x_samples
    if config["normalize_xs"]:
        res_x = task.denormalize_x(res_x, normalization_method="z-score")

    if task.is_discrete:
        res_x = res_x.reshape(-1, dim, n_classes)
        res_x = task.to_integers(res_x)
    if task.is_sequence:
        res_x = task.to_integers(res_x)

    res_y = task.predict(res_x)
    # I noticed that there's a weird bug for task DTLZ2, the shape of res_x
    # is (batch_size, n_dim), but the shape of res_y is (n_obj, batch_size)
    # Need to fix this issue and understand why this happens
    # Simply transpose the res_y
    if res_y.shape[0] != res_x.shape[0]:
        res_y = res_y.T

    # Store the results
    if not (os.path.exists(config["results_dir"])):
        os.makedirs(config["results_dir"])

    np.save(os.path.join(config["results_dir"], "res_x.npy"), res_x)
    np.save(os.path.join(config["results_dir"], "res_y.npy"), res_y)

    df = pd.DataFrame([hv_results])
    filename = os.path.join(config["results_dir"], "hv_results.csv")
    df.to_csv(filename, index=False)

    return res_x, res_y

