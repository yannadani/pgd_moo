import torch 
import numpy as np 

from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from typing import Optional
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from multiprocessing import cpu_count
from off_moo_bench.evaluation.metrics import hv
from utils import calc_crowding_distance_np
import time
tkwargs = {
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "dtype": torch.float32
}


def get_ind_hypervolume_contribution(pareto_front: np.ndarray, reference_point: np.ndarray, task_name: str):
    total_hypervolume = hv(reference_point, pareto_front, task_name=task_name)
    individual_contributions = []
    for i in range(len(pareto_front)):
        # Remove the i-th point
        remaining_points = np.delete(pareto_front, i, axis=0)
        # Compute hypervolume without the i-th point
        hv_without_i = hv(reference_point, remaining_points, task_name=task_name)
        # Contribution of the i-th point
        contribution = total_hypervolume - hv_without_i
        individual_contributions.append(contribution)
    return np.array(individual_contributions)

def get_crowding_distance(y: np.ndarray):
    return calc_crowding_distance_np(y)

def get_preference_rankings(y: np.ndarray, reference_point: np.ndarray, task_name: str, metric: str = "hypervolume"):
    nds = NonDominatedSorting()
    contributions_all = {"score": np.zeros((y.shape[0],)), "rank": np.zeros((y.shape[0],)).astype(int), "front": np.zeros((y.shape[0],)).astype(int)}
    fronts = nds.do(y)
    curr_rank = 0
    for e, front in enumerate(fronts):
        y_front = y[front]
        contributions_all["front"][front] = e
        if metric == "hypervolume":
            contributions = get_ind_hypervolume_contribution(y_front, reference_point, task_name=task_name)
        elif metric == "crowding_distance":
            contributions = get_crowding_distance(y_front)
        contributions_all["score"][front] = contributions
        ranking = np.argsort(contributions)[::-1]
        for i in range(len(front)):
            contributions_all["rank"][front[ranking[i]]] = curr_rank + i
        curr_rank += len(front)        
    return contributions_all

def get_dataloader(X: np.ndarray,
                   y: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   X_pref: Optional[np.ndarray] = None,
                   y_pref: Optional[np.ndarray] = None,
                   val_ratio: float = 0.9,
                   batch_size: int = 32,
                   hypervolumes: Optional[np.ndarray] = None,
                   preference_loader: bool = False,
                   three_dim_out: bool = False,
                   use_diversity_metric: bool = False,
                   pareto_rankings: Optional[np.ndarray] = None,
                   diversity_score_threshold: Optional[float] = 0.0):
    if X_pref is None:
        X_pref = X
        y_pref = y
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(**tkwargs)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).to(**tkwargs)
    if isinstance(X_pref, np.ndarray):
        X_pref = torch.from_numpy(X_pref).to(**tkwargs)
    if isinstance(y_pref, np.ndarray):
        y_pref = torch.from_numpy(y_pref).to(**tkwargs)
    if isinstance(X_test, np.ndarray):
        X_test = torch.from_numpy(X_test).to(**tkwargs)
    if isinstance(y_test, np.ndarray):
        y_test = torch.from_numpy(y_test).to(**tkwargs)
    if isinstance(hypervolumes, np.ndarray):
        hypervolumes = torch.from_numpy(hypervolumes).to(**tkwargs)
    lengths = int(val_ratio*len(X))
    lengths_pref = int(val_ratio*len(X_pref))
    X_train = X[:lengths]
    y_train = y[:lengths]
    X_val = X[lengths:]
    y_val = y[lengths:]
    X_pref_train = X_pref[:lengths_pref]
    y_pref_train = y_pref[:lengths_pref]
    X_pref_val = X_pref[lengths_pref:]
    y_pref_val = y_pref[lengths_pref:]

    if preference_loader:
        if use_diversity_metric:
            if pareto_rankings is None:
                raise ValueError("hv_rankings must be provided if use_individual_hv_contributions is True")
            assert three_dim_out == False
            pareto_rankings_train = {}
            pareto_rankings_val = {}
            for key in pareto_rankings.keys():
                pareto_rankings_train[key] = pareto_rankings[key][:lengths_pref]
                pareto_rankings_val[key] = pareto_rankings[key][lengths_pref:]
            train_dataset_pref = preference_dataset_ind_hv_contrib(X_pref_train, y_pref_train, pareto_rankings_train, diversity_score_threshold=diversity_score_threshold)
            val_dataset_pref = preference_dataset_ind_hv_contrib(X_pref_val, y_pref_val, pareto_rankings_val, diversity_score_threshold=diversity_score_threshold)
        else:
            train_dataset_pref = preference_dataset(X_pref_train, y_pref_train, three_dim_out=three_dim_out)
            val_dataset_pref = preference_dataset(X_pref_val, y_pref_val, three_dim_out=three_dim_out)
        test_dataset_pref = preference_dataset(X_test, y_test, three_dim_out=three_dim_out)
        train_loader_perf = DataLoader(train_dataset_pref,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  #pin_memory=True,
                                  drop_last=False)
        val_loader_perf = DataLoader(val_dataset_pref,
                                batch_size=batch_size * 4,
                                shuffle=False,
                                #pin_memory=True,
                                drop_last=False)
        test_loader_perf = DataLoader(test_dataset_pref,
                                batch_size=batch_size * 4,
                                shuffle=False,
                                drop_last=False)
        
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    if hypervolumes is not None:
        train_dataset = TensorDataset(X_train, y_train, hypervolumes[:lengths])
        val_dataset = TensorDataset(X_val, y_val, hypervolumes[lengths:])
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            #pin_memory=True,
                            drop_last=False)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size * 4,
                            shuffle=False,
                            #pin_memory=True,
                            drop_last=False)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size * 4,
                            shuffle=False,
                            #pin_memory=True,
                            drop_last=False)
    if preference_loader:
        return train_loader_perf, val_loader_perf, test_loader_perf, train_loader, val_loader, test_loader
    return train_loader, val_loader, test_loader

class preference_dataset(Dataset):
    def __init__(self, X, y, three_dim_out=False):
        self.X = X
        self.y = y
        self.three_dim_out = three_dim_out
        self.nds = NonDominatedSorting()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x_1 = self.X[idx]
        idx_2 = idx
        res = torch.zeros(2)
        if self.three_dim_out:
            coin_flip = torch.randint(2, size=(1,))[0]
        count = 0
        while count < 1000: 
            idx_2 = torch.randint(len(self.X), size=(1,))[0]
            x_2 = self.X[idx_2]
            y = self.y[torch.tensor([idx, idx_2])]
            _, res = self.nds.do(y.cpu().numpy(), return_rank=True)
            if res.sum() == 0 and coin_flip == 1:
                break
            elif not res.sum()==0  and coin_flip == 0:
                break
            count += 1
        if res.sum() == 0:
            ind = torch.tensor([2.0])
        else:
            if res[0] == 0: #x_1 dominates x_2
                ind = torch.tensor([0.0])
            else:
                ind = torch.tensor([1.0])
        return x_1, x_2, ind 

class preference_dataset_ind_hv_contrib(Dataset):
    def __init__(self, X, y, hv_contribution_rankings, diversity_score_threshold=0.0):
        self.X = X
        self.y = y
        self.hv_contribution_rankings = hv_contribution_rankings
        self.diversity_score_threshold = diversity_score_threshold
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x_1 = self.X[idx]
        while True:
            idx_2 = torch.randint(len(self.X), size=(1,))[0]
            x_2 = self.X[idx_2]
            if self.hv_contribution_rankings["front"][idx] == self.hv_contribution_rankings["front"][idx_2]:
                if self.hv_contribution_rankings["score"][idx_2] > self.diversity_score_threshold:
                    break
            else:
                break
        if self.hv_contribution_rankings["rank"][idx] < self.hv_contribution_rankings["rank"][idx_2]:
            ind = torch.tensor([0.0])
        else:
            ind = torch.tensor([1.0])
        return x_1, x_2, ind 
    
def spearman_correlation(x, y):
    n = x.size(0)
    _, rank_x = x.sort(0)
    _, rank_y = y.sort(0)
    
    d = rank_x - rank_y
    d_squared_sum = (d ** 2).sum(0).float()
    
    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return rho

task2fullname = {
    "zdt1": "ZDT1-Exact-v0",
    "re21": "RE21-Exact-v0",
    "dtlz1": "DTLZ1-Exact-v0",
}