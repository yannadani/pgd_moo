import os
import numpy as np
import torch
import sys 
import yaml 
from typing import List, Optional
from types import SimpleNamespace

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

tkwargs = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.float32
}

now_fronts = None
now_seed = None

def calc_crowding_distance(F) -> np.ndarray:

    if isinstance(F, list) or isinstance(F, np.ndarray):
        F = torch.tensor(F).to(**tkwargs)

    n_points, n_obj = F.shape

    # sort each column and get index
    I = torch.argsort(F, dim=0, descending=False)

    # sort the objective space values for the whole matrix
    F_sorted = torch.gather(F, 0, I)

    # calculate the distance from each point to the last and next
    inf_tensor = torch.full((1, n_obj), float('inf'), device=F.device, dtype=F.dtype)
    neg_inf_tensor = torch.full((1, n_obj), float('-inf'), device=F.device, dtype=F.dtype)
    dist = torch.cat([F_sorted, inf_tensor], dim=0) - torch.cat([neg_inf_tensor, F_sorted], dim=0)

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = torch.max(F_sorted, dim=0).values - torch.min(F_sorted, dim=0).values
    norm[norm == 0] = float('nan')

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist[:-1], dist[1:]
    dist_to_last, dist_to_next = dist_to_last / norm, dist_to_next / norm

    # if we divide by zero because all values in one column are equal replace by none
    dist_to_last[torch.isnan(dist_to_last)] = 0.0
    dist_to_next[torch.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = torch.argsort(I, dim=0, descending=False)
    crowding_dist = torch.sum(
        torch.gather(dist_to_last, 0, J) + torch.gather(dist_to_next, 0, J),
        dim=1
    ) / n_obj

    return crowding_dist.detach().cpu().numpy()

def calc_crowding_distance_np(F):
    n_points, n_obj = F.shape

    # sort each column and get index
    I = np.argsort(F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    F = F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.row_stack([F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), F])

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist, np.copy(dist)
    dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

    # if we divide by zero because all values in one columns are equal replace by none
    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = np.argsort(I, axis=0)
    cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    return cd


def spacing(pop_obj):
    """
    Calculate the Spacing (S) metric for a set of solutions in objective space.
    
    Parameters
    ----------
    pop_obj : np.ndarray of shape (N, M)
        A 2D array where each row represents a solution's objectives (N solutions, M objectives).
    
    Returns
    -------
    float
        The spacing metric value. Lower is better (more uniform spacing).
    """
    # Number of solutions
    N = len(pop_obj)
    if N < 2:
        return 0.0  # Spacing is not well-defined for 1 or 0 solutions
    
    # Compute the distance matrix (NxN) in objective space
    # D[i, j] = distance between solution i and j
    D = np.linalg.norm(pop_obj[:, None, :] - pop_obj[None, :, :], axis=2)
    
    # Ignore self-distances by setting them to infinity
    np.fill_diagonal(D, np.inf)
    
    # d[i] = distance to the nearest neighbor for solution i
    d = np.min(D, axis=1)
    
    # Mean of nearest-neighbor distances
    d_mean = np.mean(d)
    
    # Spacing = sqrt( sum( (d_i - d_mean)^2 ) / (N-1) )
    S = np.sqrt(np.sum((d - d_mean)**2) / (N - 1))
    return S

def pairwise_euclidean_distances(X):
    """
    Compute the pairwise Euclidean distance between points using inbuilt NumPy functions.
    
    Parameters:
    X (numpy array): A matrix of shape (N, M) where N is the number of points and M is the dimensionality of each point.
    
    Returns:
    numpy array: A matrix of shape (N, N) containing pairwise Euclidean distances.
    """
    # Compute pairwise distances using broadcasting and np.linalg.norm
    dist = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis], axis=2).mean()
    return dist

def spread(pop_obj, f_extreme=None):
    """
    Calculate the Spread (Delta) metric for a set of solutions in 2D objective space.
    
    Parameters
    ----------
    pop_obj : np.ndarray of shape (N, 2)
        A 2D array where each row is a solution's objectives: [obj1, obj2].
    f_extreme : tuple or None
        A tuple (f1, fN) containing the extreme solutions in objective space.
        Each is a 1D array of length 2.
        If None, the extremes are taken from the first and last in the sorted set.
    
    Returns
    -------
    float
        The spread (Delta) metric value.
        0 indicates perfectly even distribution.
    """
    # For 2D problems, sort by the first objective
    N = len(pop_obj)
    if N < 2:
        return 0.0  # not defined or zero for <2 solutions
    
    # Sort solutions by objective 1
    sorted_indices = np.argsort(pop_obj[:, 0])
    sorted_pop = pop_obj[sorted_indices]
    
    # Distances between consecutive solutions
    consecutive_dists = np.linalg.norm(np.diff(sorted_pop, axis=0), axis=1)  # length = N-1
    d_mean = np.mean(consecutive_dists) if len(consecutive_dists) > 0 else 0.0
    
    # Determine extremes
    if f_extreme is None:
        # Use the first and last solutions from the sorted set as extremes
        f1 = sorted_pop[0]
        fN = sorted_pop[-1]
    else:
        f1, fN = f_extreme

    # Distances of the boundary solutions to the known extremes
    d_f = np.linalg.norm(sorted_pop[0] - f1)
    d_l = np.linalg.norm(sorted_pop[-1] - fN)
    
    # Sum of absolute deviations from the mean consecutive distance
    d_sum = np.sum(np.abs(consecutive_dists - d_mean))
    
    # Deb's spread formula
    # Delta = (d_f + d_l + sum(|d_i - d_mean|)) / (d_f + d_l + (N - 1) * d_mean)
    denominator = d_f + d_l + (N - 1) * d_mean
    if denominator == 0:
        return 0.0

    Delta = (d_f + d_l + d_sum) / denominator
    return Delta

def _get_fronts(Y_all):
    global now_fronts
    if now_fronts is not None:
        return now_fronts
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    fronts = NonDominatedSorting().do(Y_all, return_rank=True)[0]
    now_fronts = fronts 
    return fronts

def get_N_nondominated_index(Y_all, num_ret, is_all_data=False) -> List[int]:
    if is_all_data:
        fronts = _get_fronts(Y_all)
    else:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        fronts = NonDominatedSorting().do(Y_all, return_rank=True, n_stop_if_ranked=num_ret)[0]
    indices_cnt = 0
    indices_select = []
    for front in fronts:
        if indices_cnt + len(front) < num_ret:
            indices_cnt += len(front)
            indices_select += [int(i) for i in front]
        else:
            n_keep = num_ret - indices_cnt
            F = Y_all[front]

            from pymoo.util.misc import find_duplicates
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]

            _F = F[is_unique]
            _d = calc_crowding_distance(_F)

            d = np.zeros(len(front))
            d[is_unique] = _d 
            I = np.argsort(d)[-n_keep:]
            indices_select += [int(i) for i in I]
            break
        
    return indices_select

def get_quantile_solutions(Y_all: np.ndarray, quantile) -> np.ndarray:
    assert 0 < quantile < 1
    n = len(Y_all)
    n_remove = int(n * (1-quantile))
    indices_to_remove = get_N_nondominated_index(Y_all, n_remove)
    indices_to_keep = np.ones(n)
    indices_to_keep[indices_to_remove] = 0
    return Y_all[np.where(indices_to_keep == 1)[0]]


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True
    global now_seed
    now_seed = seed

def process_args(return_dict=False):
    params = [arg.lstrip("--") for arg in sys.argv if arg.startswith("--")]
    cmd_config_dict = {} 
    for arg in params:
        key, value = arg.split('=')
        try:
            cmd_config_dict[key] = eval(value)
        except:
            cmd_config_dict[key] = value 
            
    # default config
    config_path = os.path.join(
        base_path,
        "configs",
        "default.yaml"
    )
    assert os.path.exists(config_path), f"Config {config_path} not found"
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in cmd_config_dict.items():
        config_dict[key] = value
        
    # model config
    model_config_path =  os.path.join(
        base_path,
        "configs",
        "algorithm",
        f"{config_dict['model']}-{config_dict['train_mode']}.yaml"
    )
    assert os.path.exists(model_config_path), \
        f"Model config {model_config_path} not found"
    with open(model_config_path, 'r') as f:
        try:
            config_dict.update(yaml.load(f, Loader=yaml.FullLoader))
        except:
            pass

    # task config
    task_config_path =  os.path.join(
        base_path,
        "configs",
        "task",
        f"{config_dict['task']}.yaml"
    )
    
    default_task_config_path =  os.path.join(
        base_path,
        "configs",
        "task",
        f"default.yaml"
    )
    assert os.path.exists(task_config_path) or \
        os.path.exists(default_task_config_path), \
        f"Problem config {task_config_path} or {default_task_config_path} not found"
    try:
        with open(task_config_path, 'r') as f:
            try:
                config_dict.update(yaml.load(f, Loader=yaml.FullLoader))
            except:
                pass
    except:
        with open(default_task_config_path, 'r') as f:
            try:
                config_dict.update(yaml.load(f, Loader=yaml.FullLoader))
            except:
                pass

    for key, value in cmd_config_dict.items():
        config_dict[key] = value 
    
    print("All config:", config_dict)
    
    return config_dict if return_dict else SimpleNamespace(**config_dict)
    

def read_data(env_name, filter_type='best', return_x=True, return_y=True, return_rank=True):
    try:
        assert return_x or return_y or return_rank, "Illegal params."
        env_name = env_name.lower()
        data_path = os.path.join(base_path, "data", env_name)
        x_file = os.path.join(data_path, f"{env_name}-x-0.npy") if return_x else None
        y_file = os.path.join(data_path, f"{env_name}-y-0.npy") if return_y else None
        rank_file = os.path.join(data_path, f"{env_name}-rank-0.npy") if return_rank else None
        
        x = np.load(x_file) if return_x else None
        y = np.load(y_file) if return_y else None
        rank = np.load(rank_file) if return_rank else None

        return (x if return_x else None,
                y if return_y else None, 
                rank if return_rank else None)
    except:
        assert return_x or return_y or return_rank, "Illegal params."
        env_name = env_name.lower()
        data_path = os.path.join(base_path, "data", env_name, filter_type)
        x_file = os.path.join(data_path, f"{env_name}-x-0.npy") if return_x else None
        y_file = os.path.join(data_path, f"{env_name}-y-0.npy") if return_y else None
        rank_file = os.path.join(data_path, f"{env_name}-rank-0.npy") if return_rank else None
        
        x = np.load(x_file) if return_x else None
        y = np.load(y_file) if return_y else None
        rank = np.load(rank_file) if return_rank else None

        return (x if return_x else None,
                y if return_y else None, 
                rank if return_rank else None)