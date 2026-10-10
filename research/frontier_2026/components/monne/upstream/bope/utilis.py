
import torch
from itertools import combinations
from botorch.utils.sampling import draw_sobol_samples
import random
import numpy as np
import heapq
from scipy.stats import norm
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double



def get_initial_points(bounds,num,device,dtype,seed=0):

        train_x = draw_sobol_samples(
        bounds=bounds, n=num, q=1,seed=seed).reshape(num,-1).to(device, dtype=dtype)

        return train_x


def get_random_points(bounds, num, device, dtype, seed=0):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    
    # Generate random points in the unit cube [0, 1]^D
    dim = bounds.shape[1]  # Extract dimensionality D
    random_points = torch.rand(num, dim, device=device, dtype=dtype)
    
    # Extract lower and upper bounds
    lower_bounds = bounds[0]  # First row: lower bounds
    upper_bounds = bounds[1]  # Second row: upper bounds
    
    # Scale and shift points to the specified bounds
    train_x = lower_bounds + (upper_bounds - lower_bounds) * random_points
    
    return train_x

def is_dominated(sol_a, sol_b):

    return torch.all(sol_b <= sol_a) and torch.any(sol_b < sol_a)


def pareto_front_with_corresponding_x(X, Y):

    pareto_mask = torch.ones(Y.shape[0], dtype=bool)  # Initially assume all solutions are non-dominated

    for i, sol_a in enumerate(Y):
        for j, sol_b in enumerate(Y):
            if i != j and is_dominated(sol_a, sol_b):
                pareto_mask[i] = False
                break

    pareto_X = X[pareto_mask]
    pareto_Y = Y[pareto_mask]
    return pareto_X, pareto_Y


def generate_pairs(Y):

    num_solutions = Y.shape[0]
    indices = list(range(num_solutions))
    return list(combinations(indices, 2))

def sample_pairs(pairs, num_pairs):

    if num_pairs > len(pairs):
        num_pairs = len(pairs)
        #raise ValueError("Requested number of pairs exceeds the total number of available pairs.")
    return random.sample(pairs, num_pairs)

def get_corresponding_XY_pairs(X, Y, num_pairs):

    all_pairs = generate_pairs(Y)
    sampled_pairs = sample_pairs(all_pairs, num_pairs)
    results = []

    for i, j in sampled_pairs:
        results.append((X[i], X[j], Y[i], Y[j]))

    return results



def tensor_to_tuple(tensor):
    return tuple(tensor.tolist())



def generate_new_pair(Y_BO, existing_comparisons):
    
    A, B = Y_BO.shape  # Get dimensions of Y_BO

    # Generate all possible unique pairs of indices
    all_indices = list(range(A))
    all_indices = list(combinations(all_indices, 2))
    all_pairs = [(tensor_to_tuple(Y_BO[pair[0]].reshape(-1,)),tensor_to_tuple(Y_BO[pair[1]].reshape(-1,))) for pair in all_indices]

    # Convert existing comparisons to tuples
    existing_comparisons_tuples = [(tensor_to_tuple(pair[0].reshape(-1,)), tensor_to_tuple(pair[1].reshape(-1,))) for pair in existing_comparisons]
    
    # Remove existing comparisons from the list of all pairs
    all_pairs = [pair for pair in all_pairs if pair not in existing_comparisons_tuples and (pair[1],pair[0]) not in existing_comparisons_tuples ]
    

    # Check if there are any pairs left to sample
    if not all_pairs:
        raise ValueError("No more pairs available to sample.")

    random_element = random.choice(all_pairs)

    return (torch.tensor(random_element[0]),torch.tensor(random_element[1]))



def pairwise_comparison_utility_noise(utility,pair,noise=0.,seed=0):

    torch.manual_seed(seed)
    noise_temp = torch.normal(mean=0, std=noise, size=(1,))
    
    if utility(pair[0])+noise_temp>=utility(pair[1]):
        label = 1
    else:
        label = -1
    
    return label



def pairwise_comparison_choice_noise(utility,pair,noise=0.):


    # Original comparison
    if utility(pair[0])>=utility(pair[1]):
        correct_label = 1
    else:
        correct_label = -1
        
    if noise == 0.:
        #print('no noise, so we return the label directly')
        return correct_label

    else:
        #print(f"there is {noise} noise")
        
        # Introduce noise
        if random.random() < noise:
            # Flip the label with probability
            print('corruption!')
            return -correct_label
        else:
            #print('no corruption!')
            return correct_label
        
        
def pairwise_comparison_X_choice_noise(utility, multioutput_fun, pair,noise=0.):

    # Original comparison
    if utility(multioutput_fun(pair[0]))>=utility(multioutput_fun(pair[1])):
        label = 1
    else:
        label = -1
        
    return label
        

def pairwise_comparison_X_utility_noise(utility, multioutput_fun, pair,noise=0.):

    
    noise_temp = torch.normal(mean=0, std=noise, size=(1,))

    # Original comparison
    if utility(multioutput_fun(pair[0]))+noise_temp>=utility(multioutput_fun(pair[1])):
        correct_label = 1
    else:
        correct_label = -1
        
    if noise == 0.:
        print('no noise, so we return the label directly')
        return correct_label

    else:
        print(f"there is {noise} noise")
        
        # Introduce noise
        if random.random() < noise:
            # Flip the label with probability
            return -correct_label
        else:
            return correct_label


def pairwise_comparison_GP(pairwise_information,UtilityFunction,input_dim,output_dim,input='Y'):
    
  # This is my new version
  if input == 'Y':
    Y_flattened_tensors = [tensor.reshape(-1,) for tup in pairwise_information['Y'] for tensor in tup]
    Y_stacked_tensor = torch.stack(Y_flattened_tensors).view(-1, output_dim)
    
    util = UtilityFunction(Y_stacked_tensor)
    util = util.reshape(-1, 2)
    comps = torch.arange(util.numel()).reshape(-1, 2)
    
    prefer_tensor = torch.tensor(pairwise_information['prefer'])
    
    print(prefer_tensor)
    flip = (prefer_tensor == -1)
    
    print(flip)
    
    
    comps[flip, [0]], comps[flip, [1]] = comps[flip, [1]], comps[flip, [0]]
    
    return Y_stacked_tensor,comps
    
  elif  input == 'X':
    X_flattened_tensors = [tensor.reshape(-1,) for tup in pairwise_information['X'] for tensor in tup]
    X_stacked_tensor = torch.stack(X_flattened_tensors).view(-1, input_dim)
    
    Y_flattened_tensors = [tensor.reshape(-1,) for tup in pairwise_information['Y'] for tensor in tup]
    Y_stacked_tensor = torch.stack(Y_flattened_tensors).view(-1, output_dim)
    
    util = UtilityFunction(Y_stacked_tensor)
    util = util.reshape(-1, 2)
    comps = torch.arange(util.numel()).reshape(-1, 2)
    
    prefer_tensor = torch.tensor(pairwise_information['prefer'])
    
    print(prefer_tensor)
    flip = (prefer_tensor == -1)
    
    print(flip)
    
    
    comps[flip, [0]], comps[flip, [1]] = comps[flip, [1]], comps[flip, [0]]
    
    return X_stacked_tensor,comps

    



def generate_all_pairs(Y_BO, existing_comparisons):
    A, B = Y_BO.shape  # Get dimensions of Y_BO

    # Generate all possible unique pairs of indices
    all_indices = list(range(A))
    all_indices = list(combinations(all_indices, 2))

    all_pairs = [(tensor_to_tuple(Y_BO[pair[0]].reshape(-1,)),tensor_to_tuple(Y_BO[pair[1]].reshape(-1,))) for pair in all_indices]


    # Convert existing comparisons to tuples
    existing_comparisons_tuples = [(tensor_to_tuple(pair[0].reshape(-1,)), tensor_to_tuple(pair[1].reshape(-1,))) for pair in existing_comparisons]

  
    all_pairs = [
        pair for pair in all_pairs 
        if pair not in existing_comparisons_tuples 
        and (pair[1], pair[0]) not in existing_comparisons_tuples
        and torch.sum(torch.abs(torch.tensor(pair[0]) - torch.tensor(pair[1])) ) > 0.0
    ]
    

    # Check if there are any pairs left to sample
    if not all_pairs:
        raise ValueError("No more pairs available to sample.")


    return all_pairs



def find_all_diff_indices(list1, list2):
    return [i for i, (a, b) in enumerate(zip(list1, list2)) if a != b]



def create_hyper_rectangle_within_boundary(X_centre, X_boundary):
    # Compute the lengths L_i for each dimension
    L = X_boundary[1] - X_boundary[0]
    
    # Compute half-length of the hyper-rectangle
    half_length = L / 8
    
    # Compute the unrestricted lower and upper bounds
    lower_bound = X_centre - half_length
    upper_bound = X_centre + half_length
    
    # Clip the bounds to ensure they lie within X_boundary
    lower_bound = torch.maximum(lower_bound, X_boundary[0])
    upper_bound = torch.minimum(upper_bound, X_boundary[1])
    
    # Stack the bounds to create the hyper-rectangle representation
    hyper_rectangle = torch.stack([lower_bound, upper_bound], dim=0)
    return hyper_rectangle



def is_in_history(x, history, tolerance=1e-6):

    # Ensure x has shape (1, d)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Add batch dimension if missing
    
    # Compute differences between x and each row in history
    # Broadcasting will automatically expand x to match history's shape
    # Result has shape (N, d)
    diffs = torch.abs(history - x)
    
    # Sum differences across features (dimension 1)
    # Result has shape (N,)
    diffs_sum = diffs.sum(dim=1)
    
    # Check if any row is close enough to x (within tolerance)
    return bool(torch.any(diffs_sum < tolerance).item())