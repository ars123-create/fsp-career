import torch
from typing import Optional
from typing import Any, Optional, Union
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor
from botorch.acquisition import AcquisitionFunction, MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
import numpy as np
import heapq
from scipy.stats import norm

class UtilityEI(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        utility_model,
        sampler: Optional[MCSampler] = None,
    ) -> None:

        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        self.sampler = sampler

        self.utility_model = utility_model
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:

        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)  # n x b x q x o

        obj = self.utility_model(samples,samples)[0].squeeze(-1)
        obj =  (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)

        q_ei = obj.max(dim=-1)[0].mean(dim=0)

        return q_ei
    
    
class UtilityEI_MultiModel(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        utility_model_list,
        best_f_list,
        sampler: Optional[MCSampler] = None,
    ) -> None:

        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.sampler = sampler

        self.utility_model_list = utility_model_list
        self.best_f_list = best_f_list
   

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:


        m = len(self.utility_model_list)


        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)  # n x b x q x o

        utility_sample_list = []

        for ii in range(m):
          utility_model_temp = self.utility_model_list[ii]
          split_samples_temp = samples  #split_samples_list[ii]
          best_f_temp = torch.as_tensor(self.best_f_list[ii])
          obj_temp = utility_model_temp(split_samples_temp,split_samples_temp)[0].squeeze(-1)

          obj_temp = ( obj_temp - best_f_temp.unsqueeze(-1).to(obj_temp) ).clamp_min(0)
          utility_sample_list.append(obj_temp)

        obj = torch.cat(utility_sample_list, dim=0)

        q_ei = obj[:,:,-1].mean(dim=0) 

        return q_ei
    
    
class UCB_MultiModel(MCAcquisitionFunction):
   def __init__(
       self,
       model: Model,
       utility_model_list,
       beta,
       sampler: Optional[MCSampler] = None,
   ) -> None:

       super(MCAcquisitionFunction, self).__init__(model=model)
       if sampler is None:
           sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
       self.sampler = sampler

       self.utility_model_list = utility_model_list
       
       self.beta = beta #1. #np.random.uniform(0.,4.,size=None)
  

   @t_batch_mode_transform()
   def forward(self, X: Tensor) -> Tensor:

       # N = self.sampler.sample_shape[0]
       m = len(self.utility_model_list)
      

       posterior = self.model.posterior(X)
       samples = self.get_posterior_samples(posterior)  # n x b x q x o

       utility_sample_list = []

       for ii in range(m):
         utility_model_temp = self.utility_model_list[ii]
         split_samples_temp = samples  
         obj_temp = utility_model_temp(split_samples_temp,split_samples_temp)[0].squeeze(-1)

         obj_temp = obj_temp 
         utility_sample_list.append(obj_temp)

       obj = torch.cat(utility_sample_list, dim=0)

       std_temp = obj[:,:,-1].std(dim=0).reshape(-1,)   #obj.std(dim=0).reshape(-1,)
       mean_temp = obj[:,:,-1].mean(dim=0)   #obj.max(dim=-1)[0].mean(dim=0)

  
       return mean_temp + self.beta*std_temp
   
    
class Mean_MultiModel(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        utility_model_list,
        sampler: Optional[MCSampler] = None,
    ) -> None:

        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        self.sampler = sampler

        self.utility_model_list = utility_model_list
   

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:

        m = len(self.utility_model_list)


        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)  # n x b x q x o



        utility_sample_list = []

        for ii in range(m):
          utility_model_temp = self.utility_model_list[ii]
          split_samples_temp = samples  
          obj_temp = utility_model_temp(split_samples_temp,split_samples_temp)[0].squeeze(-1)

          obj_temp = obj_temp 
          utility_sample_list.append(obj_temp)

        obj = torch.cat(utility_sample_list, dim=0)

        val = obj[:,:,-1].mean(dim=0)  #obj.max(dim=-1)[0].mean(dim=0)

        return val
    
    
    

# This is qEUBO (for GP utility model) code from https://github.com/RaulAstudillo06/qEUBO
class qExpectedUtilityOfBestOption(MCAcquisitionFunction):


    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_baseline: Optional[Tensor] = None,
    ) -> None:

        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            X_pending=X_baseline)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:

        posterior_X = self.model.posterior(X)
        Y_samples = self.sampler(posterior_X)
        util_val_samples = self.objective(Y_samples)
        best_util_val_samples = util_val_samples.max(dim=-1).values
        exp_best_util_val = best_util_val_samples.mean(dim=0)
        return exp_best_util_val
    
    
    
class qExpectedUtilityOfBestOption_PBO(MCAcquisitionFunction):


    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_baseline: Optional[Tensor] = None,
    ) -> None:

        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            X_pending=X_baseline,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:

        posterior_X = self.model.posterior(X)
        Y_samples = self.sampler(posterior_X)
        util_val_samples = self.objective(Y_samples)
        best_util_val_samples = util_val_samples.max(dim=-1).values
        exp_best_util_val = best_util_val_samples.mean(dim=0)
        return exp_best_util_val
    
    


# This is qEUBO (for neural network utility model) 
class qEUBO_MultiModel(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        utility_model_list,
        sampler: Optional[MCSampler] = None,
    ) -> None:

        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]))
        self.sampler = sampler

        self.utility_model_list = utility_model_list
   

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:

        m = len(self.utility_model_list)


        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)  # n x b x q x o
        
    

        utility_sample_list = []

        for ii in range(m):
          utility_model_temp = self.utility_model_list[ii]
          obj_temp = utility_model_temp(samples,samples)[0].squeeze(-1)
          utility_sample_list.append(obj_temp)


        obj = torch.cat(utility_sample_list, dim=0)
        

        val = obj.max(dim=-1)[0].mean(dim=0)
        

        return val
    
    
    
    


def monte_carlo_max_expectation(mu1, sigma1, mu2, sigma2, n_samples=100000):
    x1_samples = np.random.normal(mu1, sigma1, n_samples)
    x2_samples = np.random.normal(mu2, sigma2, n_samples)
    max_samples = np.maximum(x1_samples, x2_samples)
    return np.mean(max_samples)


def analytical_max_expectation(mu1, sigma1, mu2, sigma2):
    
    # Combined sigma
    sigma = np.sqrt(sigma1**2 + sigma2**2)
    # Standardized difference
    if sigma == 0:
        alpha = (mu1 - mu2) 
    else: 
        alpha = (mu1 - mu2) / sigma

    # Components of the formula
    term1 = mu1 * norm.cdf(alpha)
    term2 = mu2 * norm.cdf(-alpha)
    term3 = sigma * norm.pdf(alpha)

    return term1 + term2 + term3



def top_k_indices_above_threshold(lst, k, threshold=0.01):
    # Find the top K elements with their indices
    top_k = heapq.nlargest(k, [(value, i) for i, value in enumerate(lst)])
    
    # Filter out elements smaller than the threshold and return their indices
    filtered_indices = [index for value, index in top_k if value >= threshold]
    
    return filtered_indices


def DEUBO_NN(NN_list,all_pairs,k=1):

  val_list = []

  for pair in all_pairs:

    pair0 = torch.tensor(pair[0])
    pair1 = torch.tensor(pair[1])

    val0_list = []
    val1_list = []

    for model in NN_list:
      val0 = model(pair0,pair0)[0]
      val1 = model(pair1,pair1)[0]
      
      val0_list.append(val0.item())
      val1_list.append(val1.item())
    
    mu0 = np.mean(val0_list)
    sigma0 = np.std(val0_list)
    mu1 = np.mean(val1_list)
    sigma1 = np.std(val1_list)

    val = analytical_max_expectation(mu0,sigma0,mu1,sigma1) 
    val_list.append(val)

  
  idx_list = top_k_indices_above_threshold(val_list, k, threshold=-1000000)
  res_list = []
  for idx in idx_list:
      res = (torch.tensor(all_pairs[idx][0]),torch.tensor(all_pairs[idx][1]) )
      res_list.append(res)


  return res_list 




def DEUBO_GP(model,all_pairs,k=1):

  val_list = []

  for pair in all_pairs:

    pair0 = torch.tensor(pair[0])
    pair1 = torch.tensor(pair[1])

    prediction0 = model(pair0)
    mu0 = prediction0.mean.detach().numpy()
    sigma0 = np.sqrt(prediction0.variance.detach().numpy())
    
    prediction1 = model(pair1)
    mu1 = prediction1.mean.detach().numpy()
    sigma1 = np.sqrt(prediction1.variance.detach().numpy())


    val = analytical_max_expectation(mu0,sigma0,mu1,sigma1) 
    val_list.append(val)

  idx_list = top_k_indices_above_threshold(val_list, k, threshold=-1000000)
  res_list = []
  for idx in idx_list:
      res = (torch.tensor(all_pairs[idx][0]),torch.tensor(all_pairs[idx][1]) )
      res_list.append(res)


  
  return res_list 



###############################################################
def EUBO_NN(NN_list,all_pairs,k=1):

  val_list = []

  for pair in all_pairs:

    pair0 = torch.tensor(pair[0])
    pair1 = torch.tensor(pair[1])

    val0_list = []
    val1_list = []

    for model in NN_list:
      val0 = model(pair0,pair0)[0]
      val1 = model(pair1,pair1)[0]
      
      val0_list.append(val0.item())
      val1_list.append(val1.item())
    

    val = np.mean([max(a, b) for a, b in zip(val0_list, val1_list)])
    val_list.append(val)
  
  idx_list = top_k_indices_above_threshold(val_list, k, threshold=-1000000)
  res_list = []
  for idx in idx_list:
      res = (torch.tensor(all_pairs[idx][0]),torch.tensor(all_pairs[idx][1]) )
      res_list.append(res)

  
  return res_list 






def EUBO_observed_GP(model,all_pairs,k=1):

  val_list = []

  for pair in all_pairs:

    pair = torch.tensor(pair)
    
    # list_temp = []
    
    posterior = model.posterior(pair)
    samples = posterior.rsample(torch.Size([32])).reshape(32,2)
        
    max_calculation = torch.max(samples, dim=1).values
    
    val = torch.mean(max_calculation).item()

    val_list.append(val)

  idx_list = top_k_indices_above_threshold(val_list, k, threshold=-1000000)
  res_list = []
  for idx in idx_list:
      res = (torch.tensor(all_pairs[idx][0]),torch.tensor(all_pairs[idx][1]) )
      res_list.append(res)


  
  return res_list
    
    
    
    
    
    