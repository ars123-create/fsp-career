import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models import KroneckerMultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import Interval
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
import torch.nn.functional as F
from bope.acquisition import top_k_indices_above_threshold
import torch.distributions as dist
from torch.distributions import Normal, kl_divergence
import math
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.sigmoid(x) if self.inplace else x * torch.sigmoid_(x)

# Define a neural network for pairwise ranking with positive weights and biases
class PairwiseRankingNet(nn.Module):
    def __init__(self, input_size,monotoic=True):
        super(PairwiseRankingNet, self).__init__()

        if monotoic:

          self.fc1 = PositiveLinear(input_size, 100)
          self.fc2 = PositiveLinear(100, 10)
          self.fc3 = PositiveLinear(10, 1)
        
        if not monotoic:
            
            self.fc1 = nn.Linear(input_size, 100)
            self.fc2 = nn.Linear(100, 10)
            self.fc3 = nn.Linear(10, 1)

        self.af = Swish() 
        self.af2 =  Swish() 



    def forward(self, x1, x2):

        x1 = self.af(self.fc1(x1))
        x1 = self.af2(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = self.af(self.fc1(x2))
        x2 = self.af2(self.fc2(x2))
        x2 = self.fc3(x2)
           

        return x1, x2

softplus = lambda x: torch.log(1 + torch.exp(x))

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. /self.weight.size(1)
        self.weight.data.uniform_(-stdv-6, stdv) 
        self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        weight = torch.exp(self.weight) 
        bias = self.bias

        return input @ weight.t() + bias




# Define a neural network for pairwise ranking with positive weights and biases
class BayesianPairwiseRankingNet(nn.Module):
    def __init__(self, input_size,seed):
        super(BayesianPairwiseRankingNet, self).__init__()

        self.fc1 = BayesianMontonicLinear(input_size, 100,seed)
        self.fc2 = BayesianMontonicLinear(100, 10,seed)
        self.fc3 = BayesianMontonicLinear(10, 1,seed)

        self.af = Swish() 
        self.af2 = Swish() 

    def forward(self, x1, x2):

        x1 = self.af(self.fc1(x1))
        x1 = self.af2(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = self.af(self.fc1(x2))
        x2 = self.af2(self.fc2(x2))
        x2 = self.fc3(x2)

        return x1, x2

    def KL_divergence(self):
        return self.fc1.KL_divergence() + self.fc2.KL_divergence() + self.fc3.KL_divergence()
    


    
class BayesianMontonicLinear(nn.Module):
    def __init__(self, in_features, out_features,seed):
        super(BayesianMontonicLinear, self).__init__()

        init_std = 1.0 / math.sqrt(in_features)
        self.init_std = init_std
        # Mean and log variance for weights and biases
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, init_std))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-5, 1))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, init_std))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_features).normal_(-5, 1))
        
        self.weight_mu_init = copy.deepcopy(self.weight_mu)
        self.weight_log_var_init =  copy.deepcopy(self.weight_log_var)

        self.bias_mu_init =  copy.deepcopy(self.bias_mu)
        self.bias_log_var_init =  copy.deepcopy(self.bias_log_var)



    def forward(self, input):

        weight = torch.exp(self.weight_mu + torch.exp(self.weight_log_var / 2) * torch.randn_like(self.weight_mu))

        bias = self.bias_mu + torch.exp(self.bias_log_var / 2) * torch.randn_like(self.bias_mu)

        return input @ weight.t() + bias

    def KL_divergence(self):

        prior_weight = Normal(self.weight_mu_init.detach(), torch.exp(self.weight_log_var_init.detach() / 2))
        prior_bias = Normal(self.bias_mu_init.detach(), torch.exp(self.bias_log_var_init.detach() / 2))

        q_weight = Normal(self.weight_mu, torch.exp(self.weight_log_var / 2))
        q_bias = Normal(self.bias_mu, torch.exp(self.bias_log_var / 2))
        kl_weight = kl_divergence(q_weight, prior_weight).sum()
        kl_bias = kl_divergence(q_bias, prior_bias).sum()
        return kl_weight + kl_bias
    

    
def sample_bnn_to_dnn(bnn_model, input_size, num_samples=1,seed=0):
    
    torch.manual_seed(seed)
    
    networks = []

    for _ in range(num_samples):
        # Initialize a new deterministic network
        dnn_model = PairwiseRankingNet(input_size, monotoic=True)

        # Sample weights and biases from the BNN's posterior distributions
        with torch.no_grad():
            dnn_model.fc1.weight.copy_(
                (bnn_model.fc1.weight_mu + torch.exp(bnn_model.fc1.weight_log_var / 2) * torch.randn_like(bnn_model.fc1.weight_mu))
            )
            dnn_model.fc1.bias.copy_(
                bnn_model.fc1.bias_mu + torch.exp(bnn_model.fc1.bias_log_var / 2) * torch.randn_like(bnn_model.fc1.bias_mu)
            )

            dnn_model.fc2.weight.copy_(
                (bnn_model.fc2.weight_mu + torch.exp(bnn_model.fc2.weight_log_var / 2) * torch.randn_like(bnn_model.fc2.weight_mu))
            )
            dnn_model.fc2.bias.copy_(
                bnn_model.fc2.bias_mu + torch.exp(bnn_model.fc2.bias_log_var / 2) * torch.randn_like(bnn_model.fc2.bias_mu)
            )

            dnn_model.fc3.weight.copy_(
                (bnn_model.fc3.weight_mu + torch.exp(bnn_model.fc3.weight_log_var / 2) * torch.randn_like(bnn_model.fc3.weight_mu))
            )
            dnn_model.fc3.bias.copy_(
                bnn_model.fc3.bias_mu + torch.exp(bnn_model.fc3.bias_log_var / 2) * torch.randn_like(bnn_model.fc3.bias_mu)
            )

        # Add the sampled deterministic model to the list
        networks.append(dnn_model)

    return networks
    

    
################################## Loss Function ##############################################
# Pairwise hinge loss
def pairwise_hinge_loss(scores1, scores2, labels):
    loss = torch.maximum(torch.tensor(0.0), 1.0 - 100*(scores1 - scores2) * labels)
    return loss.mean()


class ParametricLoss_Hinge(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_unconstrained = nn.Parameter(torch.tensor(5.0, requires_grad=True))
       
    def forward(self, scores1, scores2, labels):
        
        zero_tensor = torch.tensor(0.0)

        alpha = torch.exp(self.alpha_unconstrained) 
        
        hinge_loss = torch.max(zero_tensor, 1.0 - alpha * (scores1 - scores2) * labels)
       
        return hinge_loss.mean()



    
##################################################################################################
def fit_outcome_model(X: torch.Tensor, Y: torch.Tensor, X_bounds: torch.Tensor) -> SingleTaskGP:
    """Fit the outcome model f"""

    input_tf = Normalize(d=X.shape[-1], bounds=X_bounds)
    outcome_model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        outcome_transform=Standardize(m=Y.shape[-1]),
        input_transform=input_tf)

    outcome_model.likelihood.noise_covar.register_constraint("raw_noise", Interval(1e-6,1e-4)) # we can remove the noise constraint
    mll = ExactMarginalLogLikelihood(outcome_model.likelihood, outcome_model) .to(device)

    try:
      fit_gpytorch_mll(mll)
      print('outcome model trained')
    except:
      print('problem with model training')
      pass

    return outcome_model



######################################################################################################
def utility_model_training_noiseless(n_features,pairwise_information,monotonic=True,seed=0,Bayesian=False):
    
  torch.manual_seed(seed)
  
  big_index = 1000000

  if Bayesian:
    model = BayesianPairwiseRankingNet(n_features,monotonic)
  else:
    model = PairwiseRankingNet(n_features,monotonic)
  
  criterion = ParametricLoss_Hinge()
  
  optimizer = torch.optim.Adam(list(model.parameters()) + [criterion.alpha_unconstrained], lr=0.01)
  #optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
  
  # Initialize the CosineAnnealingLR scheduler
  num_epochs = 1600 
  scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

  best_loss = float('inf')  # Track the best loss





  for epoch in range(num_epochs):
      epoch_loss = 0.0
      for i in range(len(pairwise_information['Y'])):
          Y_pair = pairwise_information['Y'][i]
          Y_0 = Y_pair[0]
          Y_1 = Y_pair[1]
        
          
          if (torch.all(Y_pair[0] >= Y_pair[1]) or torch.all(Y_pair[1] >= Y_pair[0])):
                continue
            
          l1_distance = torch.sum(torch.abs(Y_pair[0] - Y_pair[1]))
          threshold = 0.05 # 0.15 if Y_0.shape[0]<=5 else 0.05
          if l1_distance< threshold: 
              continue
          
          label =  torch.tensor(pairwise_information['prefer'][i])
          score0, score1 = model(Y_0, Y_1)
          
          if Bayesian:
            kl_div = model.KL_divergence()
            loss = criterion(score0, score1, label) + 1e-5*kl_div 
          else:
            loss = criterion(score0, score1, label) 
          
          epoch_loss += loss.item()

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      # Update the learning rate scheduler
      scheduler.step()

      epoch_loss /= len(pairwise_information['Y']) # Calculate the average loss for the epoch

      if epoch_loss < best_loss:
          best_loss = epoch_loss

      if (epoch + 1) % 10 == 0:
          print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        #   print(f"Epoch [{epoch + 1}/{num_epochs}], KL Loss: {kl_div:.4f}")

      if epoch_loss==0.0:
          print(f"Training stopped early at epoch {epoch + 1} due to low loss ({epoch_loss:.4f})")
          break
      
  print('alpha is: ',torch.exp(criterion.alpha_unconstrained))
  print('utility model trained')

  return model,big_index





class scale_neural_network:
  def __init__(self,model,mean,std):
    self.model = model
    self.mean = mean
    self.std = std
  def __call__(self, X1,X2):
    res = self.model(X1,X2)

    return (res[0]-self.mean)/self.std, (res[1]-self.mean)/self.std 


def fit_pref_model_GP(Y: torch.Tensor, comps: torch.Tensor) -> PairwiseGP:
    """Fit the preference model g."""

    model = PairwiseGP(Y, comps, input_transform=Normalize(d=Y.shape[-1]))
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    try:
      fit_gpytorch_mll(mll)
      print('utility model trained!!')
    except:
      print('problem with utility model training!!!')
      pass

    return model





class LinearModel(nn.Module):
    def __init__(self, weights,output_size=2):
        super(LinearModel, self).__init__()

        self.fc = nn.Linear(output_size, 1)
        # Define custom weights and biases
        custom_weights = weights.reshape(1, output_size) 

        
        # Assign custom parameters
        with torch.no_grad():  
            self.fc.weight.copy_(custom_weights)
            self.fc.weight.requires_grad_() 


    def forward(self, x1, x2):

        x1 = self.fc(x1)
        x2 = self.fc(x2)

        return x1, x2
    
    

class LaplaceBayesianPairwiseComparison:
    def __init__(self, prior_mean=0, prior_var=2.0):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.phi_weights = None  # MAP estimate
        self.hessian_inv = None  # Inverse Hessian at MAP (posterior covariance)

    def negative_log_posterior(self, phi_weights, delta, X1, X2, y, noise_var=1.0):

        # Apply exp to phi_weights to get positive weights, and add delta
        weights = torch.exp(phi_weights) + delta

        # Pairwise difference
        f_diff = (torch.matmul(X1, weights) - torch.matmul(X2, weights))

        # Probit likelihood (standard normal CDF)
        normal = Normal(0, 1)
        likelihood_term = torch.sum(-torch.log(normal.cdf(y * f_diff)))  # CDF for the probit model

        # Negative log prior (Gaussian prior on weights)
        prior_term = 0.5 * torch.sum((weights - self.prior_mean) ** 2) / self.prior_var

        # Jacobian term for the transformation (exp(phi_weights))
        jacobian = torch.sum(torch.exp(phi_weights))

        return likelihood_term + prior_term + jacobian

    def hessian(self, weights, X1, X2, y, noise_var=1.0):

        # Compute the second derivative of the log-likelihood
        X_diff = X1 - X2
        hessian_likelihood = torch.matmul(X_diff.T, X_diff) / noise_var

        # Compute the second derivative of the log-prior
        hessian_prior = torch.eye(X1.shape[1]) / self.prior_var

        # Total Hessian
        hessian = hessian_likelihood + hessian_prior
        return hessian

    def fit(self, X1, X2, y, noise_var=1.0):

        # Initial guess for weights and delta
        initial_phi_weights = torch.zeros(X1.shape[1], dtype=torch.float64, requires_grad=True)
        # initial_delta = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        initial_delta = torch.tensor(.1, dtype=torch.float64, requires_grad=True)

        # Create the optimizer (e.g., Adam)
        optimizer = torch.optim.Adam([initial_phi_weights], lr=0.01)
        num_epochs = 1600 
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

        for _ in range(num_epochs):  # Optimization loop
            optimizer.zero_grad()  # Clear previous gradients

            delta_clamped = torch.clamp(initial_delta, min=0.0)  # Clamping delta to avoid negative values

            # Compute negative log posterior and gradients
            nll = self.negative_log_posterior(initial_phi_weights, delta_clamped, X1, X2, y, noise_var)
            nll.backward()  # Compute gradients of the negative log posterior

            # Update weights and delta using Adam optimizer
            optimizer.step()
            scheduler.step()

        self.phi_weights_map = initial_phi_weights.detach()  # Detach from computation graph
        self.delta_map = initial_delta.detach()  # Detach from computation graph

        # Recalculate weights in the original space (before transformation)
        self.weights_map = torch.exp(self.phi_weights_map) + self.delta_map
        self.hessian_inv = torch.inverse(self.hessian(self.weights_map, X1, X2, y, noise_var))

        return self

    def predict(self, X1_new, X2_new):
 
        # Predict the difference in outputs
        y_pred = torch.matmul(X1_new, self.weights_map) - torch.matmul(X2_new, self.weights_map)
        return y_pred
    
    

class LaplaceShiftLogNormalBayesianPairwiseComparison:
    def __init__(self, prior_mean=0, prior_var=2.0):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.phi_weights = None  # MAP estimate
        self.hessian_inv = None  # Inverse Hessian at MAP (posterior covariance)

    def negative_log_posterior(self, phi_weights, delta, X1, X2, y, noise_var=1.0):

        # Apply exp to phi_weights to get positive weights, and add delta
        weights = torch.exp(phi_weights) + delta

        # Pairwise difference
        f_diff = (torch.matmul(X1, weights) - torch.matmul(X2, weights))

        # Probit likelihood (standard normal CDF)
        normal = Normal(0, 1)
        likelihood_term = torch.sum(-torch.log(normal.cdf(y * f_diff)))  # CDF for the probit model

        # Negative log prior (Gaussian prior on weights)
        prior_term = 0.5 * torch.sum((weights - self.prior_mean) ** 2) / self.prior_var

        # Jacobian term for the transformation (exp(phi_weights))
        jacobian = torch.sum(torch.exp(phi_weights))

        return likelihood_term + prior_term + jacobian

    def hessian(self, weights, X1, X2, y, noise_var=1.0):

        # Compute the second derivative of the log-likelihood
        X_diff = X1 - X2
        hessian_likelihood = torch.matmul(X_diff.T, X_diff) / noise_var

        # Compute the second derivative of the log-prior
        hessian_prior = torch.eye(X1.shape[1]) / self.prior_var

        # Total Hessian
        hessian = hessian_likelihood + hessian_prior
        return hessian

    def fit(self, X1, X2, y, noise_var=1.0):

        # Initial guess for weights and delta
        initial_phi_weights = torch.zeros(X1.shape[1], dtype=torch.float64, requires_grad=True)
        initial_delta = torch.tensor(0, dtype=torch.float64, requires_grad=True)

        # Create the optimizer (e.g., Adam)
        optimizer = torch.optim.Adam([initial_phi_weights, initial_delta], lr=0.01)
        num_epochs = 1600 
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)
  

        for _ in range(num_epochs):  # Optimization loop
            optimizer.zero_grad()  # Clear previous gradients

            delta_clamped = torch.clamp(initial_delta, min=0.0)  # Clamping delta to avoid negative values

            # Compute negative log posterior and gradients
            nll = self.negative_log_posterior(initial_phi_weights, delta_clamped, X1, X2, y, noise_var)
            nll.backward()  # Compute gradients of the negative log posterior

            # Update weights and delta using Adam optimizer
            optimizer.step()
            scheduler.step()

        self.phi_weights_map = initial_phi_weights.detach()  # Detach from computation graph
        self.delta_map = initial_delta.detach()  # Detach from computation graph

        # Recalculate weights in the original space (before transformation)
        self.weights_map = torch.exp(self.phi_weights_map) + self.delta_map
        self.hessian_inv = torch.inverse(self.hessian(self.weights_map, X1, X2, y, noise_var))

        return self

    def predict(self, X1_new, X2_new):

        # Predict the difference in outputs
        y_pred = torch.matmul(X1_new, self.weights_map) - torch.matmul(X2_new, self.weights_map)
        return y_pred