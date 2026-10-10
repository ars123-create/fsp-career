if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    
    import numpy as np
    import torch
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.transforms import unnormalize,normalize
    from botorch.optim import optimize_acqf
    from botorch.optim.optimize import optimize_acqf_list
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.test_functions.multi_objective import DTLZ2, VehicleSafety,ZDT1
    from bope.problem import VLMOP3,NegativeVehicleSafety,KumaraswamyCDFProduct,AdaptedOSY,CarCabDesign
    from bope.problem import LinearUtility,ExponentialUtility,CobbDouglas,QuadraticUtility,LinearExponential
    from bope.utilis import (get_initial_points,get_random_points,pairwise_comparison_GP,generate_all_pairs,
                            get_corresponding_XY_pairs,generate_new_pair,pairwise_comparison_choice_noise,
                            pairwise_comparison_utility_noise,find_all_diff_indices,create_hyper_rectangle_within_boundary)
    from bope.model import (fit_outcome_model,fit_pref_model_GP,utility_model_training_noiseless,
                            scale_neural_network)
    from bope.acquisition import UtilityEI_MultiModel,Mean_MultiModel,UCB_MultiModel,qEUBO_MultiModel
    from bope.acquisition import qExpectedUtilityOfBestOption  
    import random
    from bope.variables import NUM_RESTARTS,RAW_SAMPLES,OPTIONS   
    import torch.multiprocessing as mp
    from bope.acquisition import DEUBO_NN,EUBO_NN
    from botorch.acquisition import LearnedObjective
    from bope.variables import NUM_RESTARTS,RAW_SAMPLES,OPTIONS 
    from collections import Counter
    from scipy.stats import mode
    
    import warnings
    from botorch.exceptions.warnings import BadInitialCandidatesWarning



    torch.set_default_dtype(torch.double)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    
    noise_type = 'utility_noise' 
    preference_noise = 0.1
     
    if preference_noise == 0.0:
        
        print('we have no noise')
        
        noise_type = 'no_noise'
        pairwise_comparison_function = pairwise_comparison_utility_noise 
        utility_model_training = utility_model_training_noiseless

    else:
        
        print('we have noise in pairwise comparison')
        
        if noise_type == 'choice_noise':
        
            pairwise_comparison_function = pairwise_comparison_choice_noise
            utility_model_training = utility_model_training_noiseless
            
    
        elif noise_type == 'utility_noise':
            
            pairwise_comparison_function = pairwise_comparison_utility_noise
            utility_model_training = utility_model_training_noiseless

    def worker_function(monotonic,seed, iter, output_dim, pairwise_information, result_list,result_list2, lock):
                    utility_model_temp,index = utility_model_training(n_features=output_dim, pairwise_information=pairwise_information,
                                                                monotonic=monotonic, seed=seed+iter)
                    
                    lock.acquire()
                    result_list.append(utility_model_temp)
                    result_list2.append(index)
                    lock.release()
        

    ########################### define the problem #############################################3
    problem_information = []

    temp={}
    temp['experiment_name'] = 'ZDT1_10d_LinearExponential' 
    temp['input_dim'] = 10
    temp['output_dim'] = 2
    temp['multioutput_fun'] = ZDT1(dim=temp['input_dim'],num_objectives=temp['output_dim'] ,negate=True) 
    temp['UtilityFunction'] = LinearExponential()  
    temp['exp_num'] = 50
    temp['best_U'] = 67.83067983438613
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'DTLZ2_LinearUtilityLowDimension' 
    temp['input_dim'] = 3
    temp['output_dim'] = 2
    temp['multioutput_fun'] = DTLZ2(dim=temp['input_dim'],num_objectives=temp['output_dim'] ,negate=True) 
    temp['UtilityFunction'] = LinearUtility(theta=torch.tensor([3.5,6.5]))  
    temp['exp_num'] = 30
    temp['best_U'] = -3.5
    problem_information.append(temp)
    

    temp={}
    temp['experiment_name'] = 'VLMOP3_ExponentialUtility' 
    temp['input_dim'] = 2
    temp['output_dim'] = 3
    temp['multioutput_fun'] = VLMOP3()
    temp['UtilityFunction'] = ExponentialUtility(theta=torch.tensor([0.35]))  
    temp['exp_num'] = 50
    temp['best_U'] = -181.1189064775784
    problem_information.append(temp)
    

    temp={}
    temp['experiment_name'] = 'OSY_Quadratic' 
    temp['input_dim'] = 6
    temp['output_dim'] = 8
    temp['multioutput_fun'] = AdaptedOSY(noise_std=0.) 
    temp['UtilityFunction'] = QuadraticUtility() 
    temp['exp_num'] = 50
    temp['best_U'] = 305.4552
    problem_information.append(temp)
    


    temp={}
    temp['experiment_name'] = 'VehicleSafety_KumaraswamyCDF'   
    temp['input_dim'] = 5
    temp['output_dim'] = 3
    temp['multioutput_fun'] = NegativeVehicleSafety(noise_std=0.)
    concentration1 = torch.tensor([0.5, 1, 1.5])
    concentration2 = torch.tensor([1.0, 2.0, 3.0])
    Y_bounds = torch.tensor(
                                [
                                    [0, 0, 0],
                                    [1, 1, 1],
                                ]
                            )
    temp['UtilityFunction'] = KumaraswamyCDFProduct(
                                concentration1=concentration1, concentration2=concentration2, Y_bounds=Y_bounds)
    temp['exp_num'] = 50
    temp['best_U'] = 0.898360063231304
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'CarCabDesign' 
    temp['input_dim'] = 7
    temp['output_dim'] = 9
    temp['multioutput_fun'] = CarCabDesign(noise_std=0.)
    temp['UtilityFunction'] = CobbDouglas(theta=torch.tensor([0.75,0.5,0.5,0.3,0.7,0.5,0.65,0.8,0.55])) 
    temp['exp_num'] = 50
    temp['best_U'] =  40.436247758433794
    problem_information.append(temp)
    #########################################################################################
    
    sample_num = 8
    normalisation = True # False
    monotonic = True # False
    
    if monotonic:
        if normalisation:
            sampling_policy_name = 'MNN'+str(sample_num)+'_normalisation'
        else:
            sampling_policy_name = 'MNN'+str(sample_num)+'_no_normalisation'
    else:
        if normalisation:
            sampling_policy_name = 'NN'+str(sample_num)+'_normalisation'
        else:
            sampling_policy_name = 'NN'+str(sample_num)+'_no_normalisation'
          
        
    for information in problem_information:
        
        # read the information
        experiment_name = information['experiment_name'] 
        input_dim = information['input_dim'] 
        output_dim = information['output_dim'] 
        multioutput_fun = information['multioutput_fun']
        UtilityFunction = information['UtilityFunction'] 
        exp_num = information['exp_num'] 
        best_U = information['best_U'] 
        
        if input_dim<=5:
            NUM_Y_INITIAL = 16
            NUM_PAIRS = 10
        else:
            NUM_Y_INITIAL = 32
            NUM_PAIRS = 20
            
        q_preference = 1
        q_design = 1
                       
        print('initial sample num: ',NUM_Y_INITIAL)
        print('initial pairwise num: ',NUM_PAIRS)
        
        
        if len(sys.argv) > 1:
            
            experiment_number = int(sys.argv[1])
            print('experiment id: ',experiment_number)
            filename = [str(preference_noise), experiment_name, sampling_policy_name, str(experiment_number)]
            
            print(filename)
            results_filename = '_'.join(filename)

            random.seed(1234+experiment_number)
            torch.manual_seed(1234+experiment_number)
            
            init_Y_BO_num = NUM_Y_INITIAL
            X_bounds = multioutput_fun.bounds
            X_BO = get_initial_points(X_bounds,init_Y_BO_num,device,dtype,seed=experiment_number)
            Y_BO = multioutput_fun(X_BO)
            
            X_BO_standard = normalize(X_BO,X_bounds)
            

            utility_record = UtilityFunction(Y_BO)
            print('the best utility is: ',utility_record.max()) #print the initial best utility
            
            best_record = [utility_record.max().item()]

            # Find the initial pairwise comparison in the Pareto Front
            X_pareto, Y_pareto = X_BO_standard[is_non_dominated(Y_BO)],Y_BO[is_non_dominated(Y_BO)]

            pairwise_information = {'X':[],'Y':[],'prefer':[]}
            num_pairs = NUM_PAIRS  # Specify the number of initial pairs you want
            results = get_corresponding_XY_pairs(X_pareto, Y_pareto, num_pairs)
            for element in results:
                # print('ele ',element[:2])
                pairwise_information['X'].append(element[:2])  # maybe it is not necessary? check whether to delete this or not!
                pairwise_information['Y'].append(element[2:])
 
                Y_distance = torch.sum(torch.abs( element[2:][0] -  element[2:][1] ))
                print('Y_distance',Y_distance)
                
                torch.manual_seed(1234+experiment_number)
                prefer_res = pairwise_comparison_function(UtilityFunction,element[2:],noise=preference_noise,seed=experiment_number)
                
                pairwise_information['prefer'].append(prefer_res)

            # experiment run
            train = True
            random_time = []
            
            print('Y size: ',len(pairwise_information['Y']))
            for iter in range(int(exp_num)):
                
                if preference_noise > 0.0:
                    
                    accuracy_filter_noise_record = []
            
                    X_pareto, Y_pareto = X_BO[is_non_dominated(Y_BO)],Y_BO[is_non_dominated(Y_BO)]
                    pairwise_information_noiseless = {'X':[],'Y':[],'prefer':[]}
                    
                    for y_pair in pairwise_information['Y']:

                        pairwise_information_noiseless['Y'].append(y_pair)
                        prefer_res = pairwise_comparison_function(UtilityFunction,y_pair,noise=0.,seed=experiment_number+iter)
                        pairwise_information_noiseless['prefer'].append(prefer_res)
                    
                    diff_list = find_all_diff_indices(pairwise_information_noiseless['prefer'],pairwise_information['prefer'] )
                    print('difference between noise and non-noise noise: ',diff_list)
                    
                    # #save the error index
                    if normalisation:
                        ErrorIndexList_filename = experiment_name+'_normalisation'+'_ErrorIndexList_'+ str(experiment_number)
                    else:
                        ErrorIndexList_filename = experiment_name+'_no_normalisation'+'_ErrorIndexList_'+ str(experiment_number)
            
                    diff_list.insert(0, -1) 
                  
                
                if train:

                    print('run index: ',iter)

                    random.seed(iter)
                    torch.manual_seed(iter)

                    # Build the output surrogate model
                    output_model = fit_outcome_model(X=X_BO, Y=Y_BO, X_bounds=X_bounds)
                    
                    # Build the utility surrogate model 
                    manager = mp.Manager()
                    result_list = manager.list()  # Shared list to collect results
                    result_list2 = manager.list()  # Shared list to collect results
                    lock = manager.Lock()  # Lock for synchronization
                    
                    processes = []
                    
                    for ii in range(sample_num):
                                
                        p = mp.Process(target=worker_function, args=(monotonic,ii+sample_num*0, 0, output_dim, 
                                                                pairwise_information, result_list,result_list2, lock))
                        p.start()
                        print('model start')
                        processes.append(p)
                                                                                  
                    for p in processes:
                        p.join()
                        
                    utility_model_list = list(result_list)  # Convert manager list to regular list
                    change_index_list = list(result_list2) 
                    
                    print('our change index list is: ',change_index_list)
                    
                    if preference_noise > 0.0:
                        counter = Counter(change_index_list)

                        # Find the mode by getting the element with the highest frequency
                        mode_data = counter.most_common(1)

                        # Get the mode value and count
                        index = mode_data[0][0]

                        if index<1000000:
                            print('my index is valid and my index is ',index)
                            pairwise_information['prefer'][index] = -  pairwise_information['prefer'][index] 
                            
                            #check whether my change is right or not
                            if  index in diff_list:
                                accuracy_filter_noise_record.append(1) # my filter is right
                            else:
                                accuracy_filter_noise_record.append(0) #my filter is wrong
                        else:
                            #check whether my change is right or not
                            if len(diff_list) == 0:
                                accuracy_filter_noise_record.append(1)
                            else:
                                accuracy_filter_noise_record.append(0)
                                
                        
                    # Now utility_model_list contains the trained models from each process
                    print(f"Trained {len(utility_model_list)} models simultaneously.")     
                    
                    utility_estimate = []
                    scale_utility_models_list = []
                    best_f_list = []
                    # sum = 0
                    for ii in range(sample_num):
                        
                        
                        if normalisation:
                        
                            model = utility_model_list[ii]
                            Y_flattened_tensors = [tensor.reshape(-1,) for tup in pairwise_information['Y'] for tensor in tup]
                            Y_stacked_tensor = torch.stack(Y_flattened_tensors).view(-1, output_dim)
                            Y_stacked_tensor = torch.unique(Y_stacked_tensor, dim=0).reshape(-1, output_dim) 
                            res = model(Y_stacked_tensor,Y_stacked_tensor)
                            mean_temp = res[0].max().item()
                            std_temp = res[0].std().item()
                            if std_temp == 0:
                                std_temp = (res[0].max().item()-res[0].min().item())
                            scale_model_temp = scale_neural_network(model,mean_temp,std_temp)
                            scale_utility_models_list.append(scale_model_temp)
                        
                        else:
                        
                            scale_model_temp = utility_model_list[ii]
                            scale_utility_models_list.append(scale_model_temp)


                        estimated_utility_max =  scale_model_temp(Y_BO,Y_BO)[0].max().item()
                        best_f_list.append(estimated_utility_max)
                        
                        utility_estimate.append(scale_model_temp(Y_BO,Y_BO)[0].reshape(-1,))
                        


                    print('best_f_list: ',best_f_list)
                    

                    acqf = UtilityEI_MultiModel(model=output_model,
                        utility_model_list=scale_utility_models_list,
                        best_f_list = best_f_list)
                

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", BadInitialCandidatesWarning)  # Only catch this specific warning

                        torch.manual_seed(1234+iter)
                        X_next, val = optimize_acqf(
                                acq_function=acqf,
                                bounds=X_bounds,
                                q=q_design,
                                num_restarts=NUM_RESTARTS,
                                raw_samples=RAW_SAMPLES,
                                options=OPTIONS)
                        
                        print('AF is: ',acqf(X_next).item())
                    
                        if any(issubclass(warning.category, BadInitialCandidatesWarning) for warning in w):
           
                            print("Caught a BadInitialCandidatesWarning! We do warm-up")
                            
                            beta = 0 
                            acqf_2 = UCB_MultiModel(model=output_model,
                                        utility_model_list=scale_utility_models_list,beta=beta)
                            
                            torch.manual_seed(43+iter)
                            X_warmup, _ = optimize_acqf(
                                    acq_function=acqf_2,
                                    bounds=X_bounds,
                                    q=q_design,
                                    num_restarts=NUM_RESTARTS,
                                    raw_samples=RAW_SAMPLES,
                                    options=OPTIONS)
                            

                            batch_initial_conditions = X_warmup.reshape(1, 1, input_dim) 

                            X_next, val = optimize_acqf(
                                acq_function=acqf,
                                bounds=X_bounds,
                                q=q_design,
                                num_restarts=1,
                                batch_initial_conditions=batch_initial_conditions,
                                options=OPTIONS)
                            
                            print('AF is: ',acqf(X_next).item())



                      
                        else:
                            print("No BadInitialCandidatesWarning was caught.")
                               


                    print('my next evaluation',X_next)

                    Y_next = multioutput_fun(X_next)

                    # update the observation
                    X_BO = torch.cat((X_BO, X_next))
                    Y_BO = torch.cat((Y_BO, Y_next))


                    X_BO_standard = normalize(X_BO,X_bounds)

                    utility_record = UtilityFunction(Y_BO)
                    print('the best utility is: ',utility_record.max())
                    best_record.append(utility_record.max().item())
                    
                    if best_U - utility_record.max()<10**(-5.):
                        train= False
                        
                    ############## preference choice (EUBO) ##########################################
                    X_pareto, Y_pareto = X_BO[is_non_dominated(Y_BO)],Y_BO[is_non_dominated(Y_BO)]

                    if iter<exp_num-1: # in the last step, we do not have to learn preference 
                        
                        print('DEUBO seraching for pairwise comparison')
                        
                        # If we only compare what is observed  
                        all_candidates = generate_all_pairs(Y_BO,pairwise_information['Y'])
                        Y_new_pair_list = DEUBO_NN(scale_utility_models_list,all_candidates,q_preference)
                        #Y_new_pair_list = EUBO_NN(scale_utility_models_list,all_candidates,q_preference)
                        
                        
                        # # if we can compare anything
                        # acqf = qEUBO_MultiModel(model=output_model,
                        #                         utility_model_list=scale_utility_models_list)                        
                        # try:

                        #     cand_X, acqf_val = optimize_acqf(
                        #         acq_function=acqf,
                        #         q=2*q_preference,
                        #         bounds=X_bounds,
                        #         num_restarts=NUM_RESTARTS,
                        #         raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                        #     )
                            
                        #     # print('no problem in EUBO opt')
                            
                        # except:
                        #     print('error in EUBO optimization')
                        #     cand_X = get_initial_points(X_bounds,2*q_preference,device,dtype,
                        #                                 seed=experiment_number+iter)
                        
                        
                        # posterior_X = output_model.posterior(cand_X)
                        # sampler_1 =  SobolQMCNormalSampler(sample_shape=torch.Size([1]))
                        # Y_samples = sampler_1(posterior_X).reshape(2*q_preference,output_dim)
                        
                        # Y_samples = Y_samples.detach()
                                             
                        # Y_new_pair_list = []
                        
                        # for ii in range(q_preference):
                        #     Y_new_pair = (Y_samples[2*ii].reshape(-1,),Y_samples[2*ii+1].reshape(-1,))
                        #     Y_new_pair_list.append(Y_new_pair)
                    
                        
                        # we ask the DM about his preference
                        random.seed(1234+experiment_number+iter)
                        torch.manual_seed(1234+experiment_number+iter)
            
                        for ii in range(q_preference):
                            
                            pair_temp = Y_new_pair_list[ii]
                            print('pair: ',pair_temp)
                            pairwise_information['Y'].append(pair_temp)  # compare the new observation and the past best observation
                            torch.manual_seed(1234+experiment_number+iter)
                            prefer_res = pairwise_comparison_function(UtilityFunction,pair_temp,noise=preference_noise,seed=experiment_number+iter)
                            pairwise_information['prefer'].append(prefer_res)
                                 


                    # record the utility
                    
                    utility_record = UtilityFunction(Y_BO)
                    try:
                        np.savetxt(noise_type+'/MNN_DEUBO/'+results_filename, utility_record.numpy())
                    except:
                        os.makedirs(noise_type+'/MNN_DEUBO/', exist_ok=True)

                             
                else:
                    print('very good performance, so we stop early!')
                    break
                               
        else:
            print('please give an index when running .py file!')
