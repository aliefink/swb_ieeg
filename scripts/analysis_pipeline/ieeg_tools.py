import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.mixed_linear_model import MixedLM 

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

import patsy
from statsmodels.api import OLS
from scipy import stats
from tqdm import tqdm
# from statsmodels.regression.mixed_linear_model import MixedLM 

### from salman connectivity functions!
# def find_nearest_value(array, value):
#     """Find nearest value and index of float in array
#     Parameters:
#     array : Array of values [1d array]
#     value : Value of interest [float]
#     Returns:
#     array[idx] : Nearest value [1d float]
#     idx : Nearest index [1d float]
#     """
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx], idx

def run_individual_elec_regression(data,reg_formula,elec_col,n_permutations,plot=False):
    results_dict = {}
    
    for elec in data[elec_col].unique().tolist():
        elec_data = data[data[elec_col] == elec]
        res = permutation_regression_zscore(elec_data,reg_formula,n_permutations=1000, plot_res=False)
        results_dict[elec] = res
        
    return results_dict
    

def single_elec_permutation_results(results_dict,data_df,save_vars):

    results_df = []
    
    # extract random effects param names and coefficient estimates from model
    unique_elec_ids = list(results_dict.keys())
    
    for elec in unique_elec_ids:
        info_dict = {'unique_elec_id':elec,
                     'subj_id':elec.split('_')[0],
                     'roi':data_df[data_df.unique_reref_ch == elec].roi.unique(),
                     'bdi':data_df[data_df.unique_reref_ch == elec].bdi.unique()}
        elec_results = results_dict[elec]
    
        keys = []
        vals = []    
        for var in elec_results.columns.tolist():
            elec_save_var_res = [(('_').join([var,svar]),elec_results[var][svar]) for svar in save_vars]
            keys.extend(list(zip(*elec_save_var_res))[0])
            vals.extend(list(zip(*elec_save_var_res))[1])
        
        elec_data = {f'{k}':v for k,v in list(zip(keys,vals))}
        
        results_df.append(pd.DataFrame({**info_dict,**elec_data}))
    
    return pd.concat(results_df).reset_index(drop=True) 


def mixed_eff_results_df(mixed_model_fit,data_df,rand_eff_var='unique_reref_ch',region_type='roi'):
    results_df = []
    
    # extract fixed effect param names and coefficient estimates from model 
    fe_params = [fe.split(')')[0] if fe.endswith(']') else fe # remove 'C()[]' for categorical vars
               for fe in mixed_model_fit.fe_params.index.values.tolist()]        
    fe_coeffs = mixed_model_fit.fe_params.values
    # join into dictionary 
    fe_dict = {f'{key}':val for key,val in list(zip(fe_params,fe_coeffs))}

    
    # extract random effects param names and coefficient estimates from model
    rand_vars = list(mixed_model_fit.random_effects.keys())
    
    
    for var in rand_vars:
        rand_params  = [p.split(')')[0] if p.endswith(']') else p # remove 'C()[]' for categorical vars
                        for p in mixed_model_fit.random_effects[var].index.values.tolist()]
        rand_params  = [p if p != 'Group' else 'Intercept' for p in rand_params]
        raw_params   =  ['_'.join(['raw',p]) for p in rand_params]
        raw_coeffs   = mixed_model_fit.random_effects[var].values.tolist()
        rand_coeffs  = mixed_model_fit.fe_params[rand_params].values + raw_coeffs
        
        rand_dict = {f'{key}':val for key,val in list(zip((raw_params+rand_params),
                                                          (list(raw_coeffs)+list(rand_coeffs))))}     

        info_dict = {f'{rand_eff_var}':var,
                                          'subj_id':var.split('_')[0],
                                          'region_type':data_df[data_df[rand_eff_var] == var][region_type].unique()[0],
                                          'bdi':data_df[data_df[rand_eff_var] == var].bdi.unique()[0]}
                     
        results_df.append(pd.DataFrame({**info_dict,**rand_dict},index=[0]) )            
    
#     results_df = pd.concat(results_df).reset_index(drop=True) 
    
#     for fe in fe_dict.keys():
#         results_df[fe] = fe_dict[fe]
    
#     return results_df
    return pd.concat(results_df).reset_index(drop=True) 


def fit_permuted_model(y_permuted, X):
    return OLS(y_permuted, X).fit().params

def permutation_regression_zscore(data, formula, n_permutations=1000, plot_res=False):
    # Perform original regression
    y, X = patsy.dmatrices(formula, data, return_type='dataframe')
    original_model = OLS(y, X).fit()

    # Extract original coefficients
    original_params = original_model.params

    # Prepare data for permutations
    y_values = y.values.ravel()
    X_values = X.values

    # Perform permutations
    permuted_params = []
    permuted_y_values = []
    for _ in tqdm(range(n_permutations), desc="Permutations"):
        y_permuted = np.random.permutation(y_values)
        permuted_params.append(fit_permuted_model(y_permuted, X_values))
        permuted_y_values.append(y_permuted)

    # Convert to numpy array for faster computations
    permuted_params = np.array(permuted_params)

    # Compute z-scores
    permuted_means = np.mean(permuted_params, axis=0)
    permuted_stds = np.std(permuted_params, axis=0)
    z_scores = (original_params - permuted_means) / permuted_stds

    # Compute p-values from z-scores
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    # Prepare results
    results = pd.DataFrame({
        'Original_Estimate': original_params,
        'Permuted_Mean': permuted_means,
        'Permuted_Std': permuted_stds,
        'Z_Score': z_scores,
        'P_Value': p_values
    })

    # Plotting
    if plot_res:
        features = [col for col in X.columns if col != 'Intercept']
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(3*n_features, 3*n_features), squeeze=False, dpi=300)

        for i, feature in enumerate(features):
            ax = axes[i, 0]

            # Plot permuted data first (in black)
            for j in range(min(100, n_permutations)):  # Limit to 100 permutations for clarity
                sns.regplot(x=X[feature], y=permuted_y_values[j], ax=ax, scatter=False,
                            line_kws={'color': 'black', 'alpha': 0.05}, ci=None)

            # Plot original data (in red)
            sns.regplot(x=X[feature], y=y_values, ax=ax, scatter_kws={'alpha': 0.5},
                        line_kws={'color': 'red', 'label': 'Original'}, ci=None)

            # Add z-score and p-value to the plot
            orig_param = original_params[i+1]
            z_score = z_scores[i+1]
            p_value = p_values[i+1]
            ax.text(0.05, 0.95, f'Beta: {orig_param:.2f}\nZ-score: {z_score:.2f}\np-value: {p_value:.3f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f'{feature} vs y')
            ax.legend()

            # Despine the plot
            sns.despine(ax=ax)

        plt.tight_layout()
        plt.show()

    return results

# Example usage:
# data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'category': ['A', 'B', 'A', 'B', ...]})
# formula = 'y ~ x1 + x2 + C(category)'
# results = permutation_regression_zscore(data, formula, plot_res=True)
# print(results)


def mixed_effects_ftest_ttest(stat_fit,stattest = 'both'):
    
    '''
    input to t_test is hypothesis = string of concatenated tuples of hypotheses to test (sig diff from zero?)
    # this only works for ols i think! - hypothesis = (',').join(['('+var+'=0)' for var in ttest_vars])

    '''
    # create identity matrix from fit.k_fe num fixed effects params
    if stattest == 'both':
        # run one sample t tests for all fixed effects regressors 
        R_ttest = np.identity(stat_fit.k_fe)[1:,:]
        ttest = stat_fit.t_test(R_ttest,use_t=True)
        ftest = stat_fit.f_test(np.identity(len(stat_fit.params))[1:,:])

    return (ttest.summary_frame(),ftest)
# https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.LogitResults.f_test.html#statsmodels.discrete.discrete_model.LogitResults.f_test

def fit_mixed_model(df, regressor_vars, random_vars, Y_var, RE_var,reml=True):
    # define formula, random effects formula
    re_formula = (' + ').join(random_vars) 
    formula    = f'{Y_var} ~ 1 +  {(" + ").join(regressor_vars)}'
    return smf.mixedlm(formula = formula, re_formula = re_formula, data = df, groups=df[RE_var], missing='drop').fit(reml=reml)

# def fit_mixed_model(df, regressor_vars, random_vars, outcome_var, rand_eff_var,reml=True):
#     # define formula, random effects formula
#     re_formula = (' + ').join(random_vars) 
#     formula    = f'{outcome_var} ~ 1 +  {(" + ").join(regressor_vars)}'
#     return smf.mixedlm(formula = formula, re_formula = re_formula, data = df, groups=df[rand_eff_var], missing='drop').fit(reml=reml)

def compute_marginal_rsq(model_fit):
    # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.2041-210x.2012.00261.x
    fe_var    = model_fit.params['Group Var']
    group_var = np.sum([model_fit.params[param] for param in model_fit.params.index.tolist() if param[-3:] == 'Var'])
    resid_var = np.var(model_fit.resid)
    rsq = fe_var/(group_var + resid_var)
    return rsq
    

def vif_scores(df, regressor_vars):
    
    cov_data_dict = {f'{reg}':[] for reg in regressor_vars}
    
    # check if data is categorical
    for reg in regressor_vars: 
        if pd.api.types.is_numeric_dtype(df[reg]):
            cov_data_dict[reg] = df[reg]
        else: 
            # factorize categorical data into numeric dummy variables 
            cov_data_dict[reg] = pd.factorize(df[reg])[0]
    
    vif_df = pd.DataFrame(cov_data_dict)


    vif_df = vif_df.astype(float)
    vif_df = vif_df.dropna()


    vif_data = pd.DataFrame() 
    vif_data["feature"] = vif_df.columns 

    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(vif_df.values, i) 
                              for i in range(len(vif_df.columns))] 
    return vif_data

def norm_zscore(reg_array):
    return (reg_array-np.nanmean(reg_array))/(2*np.nanstd(reg_array))

# def bic(lmm_fit):
#     K = len(lmm_fit.params-1) #number of fixed effects - intercept  
#     n = len(lmm_fit.resid)
#     ll = lmm_fit.llf
#     bic = (K*np.log(n)) - 2*ll
    
#     return bic


def compute_cpd(full_model_fit,reduced_model_fit):
    
    sse_full = np.sum(full_model_fit.resid**2)
    sse_reduced = np.sum(reduced_model_fit.resid**2)
    
    cpd = ((sse_reduced - sse_full)/sse_reduced)*100
    
    return cpd


#### generalized likelihood ratio test for null model vs cpe model

def GLRT(full_model, empty_model):
    
    chi_square = 2 * abs(full_model.llf - empty_model.llf)
    delta_params = abs(len(full_model.params) - len(empty_model.params)) 
    
    return {"chi_square" : chi_square, "df": delta_params, "p" : 1 - scipy.stats.chi2.cdf(chi_square, delta_params)}

#https://www.cl.uni-heidelberg.de/statnlpgroup/empirical_methods/kreutzer_significance-python.html
#https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture22.pdf
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html#scipy.stats.chi2
#https://nowak.ece.wisc.edu/ece830/ece830_fall11_lecture11.pdf


def run_parallel_permutation_regression(null_df, full_var_list, outcome_var, permute_var, rand_eff_var):
    
    # define formulas for full model 
    full_formula    = (' + ').join(full_var_list)
    full_re_formula = full_formula
    full_formula    = f'{outcome_var} ~ 1 + {full_formula}'
    
    # define formulas for reduced model 
    reduced_var_list   = full_var_list.copy()
    reduced_var_list.remove(permute_var)
    reduced_formula    = (' + ').join(reduced_var_list)
    reduced_re_formula = reduced_formula
    reduced_formula    = f'{outcome_var} ~ 1 + {reduced_formula}'
    
    
    # run full model on null data
    full_model_fit = smf.mixedlm(
        formula = full_formula, re_formula = full_re_formula,
        data = null_df, groups=null_df[rand_eff_var], missing='drop').fit()
    
    # run reduced model on null data
    reduced_model_fit = smf.mixedlm(
        formula = reduced_formula, re_formula = reduced_re_formula,
        data = null_df, groups=null_df[rand_eff_var], missing='drop').fit()
    
    null_cpd = compute_cpd(full_model_fit,reduced_model_fit)
    
    return null_cpd