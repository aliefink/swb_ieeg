import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

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



def fit_mixed_model(df, regressor_vars, outcome_var, rand_eff_var,reml=True):
    # define formula, random effects formula
    formula    = (' + ').join(regressor_vars)
    re_formula = formula
    formula    = f'{outcome_var} ~ 1 + {formula}'
    # fit model
    return smf.mixedlm(formula = formula, re_formula = re_formula,
        data = df, groups=df[rand_eff_var], missing='drop').fit(reml=reml)

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
    return (reg_array-np.mean(reg_array))/(2*np.std(reg_array))

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