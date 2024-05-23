import numpy as np
import pandas as pd 
import statsmodels.api as smf



def fit_mixed_model(df, regressor_vars, outcome_var, rand_eff_var):
    # define formula, random effects formula
    formula    = (' + ').join(regressor_vars)
    re_formula = formula
    formula    = f'{outcome_var} ~ 1 + {formula}'
    # run model
    model_fit = smf.mixedlm(
        formula = formula, re_formula = re_formula,
        data = df, groups=df[rand_eff_var], missing='drop').fit()
    
#     model_bic = bic(model_fit)
    return model_fit

def bic(model_fit):
    K = len(model_fit.params)
    n = len(model_fit.resid)
    ll = model_fit.llf
    bic = (K*np.log(n)) - 2*ll
    
    return bic


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