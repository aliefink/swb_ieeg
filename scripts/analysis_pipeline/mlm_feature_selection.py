import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Shawn's idea: bootstrap the model with replacement
from IPython.display import clear_output
from tqdm import tqdm
import scipy as sp

def bootstrap_mixedlm(formula,
                     data,
                     re_formula,
                     group_key,
                     n_boot=500):
    """
    Wrapper for statsmodels mixedlm function that computed bootstrap corrected CI and p-values for mixedlm

    """

    def _resample_group(data, key='participant'):
        group = data[key].unique()

        # Resample group with replacement
        resampled_group = np.random.choice(group, size=len(group), replace=True)

        resampled_df = []
        for ix, group in enumerate(resampled_group):
            # filter down to group selected for sample
            temp_df = data[data[key]==group]
            # provide a unique ID for the resampled group
            temp_df[key] = ix
            # append the group to df
            resampled_df.append(temp_df)

        resampled_df = pd.concat(resampled_df)

        return resampled_df

    model = smf.mixedlm(formula=formula,
                       data=data,
                       re_formula=re_formula,
                       groups=data[group_key]).fit()

    boot_res = {f'{x}': [] for x in model.params.keys()}

    for i in tqdm(range(n_boot)):

        resampled_df = _resample_group(data, key=group_key)

        boot_model = smf.mixedlm(formula=formula,
                           data=resampled_df,
                           re_formula=re_formula,
                           groups=resampled_df[group_key]).fit()

        for key in boot_model.params.keys():
            boot_res[key].append(boot_model.params[key])

        clear_output(wait=True)

    b_dict = {}
    se_dict = {}
    z_dict = {}
    p_dict = {}
    CI_dict = {}
    for key in model.params.keys():
        new_beta = np.mean(boot_res[key])
        b_dict[key] = new_beta
        new_se = np.std(boot_res[key])
        se_dict[key] = new_se
        new_z = new_beta/new_se
        z_dict[key] = new_z
#         # two-tailed test on z
        new_p = sp.stats.norm.sf(abs(new_z))*2
        p_dict[key] = new_p
        CI_dict[key] = np.percentile(boot_res[key], [2.5, 97.5])


    return model, b_dict, se_dict, z_dict, p_dict, CI_dict