import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.interpolate import interp1d

base_dir = 'fill this in'
experiment_subdir = "data/experiment"
rf_experiments = [] # Add in the experiment names here

def interpolate_tpr_fpr(tpr, fpr):
    # Adds (0,0) and (1,1) to the roc curve.
    tpr = tpr.copy()
    fpr = fpr.copy()
    assert len(tpr) == len(fpr)
    l = len(tpr)
    tpr[l] = 0.
    fpr[l] = 0.
    tpr[l+1] = 1.
    fpr[l+1] = 1.
    return tpr, fpr

def interpolate_tpr_ppv(tpr, ppv):
    # Adds (0,0) and (1,1) to the roc curve.
    tpr = tpr.copy()
    ppv = ppv.copy()
    assert len(tpr) == len(ppv)
    l = len(tpr)
    tpr[l] = 1.
    ppv[l] = 0.
    tpr[l+1] = 0.
    ppv[l+1] = 1.

    return tpr, ppv

def interpolate_and_integrate(dep, ind, give_me_everything=False, interpolator=interpolate_tpr_fpr):
    dep, ind = interpolator(dep, ind)
    x = np.linspace(0, 1, 100)
    f = interp1d(ind, dep)
    y = f(x)
    if give_me_everything:
        return np.trapz(y, x), x, y
    else:
        return np.trapz(y, x)

cnn_experiments = [] # Add in the experiment names here
dnn_experiments = [] # Add in the experiment names here
diseases = ['RA', 'healthy', 'IGT', 'CRC', 'otitis', 'fatty_liver', 'T1D', 'psoriasis', 'HBV', 'T2D', 'IBD', 'adenoma', 'CDI', 'hypertension', 'AD', 'periodontitis', 'metabolic_syndrome', 'bronchitis', 'infectiousgastroenteritis']
cnn_metrics_dirs = {
    disease: [
        os.path.join(base_dir, experiment_subdir, experiment_name, "{0}_cnn_{1}_metrics.csv".format(i, disease))
        for i in range(5)
        for experiment_name in cnn_experiments
        ]
    for disease in diseases
}
dnn_metrics_dirs = {
    disease: [
        os.path.join(base_dir, experiment_subdir, experiment_name, "{0}_dnn_{1}_metrics.csv".format(i, disease))
        for i in range(5)
        for experiment_name in dnn_experiments
        ]
    for disease in diseases
}

cnn_metrics = {disease: [pd.read_csv(f) for f in dirs] for disease, dirs in cnn_metrics_dirs.items()}
dnn_metrics = {disease: [pd.read_csv(f) for f in dirs] for disease, dirs in dnn_metrics_dirs.items()}

cnn_auc = {
    disease: [
        interpolate_and_integrate(df['sensitivity'], 1-df['specificity'])
        for df in values
    ] 
        for disease, values in cnn_metrics.items()
    }
cnn_auc = pd.DataFrame(cnn_auc)
cnn_aupr = {
    disease: [
        interpolate_and_integrate(df['sensitivity'], df['ppv'].fillna(0), interpolator=interpolate_tpr_ppv)
        for df in values
    ] 
        for disease, values in cnn_metrics.items()
    }
cnn_aupr = pd.DataFrame(cnn_aupr)

dnn_auc = {
    disease: [
        interpolate_and_integrate(df['sensitivity'], 1-df['specificity'])
        for df in values
    ] 
        for disease, values in dnn_metrics.items()
    }
dnn_auc = pd.DataFrame(dnn_auc)
dnn_aupr = {
    disease: [
        interpolate_and_integrate(df['sensitivity'], df['ppv'].fillna(0), interpolator=interpolate_tpr_ppv)
        for df in values
    ] 
        for disease, values in dnn_metrics.items()
    }
dnn_aupr = pd.DataFrame(dnn_aupr)

rf_auc = []
rf_tpr = []
rf_fpr = []
rf_total = []
rf_correct = []
for exp in rf_experiments:
    for i in range(5):
        with open(os.path.join(base_dir, experiment_subdir, exp, "{0}_rf_correct.json".format(i)), 'r') as f:
            rf_correct.append(eval(f.readlines()[0]))        
        with open(os.path.join(base_dir, experiment_subdir, exp, "{0}_rf_auc.json".format(i)), 'r') as f:
            rf_auc.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, "{0}_rf_tpr.json".format(i)), 'r') as f:
            tpr = eval(f.readlines()[0])
            tpr = {key: eval(re.sub('\s+', ',', val)) for key, val in tpr.items()}
            rf_tpr.append(tpr)
        with open(os.path.join(base_dir, experiment_subdir, exp, "{0}_rf_fpr.json".format(i)), 'r') as f:
            fpr = eval(f.readlines()[0])
            fpr = {key: eval(re.sub('\s+', ',', val)) for key, val in fpr.items()}
            rf_fpr.append(fpr)
        with open(os.path.join(base_dir, experiment_subdir, exp, "{0}_rf_total.json".format(i)), 'r') as f:
            rf_total.append(eval(f.readlines()[0]))

rf_total = pd.DataFrame(rf_total)
rf_correct = pd.DataFrame(rf_correct) / rf_total
rf_auc = pd.DataFrame(rf_auc).astype(float)
rf_fpr = {key: [np.array(x[key]) for x in rf_fpr] for key in rf_total.keys()}
rf_tpr = {key: [np.array(x[key]) for x in rf_tpr] for key in rf_total.keys()}
# Compute aggregate accuracies
rf_accuracy= np.array([sum((rf_correct*rf_total).iloc[i]) / sum(rf_total.iloc[i]) for i in range(10)])

prevalence = [rf_total.iloc[i] / sum(rf_total.iloc[i]) for i in range(10)]
prevalence = pd.DataFrame(prevalence)
rf_ppv = {
    key: [
        np.nan_to_num(rf_tpr[key][i]*prevalence[key][i] / (rf_tpr[key][i]*prevalence[key][i] + rf_fpr[key][i]*(1-prevalence[key][i])), 0)
        for i in range(10) 
    ] 
    for key in rf_total.keys()
}

rf_aupr = {
    key: [
        interpolate_and_integrate(pd.Series(rf_tpr[key][i]), pd.Series(rf_ppv[key][i]), interpolator=interpolate_tpr_ppv)
        for i in range(10)
    ]
    for key in rf_total.keys()
}
rf_aupr = pd.DataFrame(rf_aupr)

df_dict = [pd.DataFrame({'RF': rf_aupr[key], 'GCN': cnn_aupr[key], 'DNN': dnn_aupr[key]}).assign(Disease=key) for key in rf_correct.keys()]  
cdf = pd.concat(df_dict)
mdf = pd.melt(cdf, id_vars=['Disease'], var_name=['Model']) 
ax = sns.boxplot(y="Disease", x="value", hue = 'Model', data=mdf)
ax.set_title("Area Under Precision-Recall")
plt.show()

df_dict = [pd.DataFrame({'RF': rf_auc[key], 'GCN': cnn_auc[key], 'DNN': dnn_auc[key]}).assign(Disease=key) for key in rf_correct.keys()]  
cdf = pd.concat(df_dict)
mdf = pd.melt(cdf, id_vars=['Disease'], var_name=['Model']) 
ax = sns.boxplot(y="Disease", x="value", hue = 'Model', data=mdf)
ax.set_title("Area Under ROC")
plt.show()
