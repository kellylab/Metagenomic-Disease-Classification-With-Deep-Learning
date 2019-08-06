import os
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt 

base_dir = 'fill in here'
experiment_subdir = "data/experiment"

cnn_experiments = [] # Add in the experiment names here
cnn_total_dirs = ["deep_{0}_cnn_total.json".format(i) for i in range(5)]
cnn_correct_dirs = ["deep_{0}_cnn_correct.json".format(i) for i in range(5)]
cnn_top3_dirs = ["deep_{0}_cnn_top_3.json".format(i) for i in range(5)]
cnn_top5_dirs = ["deep_{0}_cnn_top_5.json".format(i) for i in range(5)]

dnn_experiments = [] # Add in the experiment names here
dnn_total_dirs = ["deep_{0}_total.json".format(i) for i in range(15)]
dnn_correct_dirs = ["deep_{0}_correct.json".format(i) for i in range(15)]
dnn_top3_dirs = ["deep_{0}_top_3.json".format(i) for i in range(15)]
dnn_top5_dirs = ["deep_{0}_top_5.json".format(i) for i in range(15)]

rf_experiments = [] # Add in the experiment names here

cnn_total = []
cnn_correct = []
cnn_top3 = []
cnn_top5 = []
dnn_total = []
dnn_correct = []
dnn_top3 = []
dnn_top5 = []
rf_correct = []
rf_total = []
rf_top3 = []
rf_top5 = []
rf_auc = []
rf_tpr = []
rf_fpr = []

# Get a table of accuracy metrics accross experiments for neural networks
for exp in cnn_experiments:
    for i in range(5):
        with open(os.path.join(base_dir, experiment_subdir, exp, cnn_total_dirs[i]), 'r') as f:
            cnn_total.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, cnn_correct_dirs[i]), 'r') as f:
            cnn_correct.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, cnn_top3_dirs[i]), 'r') as f:
            cnn_top3.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, cnn_top5_dirs[i]), 'r') as f:
            cnn_top5.append(eval(f.readlines()[0]))
for exp in dnn_experiments:
    for i in range(15):
        with open(os.path.join(base_dir, experiment_subdir, exp, dnn_total_dirs[i]), 'r') as f:
            dnn_total.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, dnn_correct_dirs[i]), 'r') as f:
            dnn_correct.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, dnn_top3_dirs[i]), 'r') as f:
            dnn_top3.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, dnn_top5_dirs[i]), 'r') as f:
            dnn_top5.append(eval(f.readlines()[0]))

cnn_total = pd.DataFrame(cnn_total)
cnn_correct = pd.DataFrame(cnn_correct) / cnn_total
cnn_top3 = pd.DataFrame(cnn_top3) / cnn_total
cnn_top5 = pd.DataFrame(cnn_top5) / cnn_total
dnn_total = pd.DataFrame(dnn_total)
dnn_correct = pd.DataFrame(dnn_correct) / dnn_total
dnn_top3 = pd.DataFrame(dnn_top3) / dnn_total
dnn_top5 = pd.DataFrame(dnn_top5) / dnn_total

for exp in rf_experiments:
    for i in range(5):
        with open(os.path.join(base_dir, experiment_subdir, exp, "{0}_rf_correct.json".format(i)), 'r') as f:
            rf_correct.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, "{0}_rf_total.json".format(i)), 'r') as f:
            rf_total.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, "{0}_rf_top_3.json".format(i)), 'r') as f:
            rf_top3.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, "{0}_rf_top_5.json".format(i)), 'r') as f:
            rf_top5.append(eval(f.readlines()[0]))
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

rf_total = pd.DataFrame(rf_total)
rf_correct = pd.DataFrame(rf_correct) / rf_total
rf_top3 = pd.DataFrame(rf_top3) / rf_total
rf_top5 = pd.DataFrame(rf_top5) / rf_total
rf_auc = pd.DataFrame(rf_auc).astype(float)
rf_fpr = {key: [np.array(x[key]) for x in rf_fpr] for key in rf_total.keys()}
rf_tpr = {key: [np.array(x[key]) for x in rf_tpr] for key in rf_total.keys()}
# Compute aggregate accuracies
cnn_accuracy = np.array([sum((cnn_correct*cnn_total).iloc[i]) / sum(cnn_total.iloc[i]) for i in range(30)])
dnn_accuracy = np.array([sum((dnn_correct*dnn_total).iloc[i]) / sum(dnn_total.iloc[i]) for i in range(30)])
rf_accuracy= np.array([sum((rf_correct*rf_total).iloc[i]) / sum(rf_total.iloc[i]) for i in range(20)])
cnn_top3_accuracy = np.array([sum((cnn_top3*cnn_total).iloc[i]) / sum(cnn_total.iloc[i]) for i in range(30)])
dnn_top3_accuracy = np.array([sum((dnn_top3*dnn_total).iloc[i]) / sum(dnn_total.iloc[i]) for i in range(30)])
rf_top3_accuracy= np.array([sum((rf_top3*rf_total).iloc[i]) / sum(rf_total.iloc[i]) for i in range(20)])
cnn_top5_accuracy = np.array([sum((cnn_top5*cnn_total).iloc[i]) / sum(cnn_total.iloc[i]) for i in range(30)])
dnn_top5_accuracy = np.array([sum((dnn_top5*dnn_total).iloc[i]) / sum(dnn_total.iloc[i]) for i in range(30)])
rf_top5_accuracy= np.array([sum((rf_top5*rf_total).iloc[i]) / sum(rf_total.iloc[i]) for i in range(20)])

prevalence = [rf_total.iloc[i] / sum(rf_total.iloc[i]) for i in range(10)]
prevalence = pd.DataFrame(prevalence)
rf_ppv = {
    key: [
        np.nan_to_num(rf_tpr[key][i]*prevalence[key][i] / (rf_tpr[key][i]*prevalence[key][i] + rf_fpr[key][i]*(1-prevalence[key][i])), 0)
        for i in range(10) 
    ] 
    for key in rf_total.keys()
}

ordered_labels = ['healthy', 'CRC', 'T2D', 'RA', 'hypertension', 'IBD', 'adenoma', 'otitis', 'HBV', 'fatty_liver', 'psoriasis', 'T1D', 'metabolic_syndrome', 'IGT', 'periodontitis', 'AD', 'CDI', 'infectiousgastroenteritis', 'bronchitis',]
df_dict = [
    pd.DataFrame({
        'RF': rf_correct[key],
        'CNN': cnn_correct[key], 
        'DNN': dnn_correct[key],
        # 'RF_top3': rf_top3[key], 
        # 'CNN_top3': cnn_top3[key], 
        # 'DNN_top3': dnn_top3[key],
        # 'RF_top5': rf_top5[key], 
        # 'CNN_top5': cnn_top5[key], 
        # 'DNN_top5': dnn_top5[key],

        }
    ).assign(Disease=key)
    for key in ordered_labels
]
cdf = pd.concat(df_dict)
mdf = pd.melt(cdf, id_vars=['Disease'], var_name=['Model']) 
ax = sns.boxplot(y="Disease", x="value", hue = 'Model', data=mdf, showfliers=False)
ax.set_title("Accuracy")
ax.set_xlim([0,1.])
plt.show()

rf_radar = pd.DataFrame({
            'group': ['Top 1', 'Top 3', 'Top 5'],
            **{
                disease: [rf_correct[disease].mean(), rf_top3[disease].mean(), rf_top5[disease].mean()]
                for disease in rf_correct.keys()
            },
        }
    )

dnn_radar = pd.DataFrame({
            'group': ['Top 1', 'Top 3', 'Top 5'],
            **{
                disease: [dnn_correct[disease].mean(), dnn_top3[disease].mean(), dnn_top5[disease].mean()]
                for disease in rf_correct.keys()
            },
        }
    )

cnn_radar = pd.DataFrame({
            'group': ['Top 1', 'Top 3', 'Top 5'],
            **{
                disease: [cnn_correct[disease].mean(), cnn_top3[disease].mean(), cnn_top5[disease].mean()]
                for disease in rf_correct.keys()
            },
        }
    )

def plot_radar(df, title='', legend=False):
    #------- PART 1: Create background
    from math import pi
    # number of variable
    categories=list(df)[1:]
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([.25,.5,.75], [".25",".5",".75"], color="grey", size=7)
    plt.ylim(0,1)
    
    
    # ------- PART 2: Add plots
    
    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind3
    values=df.loc[2].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='r', linewidth=1, linestyle='solid', label="Top 5")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Ind2
    values=df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='b', linewidth=1, linestyle='solid', label="Top 3")
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind1
    values=df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='g', linewidth=1, linestyle='solid', label="Top 1")
    ax.fill(angles, values, 'g', alpha=0.1)

    ax.set_title(title)
    # Add legend
    if legend:
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

