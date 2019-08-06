import os
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from bidict import bidict

base_dir = ''
experiment_subdir = "data/experiment"

cnn_experiments = [] # Add in the experiment names here
cnn_total_dirs = ["deep_{0}_cnn_total.json".format(i) for i in range(5)]
cnn_label_dirs = ["deep_{0}_cnn_labels_dict.json".format(i) for i in range(5)]
cnn_top_5_predictions_dirs = ["deep_{0}_cnn_top_5_predictions.json".format(i) for i in range(5)]

dnn_experiments = [] # Add in the experiment names here
dnn_total_dirs = ["deep_{0}_total.json".format(i) for i in range(15)]
dnn_label_dirs = ["deep_{0}_dnn_labels_dict.json".format(i) for i in range(15)]
dnn_top_5_predictions_dirs = ["deep_{0}_top_5_predictions.json".format(i) for i in range(15)]

rf_experiments = [] # Add in the experiment names here
rf_total_dirs = ["{0}_rf_total.json".format(i) for i in range(5)]
rf_top_5_predictions_dirs = ["rf_{0}_top_5_predictions.json".format(i) for i in range(5)]

important_columns = ['CRC', 'HBV', 'IBD', 'T2D', 'healthy', 'hypertension']
all_columns = ['AD', 'CDI', 'CRC', 'HBV', 'IBD', 'IGT', 'RA', 'T1D', 'T2D', 'adenoma', 'bronchitis', 'fatty_liver', 'healthy', 'hypertension', 'infectiousgastroenteritis', 'metabolic_syndrome', 'otitis', 'periodontitis', 'psoriasis']
cnn_top5_predictions = []
cnn_total = []
cnn_labels_dict = []

for exp in cnn_experiments:
    for i in range(5):
        with open(os.path.join(base_dir, experiment_subdir, exp, cnn_label_dirs[i]), 'r') as f:    
            cnn_labels_dict.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, cnn_total_dirs[i]), 'r') as f:
            cnn_total.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, cnn_top_5_predictions_dirs[i]), 'r') as f:
            cnn_top5_predictions.append(eval(f.readlines()[0]))

cnn_total = pd.DataFrame(cnn_total)
cnn_t1p = []
for labels_dict, t5p in zip(cnn_labels_dict, cnn_top5_predictions):
    ld = bidict(labels_dict)
    labelled_t1p = [pd.Series({
        "from": disease,
        "to": ld.inv[p[0]]
    })
    for disease, preds in t5p.items()
    for p in preds
    ]
    cnn_t1p.extend(labelled_t1p)

cnn_t1p = pd.DataFrame(cnn_t1p)
cnn_t1p_counts = pd.DataFrame(
        {
        "from": x,
        "to": y,
        "count": sum( (cnn_t1p['from'] == x) * (cnn_t1p['to'] == y) )
    }
    for x in important_columns
    for y in important_columns
    if x != y    
)
cnn_counts = pd.DataFrame({"from": cnn_t1p_counts['to'], "to": cnn_t1p_counts["from"], "value": cnn_t1p_counts['count']})
for disease in important_columns:
    new_values = cnn_counts[cnn_counts["from"] == disease]['value']/sum(cnn_counts[cnn_counts["from"] == disease]['value'])
    cnn_counts.loc[new_values.index, 'value'] = new_values
cnn_counts.to_csv("images/cnn_links.csv")

dnn_top5_predictions = []
dnn_total = []
dnn_labels_dict = []
for exp in dnn_experiments:
    for i in range(15):
        with open(os.path.join(base_dir, experiment_subdir, exp, dnn_label_dirs[i]), 'r') as f:    
            dnn_labels_dict.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, dnn_total_dirs[i]), 'r') as f:
            dnn_total.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, dnn_top_5_predictions_dirs[i]), 'r') as f:
            dnn_top5_predictions.append(eval(f.readlines()[0]))

dnn_total = pd.DataFrame(dnn_total)
dnn_t1p = []
for labels_dict, t5p in zip(dnn_labels_dict, dnn_top5_predictions):
    ld = bidict(labels_dict)
    labelled_t1p = [pd.Series({
        "from": disease,
        "to": ld.inv[p[0]]
    })
    for disease, preds in t5p.items()
    for p in preds
    ]
    dnn_t1p.extend(labelled_t1p)

dnn_t1p = pd.DataFrame(dnn_t1p)
dnn_t1p_counts = pd.DataFrame(
        {
        "from": x,
        "to": y,
        "count": sum( (dnn_t1p['from'] == x) * (dnn_t1p['to'] == y) )
    }
    for x in important_columns
    for y in important_columns
    if x != y    
)
dnn_counts = pd.DataFrame({"from": dnn_t1p_counts['to'], "to": dnn_t1p_counts["from"], "value": dnn_t1p_counts['count']})
for disease in important_columns:
    new_values = dnn_counts[dnn_counts["from"] == disease]['value']/sum(dnn_counts[dnn_counts["from"] == disease]['value'])
    dnn_counts.loc[new_values.index, 'value'] = new_values
dnn_counts.to_csv("images/dnn_links.csv")

rf_top5_predictions = []
rf_total = []
for exp in rf_experiments:
    for i in range(5):
        with open(os.path.join(base_dir, experiment_subdir, exp, rf_total_dirs[i]), 'r') as f:
            rf_total.append(eval(f.readlines()[0]))
        with open(os.path.join(base_dir, experiment_subdir, exp, rf_top_5_predictions_dirs[i]), 'r') as f:
            rf_top5_predictions.append(eval(f.readlines()[0]))

rf_total = pd.DataFrame(rf_total)
rf_t1p = []
for t5p in rf_top5_predictions:
    labelled_t1p = [
        pd.Series({
            "from": disease, "to": p[0]
        })
        for disease, preds in t5p.items()
        for p in preds
    ]
    rf_t1p.extend(labelled_t1p)

rf_t1p = pd.DataFrame(rf_t1p)
rf_t1p_counts = pd.DataFrame(
        {
        "from": x,
        "to": y,
        "count": sum( (rf_t1p['from'] == x) * (rf_t1p['to'] == y) )
    }
    for x in important_columns
    for y in important_columns
    if x != y
)
# For some reason, the columns are backwards
rf_counts = pd.DataFrame({"from": rf_t1p_counts['to'], "to": rf_t1p_counts["from"], "value": rf_t1p_counts['count']})
for disease in important_columns:
    new_values = rf_counts[rf_counts["from"] == disease]['value']/sum(rf_counts[rf_counts["from"] == disease]['value'])
    rf_counts.loc[new_values.index, 'value'] = new_values
rf_counts.to_csv("images/rf_links.csv")
