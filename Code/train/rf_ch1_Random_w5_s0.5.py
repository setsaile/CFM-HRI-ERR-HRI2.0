import os
import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score

from sklearn.metrics import roc_auc_score
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
# 移除lightgbm导入


train_data_path = "./BASE/Data/rf_train_window_size_5_stride_0.5.csv"
val_data_path = "./BASE/Data/rf_val_window_size_5_stride_0.5.csv"
test_data_path = "./BASE/Data/rf_test_window_size_5_stride_0.5.csv"

train_data = pd.read_csv(train_data_path, low_memory=False)
val_data = pd.read_csv(val_data_path, low_memory=False)
test_data = pd.read_csv(test_data_path, low_memory=False)

n_windows = len(train_data)
print(f"Number of windows: {n_windows}")

n_features = len(train_data.columns) 
print(f"Number of features: {n_features}")

positive_rate = (train_data['robot_error'] == 1).sum() / len(train_data['robot_error']) * 100
print(f"Positive Rate: {positive_rate}%")

# Load Test Data
# Naming Pattern for Label Files
LABEL_FILE_PATTERNS = {
    "robot_errors": "challenge1_robot_error_labels_{task}_train.csv",
    "human_reactions_ch1": "challenge1_user_reaction_labels_{task}_train.csv",
    "human_reactions_ch2": "challenge2_user_reaction_labels_{task}_train.csv"
}

def fix_timestamp_format(ts):
    if isinstance(ts, str):
        parts = ts.split(':')
        if len(parts) == 4:
            return ':'.join(parts[:3]) + '.' + parts[3]
    return ts

"""Load all label files for a specific task and preprocess timestamps"""
# Base Path to Data
BASE_PATH = "./data_BASE"

def load_and_preprocess_labels(task):
    """Load all label files for a specific task and preprocess timestamps"""
    labels = {}    
    
    try:
        # Load robot error labels
        robot_errors = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge1_train/{LABEL_FILE_PATTERNS['robot_errors'].format(task=task)}"
        )
        robot_errors['error_onset'] = robot_errors['error_onset'].apply(fix_timestamp_format)
        robot_errors['error_onset'] = pd.to_timedelta(robot_errors['error_onset'])
        robot_errors['error_offset'] = robot_errors['error_offset'].apply(fix_timestamp_format)
        robot_errors['error_offset'] = pd.to_timedelta(robot_errors['error_offset'])
        robot_errors['trial'] = robot_errors['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['robot_errors'] = robot_errors        
        
        # Load human reaction labels
        human_reactions_ch1 = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge1_train/{LABEL_FILE_PATTERNS['human_reactions_ch1'].format(task=task)}"
        )
        human_reactions_ch1['reaction_onset'] = human_reactions_ch1['reaction_onset'].apply(fix_timestamp_format)
        human_reactions_ch1['reaction_onset'] = pd.to_timedelta(human_reactions_ch1['reaction_onset'])
        human_reactions_ch1['reaction_offset'] = human_reactions_ch1['reaction_offset'].apply(fix_timestamp_format)
        human_reactions_ch1['reaction_offset'] = pd.to_timedelta(human_reactions_ch1['reaction_offset'])
        human_reactions_ch1['trial'] = human_reactions_ch1['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['human_reactions_ch1'] = human_reactions_ch1        
        
        human_reactions_ch2 = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge2_train/{LABEL_FILE_PATTERNS['human_reactions_ch2'].format(task=task)}"
        )
        human_reactions_ch2['reaction_onset'] = human_reactions_ch2['reaction_onset'].apply(fix_timestamp_format)
        human_reactions_ch2['reaction_onset'] = pd.to_timedelta(human_reactions_ch2['reaction_onset'])
        human_reactions_ch2['reaction_offset'] = human_reactions_ch2['reaction_offset'].apply(fix_timestamp_format)
        human_reactions_ch2['reaction_offset'] = pd.to_timedelta(human_reactions_ch2['reaction_offset'])
        human_reactions_ch2['trial'] = human_reactions_ch2['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['human_reactions_ch2'] = human_reactions_ch2    
    
    except Exception as e:
        print(f"Error loading label files for task {task}: {str(e)}")
        return None    
    
    return labels, robot_errors, human_reactions_ch1, human_reactions_ch2

all_robot_errors = pd.DataFrame()
all_human_reactions_ch1 = pd.DataFrame()
all_human_reactions_ch2 = pd.DataFrame()

tasks = train_data['task'].unique()
for task in tasks:
    print(f"\n  Task: {task}")            
    
    # Load label files for this task
    labels, robot_errors, human_reactions_ch1, human_reactions_ch2 = load_and_preprocess_labels(task)
    robot_errors['task'] = task
    human_reactions_ch1['task'] = task
    human_reactions_ch2['task'] = task
    all_robot_errors = pd.concat([all_robot_errors, robot_errors], ignore_index=True)
    all_human_reactions_ch1 = pd.concat([all_human_reactions_ch1, human_reactions_ch1], ignore_index=True)
    all_human_reactions_ch2 = pd.concat([all_human_reactions_ch2, human_reactions_ch2], ignore_index=True)
    

def evaluate_error(y_pred_df, y_true_df):
    """
    Compute classification metrics and count of detected error events.
    
    Parameters
    ----------
    y_pred_df : pd.DataFrame
        DataFrame must contain 'task', 'trial', 'start', 'end', 'y_pred_err',
    y_pred_df : pd.DataFrame
        DataFrame must contain 'task', 'trial', 'error_onset', 'error_offset'.
        
    Returns
    -------
    tp: number of true positives 
    fp: number of false positives
    total_errors: number of errors in total
    """

    # Make sure y_true only include trials in pred
    evaluation_trials = y_pred_df['trial'].astype(str).unique()
    y_true_df = y_true_df.loc[y_true_df['trial'].isin(evaluation_trials)]
    
    # Look for true positive and false positive 
    pos_pred = y_pred_df.loc[y_pred_df['y_pred_err'] == 1].copy()  # 使用.copy()避免SettingWithCopyWarning
    pos_pred['id'] = pos_pred.index
    pos_pred['overlap_error'] = 0

    # initialize metrics
    tp = 0
    fp = 0
    total_errors = len(y_true_df)

    # check that the trials contain errors 
    if len(y_true_df) > 0:
        for _, row in y_true_df.iterrows():
            task = row['task']
            trial = row['trial']
            error_onset = row['error_onset'].total_seconds() - 1 # added one second tolerance
            error_offset = row['error_offset'].total_seconds() + 1 # added one second tolerance

            # check if the predicted error overlaps with actual error 
            detected_err = pos_pred[(pos_pred['task'] == task) & (pos_pred['trial'] == trial) & 
                                    ((pos_pred['start'] >= error_onset) & (pos_pred['start'] <= error_offset)) |
                                    ((pos_pred['end']   >= error_onset) & (pos_pred['end']   <= error_offset)) |
                                    ((pos_pred['start'] <= error_onset) & (pos_pred['end'] >= error_offset))]
            pos_pred.loc[pos_pred['id'].isin(detected_err['id']), 'overlap_error'] = 1  # 修复赋值方式
            if len(detected_err) > 0: 
                tp = tp + 1

        # prediction is a false positive if it does not overlap with actual error 
        fp = len(pos_pred.loc[pos_pred['overlap_error'] == 0])
        
        print(f"True Positive: {tp} ({tp / total_errors * 100}/%)") 
        print(f"False Positive: {fp}")

        fn = total_errors - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # 避免除零错误
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 避免除零错误
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # 避免除零错误
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")

    return tp, fp, total_errors

from tqdm import tqdm
import numpy as np
from sklearn.impute import SimpleImputer
import joblib
import os

# split
df_train = train_data.copy()
df_val   = val_data.copy()
df_test  = test_data.copy()

X_train = df_train.drop(columns=[
    'start','end','robot_error','reaction_ch1','reaction_ch2',
    'reaction_type','system','task','trial'
])
X_val   = df_val.drop(columns=[
    'start','end','robot_error','reaction_ch1','reaction_ch2',
    'reaction_type','system','task','trial'
])
X_test  = df_test.drop(columns=[
    'start','end','system','task','trial'
])

y_err_train = df_train['robot_error']
y_err_val   = df_val['robot_error']

#使用mean填充缺失值
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# with SMOTE 插值处理类别不平衡
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_imputed, y_err_train)
X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)

print("X_train_sm.shape:", X_train_sm.shape)
print("y_train_sm.shape:", y_train_sm.shape)

# 使用RandomForest模型替代LightGBM
model_path = "./BASE/models/clf_ch1_Randomforest_w5_s0.5.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

if os.path.exists(model_path):
    print("Loading model from disk...")
    clf_err = joblib.load(model_path)
else:
    print(" • Training challenge 1 model with RandomForest...")
    
    # 创建RandomForest分类器，优化参数以提高TP和F1，降低FP
    clf_err = RandomForestClassifier(
        n_estimators=200,  # 增加树的数量
        max_depth=15,      # 控制树的深度，避免过拟合
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced_subsample',  # 使用平衡的类权重
        n_jobs=-1,         # 使用所有CPU核心
        random_state=42,
        criterion='entropy'  # 使用entropy可能对不平衡数据更敏感
    )
    
    print("   Training RandomForest classifier...")
    with tqdm(desc="RandomForest training") as pbar:
        clf_err.fit(X_train_sm, y_train_sm)
        pbar.update(1)
    
    print("   Saving model...")
    joblib.dump(clf_err, model_path)
    
    print(" done (RandomForest model trained and saved).")

# 预测
probs_err = clf_err.predict_proba(X_val_imputed)[:, 1]
print("probs_err:", len(probs_err))
threshold = 0.4  # 调整阈值以平衡TP和FP
print(f'threshold: {threshold}')
pred_err = (probs_err > threshold).astype(int)

probs_reaction_test = clf_err.predict_proba(X_test_imputed)[:, 1]
print("probs_err:", len(probs_reaction_test))
print(f'threshold: {threshold}')
pred_reaction_test = (probs_reaction_test > threshold).astype(int)

# compute error-model metrics
f1_e = f1_score(y_err_val, pred_err, average='macro')
acc_e = accuracy_score(y_err_val, pred_err)
if sum(y_err_val) > 0: 
    tn, fp, fn, tp = confusion_matrix(y_err_val, pred_err).ravel()
else: 
    tn, fp, fn, tp = 0, 0, 0, 0

tpr_e = tp / (tp + fn) if (tp + fn) > 0 else 0  # 避免除零错误
fpr_e = fp / (fp + tn) if (fp + tn) > 0 else 0  # 避免除零错误
tnr_e = tn / (tn + fp) if (tn + fp) > 0 else 0  # 避免除零错误
fnr_e = fn / (fn + tp) if (fn + tp) > 0 else 0  # 避免除零错误
auc_e = roc_auc_score(y_err_val, probs_err)

# print fold metrics
print(
    f"AUC: {auc_e:.3f}, " 
    f"F1: {f1_e:.3f}, Acc: {acc_e:.3f}, "
    f"TPR: {tpr_e:.3f}, FPR: {fpr_e:.3f}, "
    f"TNR: {tnr_e:.3f}, FNR: {fnr_e:.3f}, "
)

df_val['y_pred_err'] = pred_err
df_test['y_pred_reaction'] = pred_reaction_test
val_trials = df_val['trial'].astype(str).unique()

# 保存test形式
pred_test_df = df_test[['task', 'trial', 'start', 'end', 'y_pred_reaction']]
# pred_test_df.to_csv("./BASE/result/y_pred_ch1_w5_s0.5_randomforest.csv", index = False)

y_pred_df = df_val[['task', 'trial', 'start', 'end', 'y_pred_err']]
print(len(y_pred_df))
y_true_df = all_robot_errors[['task', 'trial', 'error_onset', 'error_offset']].loc[all_robot_errors['trial'].isin(val_trials)]
print(len(y_true_df))
tp, fp, total_error = evaluate_error(y_pred_df, y_true_df)
