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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix
)
import numpy as np
from sklearn.metrics import roc_auc_score
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
# 首先确保导入lightgbm
import lightgbm as lgb

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

# 这进行了修改,test->train
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
# 进行修改
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
print("task:",tasks)
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
    
    
def evaluate_reaction(y_pred_df, y_true_df):
    """
    Compute classification metrics and count of detected error events.
    
    Parameters
    ----------
    y_pred_df : pd.DataFrame
        DataFrame must contain 'task', 'trial', 'start', 'end', 'y_pred_reaction',
    y_pred_df : pd.DataFrame
        DataFrame must contain 'task', 'trial', 'reaction_onset', 'reaction_offset'.
        
    Returns
    -------
    tp: number of true positives 
    fp: number of false positives
    total_reactions: number of reactions in total
    """

    # Make sure y_true only include trials in pred
    evaluation_trials = y_pred_df['trial'].astype(str).unique()
    y_true_df = y_true_df.loc[y_true_df['trial'].isin(evaluation_trials)]
    
    # Look for true positive and false positive 
    pos_pred = y_pred_df.loc[y_pred_df['y_pred_reaction'] == 1]
    pos_pred['id'] = pos_pred.index
    pos_pred['overlap_reaction'] = 0

    # initialize metrics
    tp = 0
    fp = 0
    total_reactions = len(y_true_df)

    # check that the trials contain reactions 
    if len(y_true_df) > 0:
        for _, row in y_true_df.iterrows():
            task = row['task']
            trial = row['trial']
            reaction_onset = row['reaction_onset'].total_seconds() - 1 # added one second tolerance
            reaction_offset = row['reaction_offset'].total_seconds() + 1 # added one second tolerance

            # check if the predicted reaction overlaps with actual reaction 
            detected_err = pos_pred[(pos_pred['task'] == task) & (pos_pred['trial'] == trial) & 
                                    ((pos_pred['start'] >= reaction_onset) & (pos_pred['start'] <= reaction_offset)) |
                                    ((pos_pred['end']   >= reaction_onset) & (pos_pred['end']   <= reaction_offset)) |
                                    ((pos_pred['start'] <= reaction_onset) & (pos_pred['end'] >= reaction_offset))]
            pos_pred.loc[pos_pred['id'].isin(detected_err['id']), 'overlap_reaction'] = 1
            if len(detected_err) > 0: 
                tp = tp + 1

        # prediction is a false positive if it does not overlap with actual reaction
        fp = len(pos_pred.loc[pos_pred['overlap_reaction'] == 0])
        
        print(f"True Positive: {tp} ({tp / total_reactions * 100}/%)")  
        print(f"False Positive: {fp}")

        fn = total_reactions - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"F1: {f1}")

    return tp, fp, total_reactions

from imblearn.over_sampling import SMOTE    # SMOTE是用来插值采样的,解决label不平衡的方法
from sklearn.impute import SimpleImputer    # 用于填补NaN的工具类
import joblib # 保存sklearn模型
from tqdm import tqdm

# split
df_train = train_data.copy()
df_val = val_data.copy()
df_test = test_data.copy()

# 丢弃非特征列
X_train = df_train.drop(columns=[
    'start','end','robot_error','reaction_ch1','reaction_ch2',
    'reaction_type','system','task','trial'
])
X_val = df_val.drop(columns=[
    'start','end','robot_error','reaction_ch1','reaction_ch2',
    'reaction_type','system','task','trial'
])
X_test = df_test.drop(columns=[
    'start','end','system','task','trial'
])

y_reaction_train = df_train['reaction_ch2']
y_reaction_val   = df_val['reaction_ch2'] 

# 使用SimpleImputer填充缺失值
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# 使用SMOTE处理不平衡数据
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_imputed, y_reaction_train)

# 将数据转换为DataFrame以便于特征名称的使用
X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)

print("X_train_sm.shape:", X_train_sm.shape)
print("y_train_sm.shape:", y_train_sm.shape)

# 设置模型保存路径
model_path = "./BASE/models/clf_ch2_lightgbm_win5_stri0.5.pkl"

if os.path.exists(model_path):
    print("Loading model from disk...")
    clf_reaction = joblib.load(model_path)
else:
    print(" • Training challenge 2 model with LightGBM…")
    
    # 创建LightGBM分类器
    # 使用较高的num_leaves和n_estimators以提高模型复杂度和表达能力
    # 使用较小的learning_rate以提高模型稳定性
    # 使用scale_pos_weight处理类别不平衡
    # 使用class_weight处理类别不平衡
    # 使用feature_fraction和bagging_fraction减少过拟合
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'class_weight': 'balanced',
        'scale_pos_weight': 2,  # 给予正样本更高的权重
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'importance_type': 'gain',
        'verbose': -1
    }
    
    # 创建LightGBM分类器
    clf_reaction = lgb.LGBMClassifier(**lgb_params)
    
    # 分步进行以显示更详细的进度
    print("   Training LightGBM classifier...")
    with tqdm(desc="LightGBM training") as pbar:
        # 修改为
        clf_reaction.fit(
        X_train_sm, y_train_sm,
        eval_set=[(X_val_imputed, y_reaction_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]  # 使用callbacks参数
        )
        pbar.update(1)
    
    # 保存模型
    print("   Saving model...")
    joblib.dump(clf_reaction, model_path)
    print(" done (LightGBM model trained and saved).")

# # 显示特征重要性
# feature_importance = clf_reaction.feature_importances_
# feature_names = X_train.columns
# feature_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': feature_importance
# }).sort_values(by='Importance', ascending=False)

# print("Top 10 feature importances:")
# print(feature_importance_df.head(10))

# 预测验证集
print("Predicting on validation set...")
probs_reaction = clf_reaction.predict_proba(X_val_imputed)[:, 1]

# 使用较高的阈值以减少FP，同时保持较高的TP
threshold = 0.45  # 提高阈值以减少FP
pred_reaction = (probs_reaction >= threshold).astype(int)

# 预测测试集
print("Predicting on test set...")
probs_reaction_test = clf_reaction.predict_proba(X_test_imputed)[:, 1]
pred_reaction_test = (probs_reaction_test >= threshold).astype(int)

# 计算评估指标
f1_e = f1_score(y_reaction_val, pred_reaction, average='macro')
acc_e = accuracy_score(y_reaction_val, pred_reaction)
if sum(y_reaction_val) > 0: 
    tn, fp, fn, tp = confusion_matrix(y_reaction_val, pred_reaction).ravel()
else: 
    tn, fp, fn, tp = 0, 0, 0, 0

tpr_e = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr_e = fp / (fp + tn) if (fp + tn) > 0 else 0
tnr_e = tn / (tn + fp) if (tn + fp) > 0 else 0
fnr_e = fn / (fn + tp) if (fn + tp) > 0 else 0
auc_e = roc_auc_score(y_reaction_val, probs_reaction) 

print(
    f"AUC: {auc_e:.3f}, " 
    f"F1: {f1_e:.3f}, Acc: {acc_e:.3f}, "
    f"TPR: {tpr_e:.3f}, FPR: {fpr_e:.3f}, "
    f"TNR: {tnr_e:.3f}, FNR: {fnr_e:.3f}, "
)

# 将预测结果添加到DataFrame
df_val['y_pred_reaction'] = pred_reaction
df_test['y_pred_reaction'] = pred_reaction_test
val_trials = df_val['trial'].astype(str).unique()

# 保存测试集预测结果
pred_test_df = df_test[['task', 'trial', 'start', 'end', 'y_pred_reaction']]
pred_test_df.to_csv("./BASE/result/y_pred_ch2_w5_s0.5_lightgbm.csv", index=False)

# 分别取预测标签和真实标签
y_pred_df = df_val[['task', 'trial', 'start', 'end', 'y_pred_reaction']]
y_true_df = all_human_reactions_ch2[['task', 'trial', 'reaction_onset', 'reaction_offset']].loc[all_human_reactions_ch2['trial'].isin(val_trials)]

# 查看 trial 列的唯一类别名称
print("y_pred_df trial 列唯一类别名称：")
print(y_pred_df['trial'].unique())

print("y_true_df trial 列唯一类别名称：")
print(y_true_df['trial'].unique())

# 评估模型
tp, fp, total_reaction = evaluate_reaction(y_pred_df, y_true_df)

