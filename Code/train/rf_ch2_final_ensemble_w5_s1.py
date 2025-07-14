import os
import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
# 首先确保导入lightgbm
import lightgbm as lgb

from imblearn.over_sampling import SMOTE    # SMOTE是用来插值采样的,解决label不平衡的方法
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer, KNNImputer    # 用于填补NaN的工具类
import joblib # 保存sklearn模型
from tqdm import tqdm

train_data_path = "./BASE/Data/rf_train_window_size_5_stride_1.csv"
val_data_path = "./BASE/Data/rf_val_window_size_5_stride_1.csv"
test_data_path = "./BASE/Data/rf_test_window_size_5_stride_1.csv"

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
    pos_pred = y_pred_df.loc[y_pred_df['y_pred_reaction'] == 1].copy()
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
            if not detected_err.empty:
                pos_pred.loc[pos_pred['id'].isin(detected_err['id']), 'overlap_reaction'] = 1
                tp += 1

        # prediction is a false positive if it does not overlap with actual reaction
        fp = len(pos_pred.loc[pos_pred['overlap_reaction'] == 0])
        
        print(f"True Positive: {tp} ({tp / total_reactions * 100:.2f}%)")
        print(f"False Positive: {fp}")

        fn = total_reactions - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

    return tp, fp, total_reactions

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

# 使用KNNImputer填充缺失值，比SimpleImputer更精确
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# 使用SMOTE过采样和RandomUnderSampler欠采样结合，创建更平衡的数据集
sampling_strategy = 0.8  # 正负样本比例
over_sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
sampling_pipeline = ImbPipeline([
    ('over', over_sampler),
    ('under', under_sampler)
])
X_train_balanced, y_train_balanced = sampling_pipeline.fit_resample(X_train_imputed, y_reaction_train)

# 将数据转换为DataFrame以便于特征名称的使用
X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=X_train.columns)

print("X_train_balanced.shape:", X_train_balanced.shape)
print("y_train_balanced.shape:", y_train_balanced.shape)
print(f"Balanced positive rate: {sum(y_train_balanced) / len(y_train_balanced) * 100:.2f}%")

# 设置模型保存路径
ensemble_model_path = "./BASE/models/clf_ch2_final_ensemble_w5_s1.pkl"

# 检查模型是否已存在
if os.path.exists(ensemble_model_path):
    print("Loading ensemble model from disk...")
    ensemble_clf = joblib.load(ensemble_model_path)
else:
    print(" • Training ensemble model for challenge 2...")
    
    # 1. RandomForest模型
    print("   Training RandomForest model...")
    rf_model_path = "./BASE/models/clf_ch2_rf_win5_stri1_ensemble.pkl"
    
    if os.path.exists(rf_model_path):
        print("   Loading RandomForest model from disk...")
        rf_clf = joblib.load(rf_model_path)
    else:
        # 使用特征选择减少过拟合
        selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
        X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)
        
        # 创建RandomForest分类器
        rf_clf = RandomForestClassifier(
            n_estimators=500,  # 增加树的数量
            max_depth=12,      # 增加树的深度
            min_samples_split=5,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        # 训练模型
        print("   Training RandomForest classifier...")
        with tqdm(desc="RandomForest training") as pbar:
            rf_clf.fit(X_train_selected, y_train_balanced)
            pbar.update(1)
        
        # 保存RandomForest模型和特征选择器
        joblib.dump((selector, rf_clf), rf_model_path)
    
    # 2. LightGBM模型
    print("   Training LightGBM model...")
    lgb_model_path = "./BASE/models/clf_ch2_lgb_win5_stri1_ensemble.pkl"
    
    if os.path.exists(lgb_model_path):
        print("   Loading LightGBM model from disk...")
        lgb_clf = joblib.load(lgb_model_path)
    else:
        # 创建LightGBM分类器
        lgb_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_leaves': 63,         # 增加叶子节点数量
            'max_depth': 10,          # 增加树的深度
            'learning_rate': 0.03,    # 降低学习率以减少过拟合
            'n_estimators': 500,      # 增加树的数量
            'class_weight': 'balanced',
            'scale_pos_weight': 3,    # 增加正样本权重
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'importance_type': 'gain',
            'verbose': -1
        }
        
        lgb_clf = lgb.LGBMClassifier(**lgb_params)
        
        # 训练模型
        print("   Training LightGBM classifier...")
        with tqdm(desc="LightGBM training") as pbar:
            lgb_clf.fit(
                X_train_balanced, y_train_balanced,
                eval_set=[(X_val_imputed, y_reaction_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
            pbar.update(1)
        
        # 保存LightGBM模型
        joblib.dump(lgb_clf, lgb_model_path)
    
    # 创建包装类，将特征选择器和随机森林组合成一个单一的分类器
    from sklearn.base import BaseEstimator, ClassifierMixin
    
    class RFWithSelector(BaseEstimator, ClassifierMixin):
        def __init__(self, selector, classifier):
            self.selector = selector
            self.classifier = classifier
            self._is_fitted = True
            
        def fit(self, X, y):
            # 直接对selector进行fit
            print("Fitting feature selector...")
            self.selector.fit(X, y)
                
            X_selected = self.selector.transform(X)
            self.classifier.fit(X_selected, y)
            self._is_fitted = True
            return self
                    
        def predict(self, X):
            if not self._is_fitted:
                raise ValueError("RFWithSelector has not been fitted yet.")
            X_selected = self.selector.transform(X)
            return self.classifier.predict(X_selected)
                
        def predict_proba(self, X):
            if not self._is_fitted:
                raise ValueError("RFWithSelector has not been fitted yet.")
            X_selected = self.selector.transform(X)
            return self.classifier.predict_proba(X_selected)
    
    # 创建包装后的RandomForest分类器
    selector, rf_clf_unwrapped = joblib.load(rf_model_path)
    rf_classifier = RFWithSelector(selector, rf_clf_unwrapped)
    
    # 定义集成模型 - 使用VotingClassifier
    print("   Creating ensemble model...")
    
    # 定义集成模型 - 使用VotingClassifier
    ensemble_clf = VotingClassifier(
        estimators=[
            ('randomforest', rf_classifier),
            ('lightgbm', lgb_clf)
        ],
        voting='soft',  # 使用概率进行加权投票
        weights=[0.6, 0.4]  # 根据各模型性能分配权重，RandomForest权重更高
    )
    
    # 训练集成模型
    print("   Training ensemble model...")
    ensemble_clf.fit(X_train_balanced, y_train_balanced)
    
    # 保存集成模型
    print("   Saving ensemble model...")
    joblib.dump(ensemble_clf, ensemble_model_path)
    print(" done (Ensemble model trained and saved).")

# 交叉验证评估
print("Performing cross-validation...")
cv_scores = cross_val_score(ensemble_clf, X_val_imputed, y_reaction_val, cv=5, scoring='f1')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1 score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 预测验证集
print("Predicting on validation set...")
probs_reaction = ensemble_clf.predict_proba(X_val_imputed)[:, 1]

# 寻找最佳阈值 - 基于F1分数
thresholds = np.linspace(0.3, 0.7, 20)
best_f1 = 0
best_threshold = 0.5
best_tp = 0
best_fp = 0

# 创建临时DataFrame用于评估不同阈值
temp_df_val = df_val.copy()
temp_y_true_df = all_human_reactions_ch2[['task', 'trial', 'reaction_onset', 'reaction_offset']]
val_trials = df_val['trial'].astype(str).unique()
temp_y_true_df = temp_y_true_df.loc[temp_y_true_df['trial'].isin(val_trials)]

print("Finding optimal threshold...")
for threshold in thresholds:
    # 应用阈值
    pred = (probs_reaction >= threshold).astype(int)
    temp_df_val['y_pred_reaction'] = pred
    
    # 评估当前阈值
    temp_y_pred_df = temp_df_val[['task', 'trial', 'start', 'end', 'y_pred_reaction']]
    tp, fp, total_reaction = evaluate_reaction(temp_y_pred_df, temp_y_true_df)
    
    # 计算F1分数
    fn = total_reaction - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Threshold: {threshold:.2f}, TP: {tp}, FP: {fp}, F1: {f1:.4f}")
    
    # 更新最佳阈值
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_tp = tp
        best_fp = fp

print(f"\nBest threshold: {best_threshold:.4f}")
print(f"Best F1: {best_f1:.4f}")
print(f"Best TP: {best_tp}")
print(f"Best FP: {best_fp}")

# 使用最佳阈值进行最终预测
pred_reaction = (probs_reaction >= best_threshold).astype(int)

# 预测测试集
print("Predicting on test set...")
probs_reaction_test = ensemble_clf.predict_proba(X_test_imputed)[:, 1]
pred_reaction_test = (probs_reaction_test >= best_threshold).astype(int)

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
pred_test_df.to_csv("./BASE/result/y_pred_ch2_w5_s1_FinalEnsemble.csv", index=False)

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

