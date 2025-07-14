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

# 移除sktime的MiniRocket导入
# from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

# 导入tsai的MinirocketClassifier
import pip

# 检查并安装tsai库
try:
    import tsai
except ImportError:
    pip.main(['install', 'tsai'])
    pip.main(['install', 'sktime'])

# 导入tsai的MinirocketClassifier
from tsai.models.MINIROCKET import MiniRocketClassifier

train_data_path = "./BASE/Data/rf_train_window_size_3_stride_1.csv"
val_data_path = "./BASE/Data/rf_val_window_size_3_stride_1.csv"
test_data_path = "./BASE/Data/rf_test_window_size_3_stride_1.csv"

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
    """Compute classification metrics and count of detected error events."""

    # Make sure y_true only include trials in pred
    evaluation_trials = y_pred_df['trial'].astype(str).unique()
    y_true_df = y_true_df.loc[y_true_df['trial'].isin(evaluation_trials)]
    
    # Look for true positive and false positive 
    pos_pred = y_pred_df.loc[y_pred_df['y_pred_reaction'] == 1].copy()  # 使用copy()创建副本
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
            # 将这行
            pos_pred.loc[pos_pred['id'].isin(detected_err['id']), 'overlap_reaction'] = 1
            
            # 可以保持不变，因为我们已经确保pos_pred是一个副本
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

# 使用mean填充缺失值
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# with SMOTE处理类别不平衡
smote = SMOTE(random_state=42)  # 设置随机种子,保证每一次生成一样的值
X_train_sm, y_train_sm = smote.fit_resample(X_train_imputed, y_reaction_train)

# 在导入部分添加以下内容
import numpy as np
from scipy.special import softmax

# 添加自定义MiniRocketClassifier类，继承原始类并添加predict_proba方法
class CustomMiniRocketClassifier(MiniRocketClassifier):
    def predict_proba(self, X):
        """添加predict_proba方法，使用decision_function并通过softmax转换"""
        # 获取决策函数值
        d = self.decision_function(X)
        # 对于二分类问题，将决策值转换为二维数组 [-d, d]
        if len(self.classes_) == 2:
            d_2d = np.c_[-d, d]
        else:
            # 对于多分类问题，直接使用决策值
            d_2d = d
        # 使用softmax将决策值转换为概率
        return softmax(d_2d, axis=1)

# 在模型创建部分，将MiniRocketClassifier替换为CustomMiniRocketClassifier
# MinirocketClassifier需要的格式是(n_samples, n_channels, seq_len)
# 我们的数据是(n_samples, n_features)，需要将其reshape为(n_samples, 1, n_features)
X_train_sm_3d = X_train_sm.reshape(X_train_sm.shape[0], 1, X_train_sm.shape[1])
X_val_3d = X_val_imputed.reshape(X_val_imputed.shape[0], 1, X_val_imputed.shape[1])
X_test_3d = X_test_imputed.reshape(X_test_imputed.shape[0], 1, X_test_imputed.shape[1])

print("X_train_sm_3d.shape:", X_train_sm_3d.shape)
print("y_train_sm.shape:", y_train_sm.shape)

# 使用tsai的MinirocketClassifier替代原来的MiniRocket+LogisticRegressionCV
model_path = "./BASE/models/clf_ch2_tsaiminirocket_win3_stri1.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

if os.path.exists(model_path):
    print("Loading model from disk...")
    clf_reaction = joblib.load(model_path)
    # 如果加载的是原始MiniRocketClassifier，转换为自定义类
    if isinstance(clf_reaction, MiniRocketClassifier) and not isinstance(clf_reaction, CustomMiniRocketClassifier):
        clf_reaction.__class__ = CustomMiniRocketClassifier
else:
    print(" • Training challenge 2 model with tsai MinirocketClassifier…")
    
    # 创建CustomMiniRocketClassifier，使用更多的特征和更好的参数
    clf_reaction = CustomMiniRocketClassifier(
        num_features=5000,  # 增加特征数量以提高模型表达能力
        max_dilations_per_kernel=32,  # 增加扩张率以捕获更多时间模式
        random_state=42,
        # 移除 n_jobs=-1 参数，因为RidgeClassifierCV不支持这个参数
        class_weight='balanced'  # 处理类别不平衡
    )
    
    # 训练模型
    print("   Training MinirocketClassifier...")
    with tqdm(desc="MinirocketClassifier training") as pbar:
        clf_reaction.fit(X_train_sm_3d, y_train_sm)
        pbar.update(1)
    
    # 保存模型
    print("   Saving model...")
    joblib.dump(clf_reaction, model_path)
    
    print(" done (tsai MinirocketClassifier model trained and saved).")

# 预测
print("Predicting on validation set...")
probs_reaction = clf_reaction.predict_proba(X_val_3d)[:, 1]
# 调整阈值以优化TP和F1，同时减少FP
threshold = 0.35  # 提高阈值以减少FP
pred_reaction = (probs_reaction >= threshold).astype(int)

print("Predicting on test set...")
probs_reaction_test = clf_reaction.predict_proba(X_test_3d)[:, 1]
pred_reaction_test = (probs_reaction_test >= threshold).astype(int)

# compute reaction-model metrics
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

df_val['y_pred_reaction'] = pred_reaction
df_test['y_pred_reaction'] = pred_reaction_test
val_trials = df_val['trial'].astype(str).unique()

# 保存test形式
pred_test_df = df_test[['task', 'trial', 'start', 'end', 'y_pred_reaction']]
pred_test_df.to_csv("./BASE/result/y_pred_ch2_w3_s1_tsaiminirocket.csv", index = False)

# 分别取预测标签和真实标签
y_pred_df = df_val[['task', 'trial', 'start', 'end', 'y_pred_reaction']]
y_true_df = all_human_reactions_ch2[['task', 'trial', 'reaction_onset', 'reaction_offset']].loc[all_human_reactions_ch2['trial'].isin(val_trials)]

# 查看 y_pred_df 中 trial 列的所有唯一类别名称
print("y_pred_df trial 列唯一类别名称：")
print(y_pred_df['trial'].unique())

# 查看 y_true_df 中 trial 列的所有唯一类别名称
print("y_true_df trial 列唯一类别名称：")
print(y_true_df['trial'].unique())

tp, fp, total_reaction = evaluate_reaction(y_pred_df, y_true_df)

