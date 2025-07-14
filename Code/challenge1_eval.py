import os
import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob

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


# Load Test Data
# Naming Pattern for Label Files
LABEL_FILE_PATTERNS = {
    "robot_errors": "challenge1_robot_error_labels_{task}_test.csv",
    "human_reactions_ch1": "challenge1_user_reaction_labels_{task}_test.csv",
    "human_reactions_ch2": "challenge2_user_reaction_labels_{task}_test.csv"
}


def fix_timestamp_format(ts):
    if isinstance(ts, str):
        parts = ts.split(':')
        if len(parts) == 4:
            return ':'.join(parts[:3]) + '.' + parts[3]
    return ts

"""Load all label files for a specific task and preprocess timestamps"""


def load_and_preprocess_labels(task):
    """Load all label files for a specific task and preprocess timestamps"""
    labels = {}    
    
    try:
        # Load robot error labels
        robot_errors = pd.read_csv(
            f"{BASE_PATH}/labels_test/challenge1_test/{LABEL_FILE_PATTERNS['robot_errors'].format(task=task)}"
        )
        robot_errors['error_onset'] = robot_errors['error_onset'].apply(fix_timestamp_format)
        robot_errors['error_onset'] = pd.to_timedelta(robot_errors['error_onset'])
        robot_errors['error_offset'] = robot_errors['error_offset'].apply(fix_timestamp_format)
        robot_errors['error_offset'] = pd.to_timedelta(robot_errors['error_offset'])
        robot_errors['trial'] = robot_errors['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['robot_errors'] = robot_errors        
        
        # Load human reaction labels
        human_reactions_ch1 = pd.read_csv(
            f"{BASE_PATH}/labels_test/challenge1_test/{LABEL_FILE_PATTERNS['human_reactions_ch1'].format(task=task)}"
        )
        human_reactions_ch1['reaction_onset'] = human_reactions_ch1['reaction_onset'].apply(fix_timestamp_format)
        human_reactions_ch1['reaction_onset'] = pd.to_timedelta(human_reactions_ch1['reaction_onset'])
        human_reactions_ch1['reaction_offset'] = human_reactions_ch1['reaction_offset'].apply(fix_timestamp_format)
        human_reactions_ch1['reaction_offset'] = pd.to_timedelta(human_reactions_ch1['reaction_offset'])
        human_reactions_ch1['trial'] = human_reactions_ch1['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['human_reactions_ch1'] = human_reactions_ch1        
        
        human_reactions_ch2 = pd.read_csv(
            f"{BASE_PATH}/labels_test/challenge2_test/{LABEL_FILE_PATTERNS['human_reactions_ch2'].format(task=task)}"
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


def evaluate_error(y_pred_df, y_true_df):
    """
    Compute classification metrics and count of detected error events.
    
    Parameters
    ----------
    y_pred_df : pd.DataFrame
        DataFrame must contain 'task', 'trial', 'start', 'end', 'y_pred_err',
    y_true_df : pd.DataFrame
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
    pos_pred = y_pred_df.loc[y_pred_df['y_pred_err'] == 1]
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
            pos_pred.loc[pos_pred['id'].isin(detected_err['id'])] = 1
            if len(detected_err) > 0: 
                tp = tp + 1

        # prediction is a false positive if it does not overlap with actual error 
        fp = len(pos_pred.loc[pos_pred['overlap_error'] == 0])
        
        print(f"True Positive: {tp} ({tp / total_errors * 100}/%)")
        print(f"False Positive: {fp}")

    return tp, fp, total_errors


def main():
    # Base Path to Data
    BASE_PATH = "./ACM-MM-ERR-HRI-2025-Dataset"

    # Predictions Path 
    # CSV file must contain columns 'task', 'trial', 'start', 'end', 'y_pred_err'. 
    y_pred_df = pd.read_csv(f"{BASE_PATH}/challenge1_test_preds.csv") 

    all_robot_errors = pd.DataFrame()
    all_human_reactions_ch1 = pd.DataFrame()
    all_human_reactions_ch2 = pd.DataFrame()

    tasks = test_data['task'].unique()
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

    y_pred_df['y_pred_err'] = pred_err
    val_trials = y_pred_df['trial'].astype(str).unique()

    y_pred_df = y_pred_df[['task', 'trial', 'start', 'end', 'y_pred_err']]
    y_true_df = all_robot_errors[['task', 'trial', 'error_onset', 'error_offset']].loc[all_robot_errors['trial'].isin(val_trials)]
    tp, fp, total_error = evaluate_error(y_pred_df, y_true_df)


if __name__ == "__main__":
    main()