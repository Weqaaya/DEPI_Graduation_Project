import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb  
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix, 
                           roc_curve, auc, precision_score, recall_score,
                           roc_auc_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from sklearn.utils import class_weight

# load data
df = pd.read_csv('df_cleaned_stroke.csv')
X = df.drop('stroke', axis=1)
y = df['stroke']

#  split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

# calculate the weights
classes = np.unique(y_train)
weights = class_weight.compute_class_weight(
    'balanced',
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

# apply SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

def train_xgboost_model(X_train, y_train, strategy_name, n_estimators=100, max_depth=3, scale_pos_weight=None):
    """دالة لتدريب وتسجيل نموذج XGBoost"""
    
    mlflow.set_experiment("stroke-prediction-v2")
    
    with mlflow.start_run(run_name=f"XGB_{strategy_name}"):
        
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False
        )
        
        # model train
        model.fit(X_train, y_train)
        
        # model predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # 
        mlflow.log_params({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': 0.1,
            'strategy': strategy_name,
            'scale_pos_weight': scale_pos_weight
        })
        
        # record metrics
        mlflow.log_metrics(metrics)
        
        # إنشاء توقيع النموذج
        signature = infer_signature(X_train, model.predict(X_train))
        
        #  record model
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        # حفظ الرسومات
        save_xgboost_plots(model, X_train, y_test, y_pred, y_proba, strategy_name)

def save_xgboost_plots(model, X_train, y_true, y_pred, y_proba, strategy_name):
    """حفظ رسومات XGBoost التوضيحية"""
    
    # confusion_matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Stroke', 'Stroke'],
               yticklabels=['No Stroke', 'Stroke'])
    plt.title(f'XGBoost Confusion Matrix ({strategy_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.close()
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'XGBoost ROC Curve ({strategy_name})')
    plt.legend(loc="lower right")
    mlflow.log_figure(plt.gcf(), "roc_curve.png")
    plt.close()
    
    # feature importance (XGBoost)
    if len(X_train.columns) <= 30:
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, max_num_features=15)  
        plt.title(f'XGBoost Feature Importance ({strategy_name})')
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()

# model parameters
n_estimators = 200
max_depth = 5
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])  # لموازنة الفئات

# model train
train_xgboost_model(X_train, y_train, "Baseline", n_estimators, max_depth)
train_xgboost_model(X_train, y_train, "Class_Weights", n_estimators, max_depth, scale_pos_weight)
train_xgboost_model(X_train_smote, y_train_smote, "SMOTE", n_estimators, max_depth)