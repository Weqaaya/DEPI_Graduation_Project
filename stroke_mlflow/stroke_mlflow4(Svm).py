import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix, 
                           roc_curve, auc, precision_score, recall_score,
                           roc_auc_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

def train_svm_model(X_train, y_train, strategy_name, C=1.0, kernel='rbf', class_weight=None):
    """دالة لتدريب وتسجيل نموذج SVM"""
    
    mlflow.set_experiment("stroke-prediction-v2")
    
    with mlflow.start_run(run_name=f"SVM_{strategy_name}"):
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                C=C,
                kernel=kernel,
                class_weight=class_weight,
                probability=True,  # ضروري لحساب احتمالات التنبؤ
                random_state=42
            ))
        ])
        
        # model train
        model.fit(X_train, y_train)
        
        # prediction
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
        
        # تسجيل البارامترات
        mlflow.log_params({
            'C': C,
            'kernel': kernel,
            'strategy': strategy_name
        })
        
        # تسجيل المقاييس
        mlflow.log_metrics(metrics)
        
        # إنشاء توقيع النموذج
        signature = infer_signature(X_train, model.predict(X_train))
        
        # تسجيل النموذج
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        # حفظ الرسومات
        save_svm_plots(y_test, y_pred, y_proba, strategy_name)

def save_svm_plots(y_true, y_pred, y_proba, strategy_name):
    """حفظ رسومات SVM التوضيحية"""
    
    # confusion_matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Stroke', 'Stroke'],
               yticklabels=['No Stroke', 'Stroke'])
    plt.title(f'SVM Confusion Matrix ({strategy_name})')
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
    plt.title(f'SVM ROC Curve ({strategy_name})')
    plt.legend(loc="lower right")
    mlflow.log_figure(plt.gcf(), "roc_curve.png")
    plt.close()

# model parameters
C = 1.0  
kernel = 'rbf'  # ('linear', 'poly', 'rbf', 'sigmoid')

# model train
train_svm_model(X_train, y_train, "Baseline", C, kernel)
train_svm_model(X_train, y_train, "Class_Weights", C, kernel, class_weights)
train_svm_model(X_train_smote, y_train_smote, "SMOTE", C, kernel)