"""_summary_
Module to plot some images of the model
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_roc_curve, plot_precision_recall_curve

# Function to show the distribution of classes
def distribution_classes(df):
    y_counts = df['is_fraud'].value_counts().to_frame('count_y_classes')
    y_counts['percentage_y_classes'] = y_counts['count_y_classes'] / y_counts['count_y_classes'].sum()
    print(y_counts)
    sns.countplot(x='is_fraud', data=df, palette='hls')
    return y_counts

# Fuction to plot the list with the Accuracy, Precision, Recall, F1 Score, MSE and MAE metrics of the model, using seaborn and save the image as png file
def plot_metrics(metric):
    metric = pd.DataFrame(metric)
    metric.columns = ['Accuracy','Precision','Recall','F1 Score','MSE','MAE']
    metric.plot(figsize=(16,12), title='Metrics for each manual training',marker='o',markersize=3)
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.xlabel('Manual Train')
    plt.ylabel('Metrics Value')
    plt.xticks(np.arange(0, 20, 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(title='Metrics')
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    plt.savefig('static/images/metrics.png',facecolor='white', transparent=False)
    
# Function to plot the confusion matrix
def confusion_matrix_plot(y_test,y_pred):
    cfm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cfm, display_labels = ['Not Fraud', 'Fraud'])
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12,12))
    cm_display.plot(ax=ax)
    ax.set_title('Confusion Matrix')
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    plt.savefig('static/images/confusion_matrix.png',facecolor='white', transparent=False)
    return cfm

# Function to plot the ROC curve
def roc_curve_plot(model,X_test,y_test):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(16,12))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_title('Receiver Operating Characteristic Curve')
    plot_roc_curve(model,X_test,y_test,ax=ax)
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    plt.savefig('static/images/roc_curve.png',facecolor='white', transparent=False)
    
#Function to plot the Precision - Recall curve
def precision_recall_curve_plot(model,X_test, y_test):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(16,12))
    ax.set_title('Precision-Recall Curve')
    plot_precision_recall_curve(model,X_test,y_test,ax=ax)
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    plt.savefig('static/images/precision_recall_curve.png',facecolor='white', transparent=False)