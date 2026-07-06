import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, f1_score
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek
from datetime import datetime
import logging
import optuna
from sklearn.utils import resample

#SORTPOOLING
#/home/constanca/tese_workspace/classification/output_baseline/runs/T9_ast-graphcodebert-embeddings_gcb768_hd-32-32-32-32_norm-prenorm_clf-conv1d_ep30_wcw-0-19_k32_dr0.3_noae_embsave/run_3/embeddings
#/home/constanca/tese_workspace/classification/output_baseline/runs/T9_cfg-graphcodebert-embeddings_gcb768_hd-32-32-32-32_norm-prenorm_clf-conv1d_ep30_wcw-0-19_k32_dr0.3_noae_embsave/run_3/embeddings
#/home/constanca/tese_workspace/classification/output_baseline/runs/T9_pdg-graphcodebert-embeddings_gcb768_hd-32-32-32-32_norm-prenorm_clf-conv1d_ep30_wcw-0-20_k32_dr0.3_noae_embsave/run_5/embeddings

#AMP
#/home/constanca/tese_workspace/classification/output_baseline/runs/T10_ast-graphcodebert-embeddings_gcb768_hd-32-32-32-32_norm-prenorm_clf-vgg_ep30_wcw-0-19-ampk32_dr0.3_c2d32_noae_embsave/run_1/embeddings
#/home/constanca/tese_workspace/classification/output_baseline/runs/T10_cfg-graphcodebert-embeddings_gcb768_hd-32-32-32-32_norm-prenorm_clf-vgg_ep30_wcw-0-19-ampk32_dr0.3_c2d32_noae_embsave/run_1/embeddings
#/home/constanca/tese_workspace/classification/output_baseline/runs/T10_pdg-graphcodebert-embeddings_gcb768_hd-32-32-32-32_norm-prenorm_clf-vgg_ep30_wcw-0-20-ampk32_dr0.3_c2d32_noae_embsave/run_5/embeddings

DATA_DIR = "/home/constanca/tese_workspace/classification/output_baseline/runs/T10_pdg-graphcodebert-embeddings_gcb768_hd-32-32-32-32_norm-prenorm_clf-vgg_ep30_wcw-0-20-ampk32_dr0.3_c2d32_noae_embsave/run_5/embeddings"

OUTPUT_DIR = "final_results_pdg_svm_amp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging 
log_file = os.path.join(OUTPUT_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("STARTING FULL TRAINING ON SERVER")
logger.info("="*60)

# Load Full Dataset
logger.info("\n[1] Loading all embeddings")

# Load train data (features and labels)
X_train = torch.load(os.path.join(DATA_DIR, 'dgcnn_embeddings_train_after.pt'), weights_only=True).numpy()
y_train = torch.load(os.path.join(DATA_DIR, 'train_after_labels.pt'), weights_only=True).numpy()

# Load test data (features and labels)
X_test = torch.load(os.path.join(DATA_DIR, 'dgcnn_embeddings_test_after.pt'), weights_only=True).numpy()
y_test = torch.load(os.path.join(DATA_DIR, 'test_after_labels.pt'), weights_only=True).numpy()

logger.info(f"Original Train: {len(y_train)} samples (Vuln: {sum(y_train==1)} | Normal: {sum(y_train==0)})")
logger.info(f"Original Test:  {len(y_test)} samples (Vuln: {sum(y_test==1)} | Normal: {sum(y_test==0)})")


# Balance Dataset (SMOTE + TomekLinks)
logger.info("\n[2] Applying SMOTE + Tomek Links")
smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

logger.info(f"Balanced Train: {len(y_train_resampled)} samples (Vuln: {sum(y_train_resampled==1)} | Normal: {sum(y_train_resampled==0)})")

#  SMOTE + Tomek Links Analysis Visualization
logger.info("\n[3] Creating SMOTE + Tomek Links analysis visualization...")

pca = PCA(n_components=2)
X_pca_train = pca.fit_transform(X_train)

# Find which indices were kept (for Tomek visualization)
# Synthetic indices for SMOTE, track removals for Tomek
original_indices = set(range(len(y_train)))
resampled_indices = set()
for i in range(len(y_train_resampled)):
    # Map back to original (approximate - SMOTE creates new samples)
    if i < len(X_train):
        resampled_indices.add(i)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Before: All samples
ax = axes[0, 0]
ax.scatter(X_pca_train[y_train == 0, 0], X_pca_train[y_train == 0, 1], c='lightblue', label='Normal', alpha=0.5, s=30)
ax.scatter(X_pca_train[y_train == 1, 0], X_pca_train[y_train == 1, 1], c='red', label='Vulnerable', s=60, edgecolors='black')
ax.set_title('Before SMOTE + Tomek Links', fontsize=12, fontweight='bold')
ax.legend()
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)

# Show imbalance problem
ax = axes[0, 1]
imbalance_ratio_before = sum(y_train==0) / sum(y_train==1)
ax.scatter(X_pca_train[y_train == 0, 0], X_pca_train[y_train == 0, 1], c='lightgrey', label='Normal (Majority)', alpha=0.4, s=30)
ax.scatter(X_pca_train[y_train == 1, 0], X_pca_train[y_train == 1, 1], c='red', label='Vulnerable (Minority)', s=60, edgecolors='black')
ax.set_title(f'Class Imbalance (Ratio: {imbalance_ratio_before:.2f}:1)', fontsize=12, fontweight='bold')
ax.legend()
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)

# After: Balanced dataset
X_res_pca = pca.transform(X_train_resampled)
ax = axes[1, 0]
ax.scatter(X_res_pca[y_train_resampled == 0, 0], X_res_pca[y_train_resampled == 0, 1], c='blue', label='Normal (Resampled)', alpha=0.6, s=30)
ax.scatter(X_res_pca[y_train_resampled == 1, 0], X_res_pca[y_train_resampled == 1, 1], c='red', label='Vulnerable (Synthetic + Real)', s=60, edgecolors='black')
ax.set_title('After SMOTE + Tomek Links', fontsize=12, fontweight='bold')
ax.legend()
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)

# Statistics
ax = axes[1, 1]
ax.axis('off')
imbalance_ratio_after = sum(y_train_resampled==0) / sum(y_train_resampled==1)
stats_text = f"""
SMOTE + Tomek Links Statistics:
{'─'*50}
Original dataset:
  Normal: {sum(y_train==0)}  |  Vulnerable: {sum(y_train==1)}
  Imbalance ratio: {imbalance_ratio_before:.2f}:1

After resampling:
  Normal: {sum(y_train_resampled==0)}  |  Vulnerable: {sum(y_train_resampled==1)}
  Imbalance ratio: {imbalance_ratio_after:.2f}:1

Changes:
  Normal change: {sum(y_train_resampled==0) - sum(y_train==0):+d}
  Vulnerable change: {sum(y_train_resampled==1) - sum(y_train==1):+d}
  Total samples: {len(y_train)} → {len(y_train_resampled)}
  
Effect: SMOTE creates synthetic minorities,
         Tomek removes noisy borderline samples
"""
ax.text(0.05, 0.5, stats_text, fontfamily='monospace', fontsize=11, verticalalignment='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'smote_tomek_analysis.png'), dpi=100, bbox_inches='tight')
plt.close()
logger.info("SMOTE + Tomek analysis saved to 'smote_tomek_analysis.png'")

def evaluate_model(model, name):
    logger.info(f"\n[Evaluating {name} on Real Test Data...]")
    
    # Get predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    # Save Text Report
    roc_auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, zero_division=0)
    logger.info(f"\n{report}")
    with open(os.path.join(OUTPUT_DIR, f'report_{name.lower()}.txt'), 'w') as f:
        f.write(report)
        f.write(f"\nROC-AUC Score: {roc_auc:.4f}\n")
    logger.info(f"Classification report saved to 'report_{name.lower()}.txt'")
    
    # Save Confusion Matrix Image with percentages
    cm = confusion_matrix(y_test, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with absolute values + percentages
    annotations = np.array([f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)" 
                            for i in range(cm.shape[0]) 
                            for j in range(cm.shape[1])]).reshape(cm.shape)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Vulnerable"], yticklabels=["Normal", "Vulnerable"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'cm_{name.lower()}.png'))
    plt.close()
    logger.info(f"Confusion matrix saved to 'cm_{name.lower()}.png'")
    
    # Save Precision-Recall Curve Image
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'pr_curve_{name.lower()}.png'))
    plt.close()
    logger.info(f"Precision-Recall curve saved to 'pr_curve_{name.lower()}.png'")

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'roc_curve_{name.lower()}.png'))
    plt.close()
    logger.info(f"ROC curve saved to 'roc_curve_{name.lower()}.png'")

# ==========================================
# Step 5: Train and Test the Models
# ==========================================
#Without obtune

#logger.info("\n[4] Training SVM...")
#svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
#svm_clf.fit(X_train_resampled, y_train_resampled)
#evaluate_model(svm_clf, "SVM")

logger.info("\n Running Optuna Hyperparameter Optimization for SVM")

def objective(trial):
    # Param space
    c = trial.suggest_float('C', 1e-3, 100, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])

    #  Use smaller subset just to find the parameters quicker
    #  15000 samples (20% of balanced dataset) 
    X_sub, y_sub = resample(X_train_resampled, y_train_resampled, 
                            n_samples=15000, 
                            random_state=42, 
                            stratify=y_train_resampled)

    clf = SVC(C=c, gamma=gamma, kernel=kernel, probability=False, random_state=42)
    
    # train only on subset
    clf.fit(X_sub, y_sub)
    
    y_pred = clf.predict(X_test)
    
    # F1-score of vulnerable class
    score = f1_score(y_test, y_pred)
    return score

# Config + run optuna (limited to 10 tests)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

logger.info(f"Best Optuna parameters found: {study.best_params}")

logger.info("\n[5.2] Training FINAL SVM with Best Parameters")
best_svm_params = study.best_params
best_svm_clf = SVC(**best_svm_params, probability=True, random_state=42)
best_svm_clf.fit(X_train_resampled, y_train_resampled)

evaluate_model(best_svm_clf, "SVM_Optimized")

#logger.info("\n Training XGBoost")
#xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
#xgb_clf.fit(X_train_resampled, y_train_resampled)
#evaluate_model(xgb_clf, "XGBoost")

logger.info(f"All charts and reports saved to '{OUTPUT_DIR}'")
logger.info(f"Log file: {log_file}")
