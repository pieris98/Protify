import torch
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    confusion_matrix,
    hamming_loss,
    accuracy_score,
    make_scorer,
)
from scipy.stats import pearsonr, spearmanr
from transformers import EvalPrediction


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def regression_scorer():
    def dual_score(y_true, y_pred):
        return spearmanr(y_true, y_pred).correlation * r2_score(y_true, y_pred)
    return dual_score


def classification_scorer():
    def mcc_scorer(y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)
    return mcc_scorer


def get_classification_scorer():
    return make_scorer(classification_scorer(), greater_is_better=True)


def get_regression_scorer():
    return make_scorer(regression_scorer(), greater_is_better=True)


def calculate_max_metrics(ss: torch.Tensor, labels: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate precision, recall and F1 metrics for binary classification at a specific cutoff threshold.

    Args:
        ss: Prediction scores tensor, typically between -1 and 1
        labels: Ground truth binary labels tensor (0 or 1)
        cutoff: Classification threshold value

    Returns:
        Tuple containing:
            - F1 score (torch.Tensor)
            - Precision score (torch.Tensor) 
            - Recall score (torch.Tensor)

    Note:
        - Input tensors are converted to float type
        - Handles division by zero cases by returning 0
        - Uses standard binary classification metrics formulas:
            - Precision = TP / (TP + FP)
            - Recall = TP / (TP + FN)
            - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    ss, labels = ss.float(), labels.float()
    tp = torch.sum((ss >= cutoff) & (labels == 1.0))
    fp = torch.sum((ss >= cutoff) & (labels == 0.0))
    fn = torch.sum((ss < cutoff) & (labels == 1.0))
    precision_denominator = tp + fp
    precision = torch.where(precision_denominator != 0, tp / precision_denominator, torch.tensor(0.0))
    recall_denominator = tp + fn
    recall = torch.where(recall_denominator != 0, tp / recall_denominator, torch.tensor(0.0))
    f1 = torch.where((precision + recall) != 0, (2 * precision * recall) / (precision + recall), torch.tensor(0.0))
    return f1, precision, recall


def max_metrics(ss: torch.Tensor, labels: torch.Tensor, increment: float = 0.01) -> tuple[float, float, float, float]:
    """
    Find optimal classification metrics by scanning different cutoff thresholds.
    Optimized version that vectorizes calculations across all cutoffs.

    Args:
        ss: Prediction scores tensor, typically between -1 and 1
        labels: Ground truth binary labels tensor (0 or 1)
        increment: Step size for scanning cutoff values, defaults to 0.01

    Returns:
        Tuple containing:
            - Maximum F1 score (float)
            - Maximum precision score (float)
            - Maximum recall score (float) 
            - Optimal cutoff threshold (float)

    Note:
        - Input scores are clamped to [-1, 1] range
        - Handles edge case where all scores are >= 1
        - Scans cutoff values from min score to 1 in increments
        - Handles NaN F1 scores by replacing with -1 before finding max
        - Returns metrics at the threshold that maximizes F1 score
        - Optimized to compute metrics for all cutoffs in parallel using vectorization
    """
    ss = torch.clamp(ss, -1.0, 1.0)
    min_val = ss.min().item()
    max_val = 1
    if min_val >= max_val:
        min_val = 0
    
    # Convert to float and ensure labels are binary
    ss = ss.float()
    labels = labels.float()
    
    # Create cutoff tensor
    cutoffs = torch.arange(min_val, max_val, increment, device=ss.device, dtype=ss.dtype)
    n_cutoffs = len(cutoffs)
    
    if n_cutoffs == 0:
        # Edge case: no cutoffs to test
        return 0.0, 0.0, 0.0, min_val
    
    # Vectorize across all cutoffs: shape (n_cutoffs, n_samples)
    # Expand cutoffs to (n_cutoffs, 1) and ss to (1, n_samples) for broadcasting
    ss_expanded = ss.unsqueeze(0)  # (1, n_samples)
    cutoffs_expanded = cutoffs.unsqueeze(1)  # (n_cutoffs, 1)
    labels_expanded = labels.unsqueeze(0)  # (1, n_samples)
    
    # Compute predictions for all cutoffs at once: (n_cutoffs, n_samples)
    predictions = (ss_expanded >= cutoffs_expanded).float()
    
    # Compute TP, FP, FN for all cutoffs simultaneously
    # TP: predicted positive and label positive
    tp = torch.sum(predictions * labels_expanded, dim=1)  # (n_cutoffs,)
    # FP: predicted positive but label negative
    fp = torch.sum(predictions * (1.0 - labels_expanded), dim=1)  # (n_cutoffs,)
    # FN: predicted negative but label positive
    fn = torch.sum((1.0 - predictions) * labels_expanded, dim=1)  # (n_cutoffs,)
    
    # Compute precision, recall, F1 for all cutoffs
    precision_denominator = tp + fp
    precision = torch.where(precision_denominator != 0, tp / precision_denominator, torch.tensor(0.0, device=ss.device))
    
    recall_denominator = tp + fn
    recall = torch.where(recall_denominator != 0, tp / recall_denominator, torch.tensor(0.0, device=ss.device))
    
    # Compute F1 scores
    f1_denominator = precision + recall
    f1s = torch.where(f1_denominator != 0, (2 * precision * recall) / f1_denominator, torch.tensor(0.0, device=ss.device))
    
    # Handle NaN values by replacing with -1
    valid_f1s = torch.where(torch.isnan(f1s), torch.tensor(-1.0, device=ss.device), f1s)
    max_index = torch.argmax(valid_f1s)
    
    return f1s[max_index].item(), precision[max_index].item(), recall[max_index].item(), cutoffs[max_index].item()


def compute_single_label_classification_metrics(p: EvalPrediction) -> dict[str, float]:
    """
    Compute comprehensive metrics for single-label classification tasks.

    Args:
        p: EvalPrediction object containing model predictions and ground truth labels

    Returns:
        Dictionary with the following metrics (all rounded to 5 decimal places):
            - f1: F1 score (weighted average)
            - precision: Precision score (weighted average)
            - recall: Recall score (weighted average)
            - accuracy: Overall accuracy
            - mcc: Matthews Correlation Coefficient
            - roc_auc: Area Under ROC Curve (weighted average)
            - pr_auc: Area Under Precision-Recall Curve (weighted average)

    Note:
        - Handles both binary and multi-class cases
        - For binary case: uses 0.5 threshold on probabilities
        - For multi-class: uses argmax for class prediction
        - Prints confusion matrix for detailed error analysis
        - Uses weighted averaging for multi-class metrics
        - Handles AUC calculation for both binary and multi-class cases
    """
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    y_pred = logits.argmax(axis=-1).flatten()
    y_true = labels.flatten().astype(int)
    probs = softmax(logits)

    # Calculate ROC AUC
    try:
        # binary
        if len(np.unique(y_true)) == 2:
            # For binary, use probabilities of positive class
            roc_auc = roc_auc_score(y_true, probs[:, 1] if probs.shape[1] == 2 else probs.flatten())
        else:
            # multi-class
            n_classes = probs.shape[1]
            y_true_onehot = np.eye(n_classes)[y_true]
            roc_auc = roc_auc_score(y_true_onehot, probs, multi_class='ovr', average='weighted')
    except:
        roc_auc = -100.0
    
    # Calculate PR AUC (true AUC of Precision-Recall curve)
    try:
        # binary
        if len(np.unique(y_true)) == 2:
            # For binary, use probabilities of positive class
            y_scores = probs[:, 1] if probs.shape[1] == 2 else probs.flatten()
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
        else:
            # multi-class: compute PR AUC for each class and average
            n_classes = probs.shape[1]
            y_true_onehot = np.eye(n_classes)[y_true]
            pr_aucs = []
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], probs[:, i])
                pr_aucs.append(auc(recall, precision))
            # Weighted average by class support
            class_weights = y_true_onehot.sum(axis=0)
            if class_weights.sum() > 0:
                pr_auc = np.average(pr_aucs, weights=class_weights)
            else:
                pr_auc = np.mean(pr_aucs) if pr_aucs else -100.0
    except:
        pr_auc = -100.0
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        'f1': round(f1, 5),
        'precision': round(precision, 5),
        'recall': round(recall, 5),
        'accuracy': round(accuracy, 5),
        'mcc': round(mcc, 5),
        'roc_auc': round(roc_auc, 5),
        'pr_auc': round(pr_auc, 5)
    }


def compute_tokenwise_classification_metrics(p: EvalPrediction) -> dict[str, float]:
    """
    Compute metrics for token-level classification tasks.

    Args:
        p: EvalPrediction object containing model predictions and ground truth labels

    Returns:
        Dictionary containing the following metrics (all rounded to 5 decimal places):
            - accuracy: Overall accuracy
            - f1: F1 score (macro average)
            - precision: Precision score (macro average)
            - recall: Recall score (macro average)
            - mcc: Matthews Correlation Coefficient
            - roc_auc: Area Under ROC Curve (weighted average)
            - pr_auc: Area Under Precision-Recall Curve (weighted average)

    Note:
        - Handles special token padding (-100) by filtering before metric calculation
        - Uses macro averaging for multi-class metrics
        - Converts predictions to class labels using argmax
        - Handles AUC calculation for both binary and multi-class cases
    """
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    # Compute f1 score
    y_pred = logits.argmax(axis=-1).flatten()
    y_true = labels.flatten()
    valid_indices = y_true != -100
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Calculate probabilities for AUC metrics
    probs = softmax(logits)
    probs = probs.reshape(-1, probs.shape[-1])  # Flatten to (n_samples, n_classes)
    probs = probs[valid_indices]  # Filter by valid indices
    
    # Calculate ROC AUC
    try:
        # binary
        if len(np.unique(y_true)) == 2:
            # For binary, use probabilities of positive class
            roc_auc = roc_auc_score(y_true, probs[:, 1] if probs.shape[1] == 2 else probs.flatten())
        else:
            # multi-class
            n_classes = probs.shape[1]
            y_true_onehot = np.eye(n_classes)[y_true]
            roc_auc = roc_auc_score(y_true_onehot, probs, multi_class='ovr', average='weighted')
    except:
        roc_auc = -100.0
    
    # Calculate PR AUC (true AUC of Precision-Recall curve)
    try:
        # binary
        if len(np.unique(y_true)) == 2:
            # For binary, use probabilities of positive class
            y_scores = probs[:, 1] if probs.shape[1] == 2 else probs.flatten()
            pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(pr_recall, pr_precision)
        else:
            # multi-class: compute PR AUC for each class and average
            n_classes = probs.shape[1]
            y_true_onehot = np.eye(n_classes)[y_true]
            pr_aucs = []
            for i in range(n_classes):
                pr_precision, pr_recall, _ = precision_recall_curve(y_true_onehot[:, i], probs[:, i])
                pr_aucs.append(auc(pr_recall, pr_precision))
            # Weighted average by class support
            class_weights = y_true_onehot.sum(axis=0)
            if class_weights.sum() > 0:
                pr_auc = np.average(pr_aucs, weights=class_weights)
            else:
                pr_auc = np.mean(pr_aucs) if pr_aucs else -100.0
    except:
        pr_auc = -100.0
    
    return {
        'accuracy': round(accuracy, 5),
        'f1': round(f1, 5),
        'precision': round(precision, 5),
        'recall': round(recall, 5),
        'mcc': round(mcc, 5),
        'roc_auc': round(roc_auc, 5),
        'pr_auc': round(pr_auc, 5)
    }


def compute_multi_label_classification_metrics(p: EvalPrediction) -> dict[str, float]:
    """
    Compute comprehensive metrics for multi-label classification tasks.

    Args:
        p: EvalPrediction object containing model predictions and ground truth labels

    Returns:
        Dictionary containing the following metrics (all rounded to 5 decimal places):
            - accuracy: Overall accuracy
            - f1: F1 score (optimized across thresholds)
            - precision: Precision score (at optimal threshold)
            - recall: Recall score (at optimal threshold)
            - hamming_loss: Proportion of wrong labels
            - threshold: Optimal classification threshold
            - mcc: Matthews Correlation Coefficient
            - roc_auc: Area Under ROC Curve (macro average)
            - pr_auc: Area Under Precision-Recall Curve (macro average)

    Note:
        - Converts inputs to PyTorch tensors
        - Applies softmax to raw predictions
        - Uses threshold optimization for best F1 score
        - Handles multi-class ROC AUC using one-vs-rest
        - All metrics are computed on flattened predictions
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    # Convert to tensors efficiently, avoiding unnecessary numpy round-trip
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(labels, torch.Tensor):
        y_true = torch.tensor(labels, dtype=torch.int)
    else:
        y_true = labels.int()

    probs = preds.softmax(dim=-1)
    y_pred = (probs > 0.5).int()

    # Flatten before max_metrics for efficiency - max_metrics expects flattened tensors
    probs_flat = probs.flatten()
    y_true_flat = y_true.flatten()
    f1, prec, recall, thres = max_metrics(probs_flat, y_true_flat)
    
    y_pred, y_true = y_pred.flatten().numpy(), y_true.flatten().numpy()
    probs = probs.flatten().numpy()
    
    accuracy = accuracy_score(y_pred, y_true)
    hamming = hamming_loss(y_pred, y_true)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Calculate ROC AUC for multilabel case
    try:
        roc_auc = roc_auc_score(y_true, probs, average='macro')
    except ValueError:
        # Fallback in case of invalid predictions
        roc_auc = -100.0
    
    # Calculate PR AUC for multilabel case (true AUC of Precision-Recall curve)
    try:
        # For flattened multi-label, compute PR AUC directly
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(pr_recall, pr_precision)
    except ValueError:
        # Fallback in case of invalid predictions
        pr_auc = -100.0

    return {
        'accuracy': round(accuracy, 5),
        'f1': round(f1, 5),
        'precision': round(prec, 5),
        'recall': round(recall, 5),
        'hamming_loss': round(hamming, 5),
        'threshold': round(thres, 5),
        'mcc': round(mcc, 5),
        'roc_auc': round(roc_auc, 5),
        'pr_auc': round(pr_auc, 5)
    }


def compute_regression_metrics(p: EvalPrediction) -> dict[str, float]:
    """
    Compute comprehensive metrics for regression tasks.

    Args:
        p: EvalPrediction object containing model predictions and ground truth values

    Returns:
        Dictionary containing the following metrics (all rounded to 5 decimal places):
            - r_squared: Coefficient of determination (R²)
            - spearman_rho: Spearman rank correlation coefficient
            - spear_pval: P-value for Spearman correlation
            - pearson_rho: Pearson correlation coefficient
            - pear_pval: P-value for Pearson correlation
            - mse: Mean Squared Error
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error

    Note:
        - Handles both raw predictions and tuple predictions
        - Flattens inputs to 1D arrays
        - Includes both correlation and error metrics
        - P-values indicate statistical significance of correlations
        - RMSE is calculated as square root of MSE
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    y_pred = np.array(preds).flatten()
    y_true = np.array(labels).flatten()

    if np.isnan(y_true).any():
        print("y_true Nans were cast to 0")
        y_true = np.where(np.isnan(y_true), 0, y_true)
    if np.isnan(y_pred).any():
        print("y_pred Nans were cast to 0")
        y_pred = np.where(np.isnan(y_pred), 0, y_pred)

    try:
        spearman_rho, spear_pval = spearmanr(y_pred, y_true)
        pearson_rho, pear_pval = pearsonr(y_pred, y_true)
    except:
        spearman_rho = -100.0
        spear_pval = -100.0
        pearson_rho = -100.0
        pear_pval = -100.0

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        'r_squared': round(r2, 5),
        'spearman_rho': round(spearman_rho, 5),
        'spear_pval': round(spear_pval, 5),
        'pearson_rho': round(pearson_rho, 5),
        'pear_pval': round(pear_pval, 5),
        'mse': round(mse, 5),
        'mae': round(mae, 5),
        'rmse': round(rmse, 5),
    }


def compute_tokenwise_regression_metrics(p: EvalPrediction) -> dict[str, float]:
    """
    Compute regression metrics tokenwise, ignoring label positions equal to -100.

    Compatible with HF Trainer `compute_metrics` API.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    y_pred = np.array(preds)
    y_true = np.array(labels)

    # If predictions have an extra trailing dim of size 1, squeeze it
    if y_pred.ndim == y_true.ndim + 1 and y_pred.shape[-1] == 1:
        y_pred = np.squeeze(y_pred, axis=-1)

    # Flatten to align and filter by valid positions (labels != -100)
    valid_mask = (y_true != -100)
    y_true = y_true[valid_mask].astype(float)
    y_pred = y_pred[valid_mask].astype(float)

    if y_true.size == 0:
        return {
            'r_squared': -100.0,
            'spearman_rho': -100.0,
            'spear_pval': -100.0,
            'pearson_rho': -100.0,
            'pear_pval': -100.0,
            'mse': -100.0,
            'mae': -100.0,
            'rmse': -100.0,
        }

    if np.isnan(y_true).any():
        print("y_true Nans were cast to 0")
        y_true = np.where(np.isnan(y_true), 0, y_true)
    if np.isnan(y_pred).any():
        print("y_pred Nans were cast to 0")
        y_pred = np.where(np.isnan(y_pred), 0, y_pred)

    try:
        spearman_rho, spear_pval = spearmanr(y_pred, y_true)
        pearson_rho, pear_pval = pearsonr(y_pred, y_true)
    except Exception:
        spearman_rho = -100.0
        spear_pval = -100.0
        pearson_rho = -100.0
        pear_pval = -100.0

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        'r_squared': round(float(r2), 5),
        'spearman_rho': round(float(spearman_rho), 5),
        'spear_pval': round(float(spear_pval), 5),
        'pearson_rho': round(float(pearson_rho), 5),
        'pear_pval': round(float(pear_pval), 5),
        'mse': round(float(mse), 5),
        'mae': round(float(mae), 5),
        'rmse': round(float(rmse), 5),
    }


def get_compute_metrics(task_type: str, tokenwise: bool = False):
    if task_type == 'singlelabel':
        compute_metrics = compute_single_label_classification_metrics
    elif task_type == 'multilabel':
        compute_metrics = compute_multi_label_classification_metrics
    elif task_type == 'sigmoid_regression':
        # Treat sigmoid_regression like regression for metrics
        compute_metrics = compute_tokenwise_regression_metrics if tokenwise else compute_regression_metrics
    elif not task_type == 'regression' and tokenwise:
        compute_metrics = compute_tokenwise_classification_metrics
    elif task_type == 'regression' and not tokenwise:
        compute_metrics = compute_regression_metrics
    elif task_type == 'regression' and tokenwise:
        compute_metrics = compute_tokenwise_regression_metrics
    else:
        raise ValueError(f'Task type {task_type} not supported')
    return compute_metrics


if __name__ == "__main__":
    scorer = get_classification_scorer()
    print(scorer.__name__)
