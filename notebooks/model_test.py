import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from model import RichFeedbackSystem 
def compute_metrics(predictions, ground_truths):
    """
    Computes PLCC, SRCC, and MSE between predictions and ground truths.
    """
    # Pearson Linear Correlation Coefficient
    plcc = pearsonr(predictions, ground_truths)[0]
    # Spearman Rank Correlation Coefficient
    srcc = spearmanr(predictions, ground_truths)[0]
    # Mean Squared Error
    mse = mean_squared_error(ground_truths, predictions)
    
    return plcc, srcc, mse

def evaluate_model(model, data_loader, device):
    """
    Evaluates the model on the given data loader and computes metrics.
    """
    model.eval()
    predictions = {"plausibility": [], "alignment": [], "aesthetics": [], "overall_quality": []}
    ground_truths = {"plausibility": [], "alignment": [], "aesthetics": [], "overall_quality": []}
    
    with torch.no_grad():
        for batch in data_loader:
            images, input_ids, attention_mask, scores = batch
            images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)
            
            # Model forward pass
            output_scores, _ = model(images, input_ids, attention_mask)
            
            # Collect predictions and ground truths for each metric
            for metric in predictions.keys():
                predictions[metric].extend(output_scores[metric].squeeze().cpu().numpy())
                ground_truths[metric].extend(scores[metric].cpu().numpy())

    metrics = {}
    for metric in predictions.keys():
        plcc, srcc, mse = compute_metrics(np.array(predictions[metric]), np.array(ground_truths[metric]))
        metrics[metric] = {"PLCC": plcc, "SRCC": srcc, "MSE": mse}

    return metrics

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RichFeedbackSystem().to(device)
    model.load_state_dict(torch.load("path_to_trained_model.pth"))
    
    from dataset import get_data_loader  # Assuming dataset.py contains your data loader
    eval_loader = get_data_loader("path_to_eval_tfrecord.tfrecord", batch_size=16, shuffle=False)
    
    metrics = evaluate_model(model, eval_loader, device)
    
    for metric, values in metrics.items():
        print(f"Metric: {metric}")
        print(f"  PLCC: {values['PLCC']:.3f}")
        print(f"  SRCC: {values['SRCC']:.3f}")
        print(f"  MSE: {values['MSE']:.3f}")