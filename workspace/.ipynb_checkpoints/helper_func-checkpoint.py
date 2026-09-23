import torch
import numpy as np
import matplotlib.pyplot as plt

# define a function to evaluate the models and predict/evaluate
def evaluate_model(X_val_seq, y_val_seq, model, instance, ckpt, y_scaler, shape, trainer=None,
                   predict=False, evaluate=False):
    best_model = instance.RegressionLSTMTrainer.load_from_checkpoint(
    ckpt.best_model_path,
    n_features=shape, hidden_units=model.hidden_units, n_layers=model.n_layers,
    dropout=model.dropout)

    if (evaluate):
        best_model.eval()

        with torch.no_grad():
            preds = best_model(X_val_seq.to(best_model.device)).cpu().numpy()

        y_pred_lstm = y_scaler.inverse_transform(preds)
        y_true_lstm = y_scaler.inverse_transform(y_val_seq.numpy())
        
    elif (predict):
        # Generate predictions using trainer.predict
        preds_list = trainer.predict(best_model, datamodule=lstm_dm)
        preds = torch.cat(preds_list, dim=0).cpu().numpy()
        y_pred_lstm = y_scaler.inverse_transform(preds)
        y_true_lstm = y_scaler.inverse_transform(y_test_seq.numpy())

    return y_pred_lstm, y_true_lstm



    