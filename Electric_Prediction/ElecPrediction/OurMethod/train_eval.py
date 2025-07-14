import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, train_loader, criterion, optimizer, scheduler=None, epochs=20):

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        alpha_vals = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred, _, _, alpha = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            alpha_vals.append(alpha.detach().cpu().numpy())

        alpha_vals = np.concatenate(alpha_vals)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
        print(f"alpha: {alpha_vals.mean():.4f}")

        if scheduler is not None:
            scheduler.step(total_loss / len(train_loader))

def evaluate_model(model, data_loader, label_scaler, delta=0.2):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            # out, _, _ = model(x).cpu().numpy()
            out, _, _, _ = model(x)
            preds.append(out)
            targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    preds_real = label_scaler.inverse_transform(preds)
    targets_real = label_scaler.inverse_transform(targets)

    mse = np.mean((preds_real - targets_real) ** 2)
    mae = np.mean(np.abs(preds_real - targets_real))
    std = np.std(preds_real - targets_real)
    epsilon = 1e-6
    acc = np.mean((np.abs(preds_real - targets_real) / (np.abs(targets_real) + epsilon)) < delta)

    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, STD: {std:.4f}, Accuracy (<{delta*100:.0f}% error): {acc*100:.2f}%")
    return preds_real.flatten(), targets_real.flatten()