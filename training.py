import torch
from tqdm import tqdm

from config import TrainConfig
from utils import ev_mean, mse, t_rescaling


def train(circuit, thetas, dataloader, config: TrainConfig, device="cpu"):
    """
    Train the PQC energy function Q_theta(x, y).

    Returns:
        (trained_thetas, loss_history)
    """
    opt = torch.optim.Adam([thetas], lr=config.lr)
    loss_history = []

    for epoch in tqdm(range(config.n_epochs), desc="Training"):
        epoch_total = 0.0

        for x_batch, y_batch in dataloader:
            opt.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_t = x_batch.transpose(0, 1)
            y_t = y_batch.transpose(0, 1)

            t = 2 * torch.rand(len(x_batch), device=device) - 0.5

            dists = mse(x_batch, y_batch)
            t_res = t_rescaling(t, dists, config.r_rate)
            cost_value = torch.sin(torch.pi / 2 * t_res)

            interp_t = y_t * t + x_t * (1 - t)
            mid_term = (ev_mean(circuit(thetas, x_t, interp_t)) - cost_value) ** 2
            min_term = (ev_mean(circuit(thetas, x_t, y_t)) - 1) ** 2

            losses = config.alpha * min_term + config.beta * mid_term
            batch_loss = torch.mean(losses)
            batch_loss.backward()
            opt.step()

            epoch_total += batch_loss.item()

        epoch_mean = epoch_total / len(dataloader)
        loss_history.append(epoch_mean)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{config.n_epochs}, Loss: {epoch_mean:.6f}")

    return thetas, loss_history
