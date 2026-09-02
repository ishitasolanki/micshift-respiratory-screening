"""Small CNN classifier over log-mel spectrograms.

Sized to the dataset (order 10k clips), not to fashion. A large pretrained audio
transformer would overfit this corpus and could not run on the CPU-only hardware
the deployment target implies (NFR-1).
"""
import numpy as np
import torch
import torch.nn as nn

import config


class CoughCNN(nn.Module):
    def __init__(self, n_mels=config.N_MELS, n_classes=2):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        # Global pooling over time makes the model accept any clip length, so a
        # demo upload does not have to match the training clip duration.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.head(self.pool(self.features(x)).flatten(1))


def predict(model, feats, device="cpu", batch=64):
    """Class probabilities for a batch of features.

    Batched deliberately: a full-split forward pass allocates the first conv
    layer's activations for every sample at once (~800 MB for 500 clips), which
    silently kills the process on a normal machine.
    """
    model.eval()
    x = torch.as_tensor(np.asarray(feats), dtype=torch.float32, device=device)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(torch.softmax(model(x[i:i + batch]), dim=1).cpu().numpy())
    return np.concatenate(out)


def _demo():
    torch.manual_seed(config.SEED)
    m = CoughCNN()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  parameters: {n_params:,}")
    assert n_params < 5e6, "model too large for this dataset size"

    x = torch.randn(4, config.N_MELS, 401)
    out = m(x)
    assert out.shape == (4, 2), f"bad output shape {out.shape}"
    print(f"  forward {tuple(x.shape)} -> {tuple(out.shape)}")

    # Variable clip length must work, or the demo can only accept 4 s uploads.
    assert m(torch.randn(2, config.N_MELS, 137)).shape == (2, 2)
    print("  variable-length input OK")

    p = predict(m, np.random.randn(3, config.N_MELS, 401).astype(np.float32))
    assert p.shape == (3, 2) and np.allclose(p.sum(1), 1)
    print(f"  probabilities sum to 1, shape {p.shape}")

    # The model must be able to fit a trivial signal -- catches dead gradients,
    # bad init, or a detached graph before a real training run wastes an hour.
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xb = torch.randn(16, config.N_MELS, 100)
    yb = (xb.mean(dim=(1, 2)) > 0).long()
    xb[yb == 1] += 3.0
    for _ in range(60):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(xb), yb)
        loss.backward()
        opt.step()
    acc = (m(xb).argmax(1) == yb).float().mean().item()
    print(f"  overfit sanity check: loss={loss.item():.3f} acc={acc:.2f}")
    assert acc > 0.9, "model cannot fit a trivial separable signal"

    print("\nmodel OK - shapes, lengths, gradients all sound.")


if __name__ == "__main__":
    _demo()
