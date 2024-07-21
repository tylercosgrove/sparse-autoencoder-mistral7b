import torch
from sae import SparseAutoencoder
import torch.nn.functional as F
from utils.activationsLoader import ActivationsLoader

"""

Below is a _very_ rough training loop. Most of the hyperparameters are chosen based on 'Scaling and evaluating sparse autoencoders' (Gao et al. 2024)
You might notice some weird GPU memory management (see line 83), a result of my limited compute budget (a single 4090 I have in my room). Periodically, I need move the model off the GPU and refresh the residual activations I use as training data for the SAE. This is done by running some tokens through the underlying model, details of which can be found in 'utils'.

"""

D_MODEL = 4096
D_HIDDEN = 131072
BATCH_SIZE = 256
scale = D_HIDDEN / (2**14)
lr = 2e-4 / scale**0.5

model = SparseAutoencoder(D_MODEL, D_HIDDEN)
model = model.to("cuda")
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, eps=6.25e-10, betas=(0.9, 0.999)
)

MISTRAL_MODEL_PATH = "PATH TO MISTRAL7b WEIGHTS"
actsLoader = ActivationsLoader(128, 512, MISTRAL_MODEL_PATH, target_layer=16)


def loss_fn(x, recons, auxk):
    mse_scale = 1.0 / 19.9776  # arbitrary mean of my input data
    auxk_coeff = 1.0 / 32.0

    mse_loss = mse_scale * F.mse_loss(recons, x)
    if auxk is not None:
        auxk_loss = auxk_coeff * F.mse_loss(auxk, x - recons).nan_to_num(0)
    else:
        auxk_loss = torch.tensor(0.0)
    return mse_loss, auxk_loss


scaler = torch.cuda.amp.GradScaler()
count = 0
while True:
    new_batch = actsLoader.new_data().to("cuda")
    single_batches = torch.split(new_batch, BATCH_SIZE)
    for batch in single_batches:
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            recons, auxk, num_dead = model(batch)
            mse_loss, auxk_loss = loss_fn(batch, recons, auxk)
            loss = mse_loss + auxk_loss

        loss = scaler.scale(loss)
        loss.backward()

        model.norm_weights()
        model.norm_grad()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if count % 1000 == 0:
            # Periodically save model
            torch.save(model.state_dict(), "sae.pth")

        if count % 50 == 0:
            # Logging...
            continue

        count += 1

    if actsLoader.needs_refresh():
        # If model is running out of activations to train on, need to generate more

        # move model off GPU to make room for Mistral 7b
        model = model.cpu()
        optimizer.zero_grad(set_to_none=True)
        scaler_state = scaler.state_dict()

        torch.cuda.empty_cache()
        actsLoader.refresh_data()

        # move everything back onto the GPU
        scaler = torch.cuda.amp.GradScaler()
        scaler.load_state_dict(scaler_state)
        model = model.to("cuda")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, eps=6.25e-10, betas=(0.9, 0.999)
        )
