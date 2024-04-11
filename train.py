import json
import modules
import time
import torch as th
import torch.utils.data as thud
import torchvision as tv
import torchvision.transforms.functional as tvf
from typing import Optional
import utils


def main(batch_size: int,
         patch_size: int,
         epochs: int,
         checkpoint_path: str,
         config_path: str,
         save_path: str,
         data_path: str,
         num_workers: int,
         seed: Optional[int]) -> None:

    transform = tv.transforms.Compose([
        tv.transforms.RandomCrop(patch_size),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomVerticalFlip(),
        tv.transforms.ToTensor(),
    ])

    if seed is not None:
        th.manual_seed(10)

    dataset = tv.datasets.ImageFolder(data_path, transform=transform)
    dataloader = thud.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(f"{len(dataset)} images found in {data_path} and loaded into {len(dataloader)} batches of size {batch_size}.")

    device = "cuda" if th.cuda.is_available() else "cpu"
    with open(config_path, "r") as f:
        config = json.load(f)
    model = modules.IRN(num_channels=config["num_channels"],
                        transform_cfgs=config["transforms"]).to(device)
    print(f"Loaded {config_path} model ({device}) with {utils.count_parameters(model)} parameters.")

    optim = th.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    sched = th.optim.lr_scheduler.StepLR(optim, round(1e5 / len(dataloader)), gamma=0.5)

    if checkpoint_path is not None:
        utils.load_state(checkpoint_path, model, optim, sched)
        print(f"Resuming from {checkpoint_path}, epoch {sched.last_epoch}.")

    s = 2 ** len(config["transforms"])
    grad_clip = config["gradient_clip"]
    max_epoch = epochs + sched.last_epoch

    print(f"Starting training for {s}x.")
    for e in range(sched.last_epoch + 1, max_epoch + 1):
        avg_loss, avg_hr_loss, avg_lr_loss, avg_pdm_loss = 0, 0, 0, 0
        start = time.perf_counter()
        if th.cuda.is_available():
            th.cuda.reset_peak_memory_stats()
        for x, _ in dataloader:
            optim.zero_grad()
            x = x.to(device)
            xlr = tvf.resize(x, [x.shape[2] // s, x.shape[3] // s], interpolation=tvf.InterpolationMode.BICUBIC)

            c, d = model(x)
            lr_loss = th.nn.functional.mse_loss(c, xlr, reduction="sum") / c.shape[0]
            pdm_loss = (d ** 2).sum() / d.shape[0]
            z = th.zeros_like(d)
            c = model.inverse(utils.quantize(c), z)
            hr_loss = utils.charbonnier_loss(c, x)
            loss = hr_loss + pdm_loss + lr_loss * s ** 2

            avg_hr_loss += hr_loss.item() / len(dataloader)
            avg_lr_loss += lr_loss.item() / len(dataloader)
            avg_pdm_loss += pdm_loss.item() / len(dataloader)
            avg_loss += loss.item() / len(dataloader)
            if not th.isfinite(loss):
                raise RuntimeError("Loss is not finite.")
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
        sched.step()
        utils.save_state(save_path, model, optim, sched)
        print(f"Epoch {str(e).zfill(len(str(max_epoch)))}/{max_epoch}, \
              Avg Loss: {avg_loss:.6e}, Avg HR Loss: {avg_hr_loss:.6e}, Avg LR Loss: {avg_lr_loss:.6e}, \
              Avg PDM Loss: {avg_pdm_loss:.6e}, Time: {time.perf_counter() - start:.2f} s, \
              Max Mem: {th.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=144)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    main(batch_size=args.batch_size,
         patch_size=args.patch_size,
         epochs=args.epochs,
         checkpoint_path=args.checkpoint_path,
         config_path=args.config_path,
         save_path=args.save_path,
         data_path=args.data_path,
         num_workers=args.num_workers,
         seed=args.seed)
