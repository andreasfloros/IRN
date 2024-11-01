import json
import modules
import time
import torch as th
import torch.utils.data as thud
import torchvision as tv
import utils


@th.inference_mode()
def main(checkpoint_path: str, config_path: str, data_path: str) -> None:

    dataset = tv.datasets.ImageFolder(data_path, transform=tv.transforms.ToTensor())
    dataloader = thud.DataLoader(dataset)
    print(f"{len(dataset)} images found in {data_path} and loaded into {len(dataloader)} batches of size 1.")

    device = "cuda" if th.cuda.is_available() else "cpu"
    with open(config_path, "r") as f:
        config = json.load(f)
    model = modules.IRN(num_channels=config["num_channels"],
                        transform_cfgs=config["transforms"]).to(device)
    utils.load_state(checkpoint_path, model)
    print(f"Loaded {config_path} model ({device}) with {utils.count_parameters(model)} parameters.")

    s = 2 ** len(config['transforms'])
    print(f"Starting evaluation for {s}x.")
    avg_loss = 0
    start = time.perf_counter()
    if th.cuda.is_available():
        th.cuda.reset_peak_memory_stats()
    for x, _ in dataloader:
        x = x.to(device)
        x = utils.modcrop(x, s)
        c, d = model(x)
        c = model.inverse(utils.quantize(c), th.zeros_like(d))

        c = utils.rgb2y(c)
        x = utils.rgb2y(x)
        loss = -10 * th.nn.functional.mse_loss(c, x).log10()
        avg_loss += loss.detach() / len(dataloader)
    print(f"PSNR: {avg_loss.item():.6e} dB, Time: {time.perf_counter() - start:.2f} s, \
          Max Mem: {th.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    main(checkpoint_path=args.checkpoint_path, config_path=args.config_path, data_path=args.data_path)
