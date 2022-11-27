import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from functools import partial
from pprint import pprint

import albumentations as A
import hydra
import timm
import torch
import torchvision
import torchvision.transforms as T

from albumentations.pytorch import ToTensorV2
from alibi_detect.cd import MMDDrift
from alibi_detect.cd.pytorch import preprocess_drift
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# set random seed and device
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def calc_drift(cfg):
    test_transforms = T.Compose([T.ToTensor(), T.Normalize(mean=cfg.MEAN, std=cfg.STD)])

    test_ds = torchvision.datasets.CIFAR10(
        root="data/", train=False, download=True, transform=test_transforms
    )
    test_ref = next(iter(DataLoader(test_ds, batch_size=500, shuffle=False)))
    print(test_ref[0].shape, test_ref[1].shape)

    # model = timm.create_model(model_name="resnet18", pretrained=True, num_classes=10).to(
    #     device
    # )
    model = hydra.utils.instantiate(cfg.model).to(device)
    if cfg.chpt_path:
        checkpoint = torch.load(cfg.chpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

    preprocess_fn = partial(
        preprocess_drift, model=model, device=device, batch_size=512
    )
    cd = MMDDrift(
        test_ref[0],
        backend="pytorch",
        p_val=0.05,
        preprocess_fn=preprocess_fn,
        n_permutations=100,
    )

    perturbed_images = []

    perturb = A.Compose(
        [
            A.Rotate(limit=5, interpolation=1, border_mode=4),
            A.HorizontalFlip(),
            A.CoarseDropout(2, 8, 8, 1, 8, 8),
            A.RandomBrightnessContrast(brightness_limit=1.5, contrast_limit=0.9),
            ToTensorV2(),
        ]
    )

    for idx in range(500):
        perturbed_image = torch.tensor(
            perturb(
                image=test_ref[0][idx].numpy().transpose(1, 2, 0),
            )["image"]
        )

        perturbed_images.append(perturbed_image)

    perturbed_images = torch.stack(perturbed_images)

    print("Calculating Drift")
    pprint(cd.predict(perturbed_images), sort_dicts=False)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="drift.yaml")
def main(cfg: DictConfig) -> None:
    calc_drift(cfg)


if __name__ == "__main__":
    main()
