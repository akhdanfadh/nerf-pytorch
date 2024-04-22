"""
A set of utility functions commonly used in training/evaluating model.
"""

import os
import random
from pathlib import Path
from typing import Any, Optional, Tuple, Type

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from radiance_fields import data_loader as module_dataset
from radiance_fields import network as module_network
from radiance_fields import scene as module_scene
from radiance_fields import signal_encoder as module_encoder
from radiance_fields.renderer import integrator as module_integrator
from radiance_fields.renderer import ray_sampler as module_sampler
from radiance_fields.renderer.volume_renderer import VolumeRenderer


def _init_object(
    cfg: DictConfig, cfg_name: str, module: Type[Any], *args, **kwargs
) -> Any:
    """Initialize an object from a module using the configuration.

    This method finds a function handle with the name given as 'type' in the
    configuration file, and returns the instance initialized with corresponding
    arguments given.

    `function = _init_object(cfg, 'name', module, a, b=1)`
    is equivalent to
    `function = module."cfg['name']['type']"(a, b=1)`

    Args:

        cfg_name: The name of the configuration to use.
        module: The module to initialize the object from.

    Returns:
        The initialized object.

    Raises:
        AssertionError: Keyword arguments should not changed the specified
            configuration file.
    """
    try:
        cfg = getattr(cfg, cfg_name)
    except AttributeError as exc:
        raise ValueError(f"Configuration file does not have '{cfg_name}' key.") from exc
    try:
        module_type = getattr(cfg, "type")
    except AttributeError as exc:
        raise ValueError(f"Unsupported '{cfg_name}' type.") from exc
    try:
        module_args = getattr(cfg, "args")
    except AttributeError:
        module_args = {}

    # update module_args with kwargs if not in config file
    # assert all([k not in module_args for k in kwargs]), (
    #     "Overwriting object arguments in config file is not allowed. "
    #     "Passed them in config file instead."
    # )
    module_args.update(kwargs)

    return getattr(module, module_type)(*args, **module_args)


def init_tensorboard(tb_log_dir: Path) -> SummaryWriter:
    """Initialize Tensorboard writer."""
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    return writer


def init_device(cfg: DictConfig) -> None:
    """Initialize RNG's seed, CUDA device, and PyTorch CPU threads.

    Note that it does not guarantee full reproducibility and it trades off speed.
    See more here: https://pytorch.org/docs/stable/notes/randomness.html.
    """
    # set seed for reproducibility
    seed = cfg.device.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set CUDA device
    if torch.cuda.is_available():
        device_id = cfg.device.cuda.device_id

        if device_id > torch.cuda.device_count() - 1:
            print(
                "Invalid device ID. "
                f"There are {torch.cuda.device_count()} devices but got index {device_id}."
            )
            device_id = 0
            cfg.device.cuda.device_id = device_id  # overwrite config
            print("Set device ID to 0 by default.")
        torch.cuda.set_device(cfg.device.cuda.device_id)
        print(f"CUDA device detected. Using device {torch.cuda.current_device()}.")
    else:
        print("CUDA is not supported on this system. Using CPU by default.")

    # set PyTorch CPU threads
    torch.set_num_threads(cfg.device.torch.num_threads)


def init_renderer(cfg: DictConfig) -> VolumeRenderer:
    """Initializes the renderer for rendering scene representations."""
    renderer_cfg = cfg.renderer
    integrator = _init_object(renderer_cfg, "integrator", module_integrator)
    sampler = _init_object(renderer_cfg, "sampler", module_sampler)
    renderer = VolumeRenderer(sampler=sampler, integrator=integrator)
    return renderer


def init_dataset_and_loader(cfg: DictConfig, mode: str) -> Tuple[Dataset, DataLoader]:
    """Initializes dataset and data loader."""
    dataset = _init_object(cfg, "data", module_dataset, data_type=mode)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
    )

    return dataset, loader


def init_scene(
    cfg: DictConfig,
) -> Tuple[module_scene.BasePrimitive, module_scene.BasePrimitive]:
    """Initializes the scene object."""
    encoder_cfg = cfg.signal_encoder
    coord_enc = _init_object(encoder_cfg, "coordinate", module_encoder)
    dir_enc = _init_object(encoder_cfg, "direction", module_encoder)
    encoders = {
        "coord_enc": coord_enc,
        "dir_enc": dir_enc,
    }

    device_id = cfg.device.cuda.device_id
    if cfg.scene.type == "PrimitiveCube":
        default_network = _init_object(
            cfg,
            "network",
            module_network,
            pos_dim=coord_enc.out_dim,
            view_dir_dim=dir_enc.out_dim,
        ).to(device_id)
        default_scene = module_scene.CubePrimitive(default_network, encoders)

        fine_scene = None
        if cfg.renderer.num_samples_fine > 0:
            fine_network = _init_object(
                cfg,
                "network",
                module_network,
                pos_dim=coord_enc.out_dim,
                view_dir_dim=dir_enc.out_dim,
            ).to(device_id)
            fine_scene = module_scene.CubePrimitive(fine_network, encoders)

    return default_scene, fine_scene


def init_optimizer_and_scheduler(
    cfg: DictConfig,
    default_scene: module_scene.BasePrimitive,
    fine_scene: Optional[module_scene.BasePrimitive] = None,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Initializes optimizer and learning rate scheduler for training."""
    # identify parameters to optimize
    params = list(default_scene.radiance_field.parameters())
    if fine_scene:
        params += list(fine_scene.radiance_field.parameters())

    # initialize optimizer
    optimizer = None
    optimizer = _init_object(cfg.trainer, "optimizer", torch.optim, params)

    # initialize scheduler
    scheduler = None
    if cfg.trainer.scheduler.type == "ExponentialLR":
        # compute decay rate
        init_lr = cfg.trainer.scheduler.init_lr
        end_lr = cfg.trainer.scheduler.end_lr
        num_iter = cfg.trainer.num_iter
        gamma = pow(end_lr / init_lr, 1 / num_iter)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    else:
        raise NotImplementedError(
            f"Unsupported scheduler type: {cfg.trainer.scheduler.type}"
        )

    return optimizer, scheduler


def init_loss_function(cfg: DictConfig) -> torch.nn.Module:
    """Initializes the loss function."""
    loss_fn = _init_object(cfg.trainer, "loss_fn", torch.nn)
    return loss_fn


def save_checkpoint(
    ckpt_dir: str,
    epoch: int,
    default_scene: module_scene.BasePrimitive,
    optimizer: torch.optim.Optimizer,
    fine_scene: module_scene.BasePrimitive = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
) -> None:
    """Save the model checkpoint."""
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(ckpt_dir) / f"epoch-{epoch}.pth"

    state = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "default_scene": default_scene.radiance_field.state_dict(),
    }
    if scheduler:
        state["scheduler"] = scheduler.state_dict()
    if fine_scene:
        state["fine_scene"] = fine_scene.radiance_field.state_dict()

    torch.save(state, fname)


def load_checkpoint(
    ckpt_dir: str,
    default_scene: module_scene.BasePrimitive,
    optimizer: torch.optim.Optimizer,
    fine_scene: module_scene.BasePrimitive = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
) -> Tuple[int, str]:
    """Load the model checkpoint."""
    epoch = 1

    if not Path(ckpt_dir).exists():
        return (
            epoch,
            f"Checkpoint directory not found in {ckpt_dir}. Starting from scratch.",
        )

    ckpt_files = sorted(list(Path(ckpt_dir).glob("*.pth")))
    if len(ckpt_files) == 0:
        return epoch, f"No checkpoint found in {ckpt_dir}. Starting from scratch."

    ckpt_file = ckpt_files[-1]
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load epoch
    epoch = ckpt["epoch"]

    # load scene(s) states
    default_scene.radiance_field.load_state_dict(ckpt["default_scene"])
    default_scene.radiance_field.to(torch.cuda.current_device())
    if fine_scene:
        fine_scene.radiance_field.load_state_dict(ckpt["fine_scene"])
        fine_scene.radiance_field.to(torch.cuda.current_device())

    # load optimizer and scheduler states
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler"])

    return epoch, f"Checkpoint loaded successfully from {ckpt_file}"
