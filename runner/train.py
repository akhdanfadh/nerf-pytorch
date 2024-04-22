"""
Training script.
"""

import shutil
from pathlib import Path
from typing import Dict, Tuple

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from tqdm import tqdm

import wandb
from radiance_fields import renderer as module_renderer
from radiance_fields import scene as module_scene
from runner.utils import (
    init_dataset_and_loader,
    init_device,
    init_loss_function,
    init_optimizer_and_scheduler,
    init_renderer,
    init_scene,
    init_tensorboard,
    load_checkpoint,
    save_checkpoint,
)


def train_one_epoch(
    epoch: int,
    cfg: DictConfig,
    default_scene: module_scene.BasePrimitive,
    renderer: module_renderer.VolumeRenderer,
    dataset: torch.utils.data.Dataset,
    loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    fine_scene: module_scene.BasePrimitive = None,
    pbar: tqdm = None,
) -> Dict:
    """
    Training routine for one epoch of the model.

    Returns:
        loss_dict (Dict): A dictionary containing several loss values.
    """
    # get device
    device_idx = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_idx}")

    # store several loss values (e.g., coarse loss, total loss, etc.)
    loss_dict = {}

    for batch in loader:
        total_loss = 0.0

        # parse batch
        img_data, camera_pose = batch  # still loads in cpu
        img_data = img_data.squeeze().reshape(-1, 3)  # (H, W, 3) -> (H*W, 3)
        camera_pose = camera_pose.squeeze()

        # initialize gradients
        optimizer.zero_grad()

        # create camera for current viewpoint
        img_height, img_width = dataset.img_height, dataset.img_width
        camera = module_renderer.Camera(
            camera_to_world=camera_pose,
            f_x=dataset.focal_length,
            f_y=dataset.focal_length,
            c_x=img_width / 2,
            c_y=img_height / 2,
            near=cfg.renderer.t_near,
            far=cfg.renderer.t_far,
            image_width=img_width,
            image_height=img_height,
            device=device,
        )

        # sample pixels to render
        num_pixels = cfg.renderer.num_pixels
        pixel_indices = None

        # for early stage of training, sample pixels around the center of the image
        if epoch < cfg.trainer.warmup_epochs:
            center_i = (img_height - 1) // 2
            center_is = torch.arange(
                start=center_i - center_i // 2, end=center_i + center_i // 2 + 1
            )

            center_j = (img_width - 1) // 2
            center_js = torch.arange(
                start=center_j - center_j // 2, end=center_j + center_j // 2 + 1
            )

            center_indices = torch.cartesian_prod(center_is, center_js)
            center_indices = center_indices[:, 0] * img_width + center_indices[:, 1]

            # random sample pixels around the center
            pixel_indices = center_indices[
                torch.randperm(len(center_indices))[:num_pixels]
            ]

        # forward prop (coarse scene)
        coarse_rgb_pred, coarse_weights, coarse_t_samples = renderer.render_scene(
            target_scene=default_scene,
            camera=camera,
            num_pixels=num_pixels,
            num_samples=cfg.renderer.num_samples_coarse,
            pixel_indices=pixel_indices,
        )

        # compute loss
        rgb_target = img_data[renderer.pixel_indices, ...]
        coarse_loss = loss_fn(coarse_rgb_pred, rgb_target.to(coarse_rgb_pred))
        total_loss += coarse_loss

        if "coarse_loss" not in loss_dict:
            loss_dict["coarse_loss"] = coarse_loss.item()
        else:
            loss_dict["coarse_loss"] += coarse_loss.item()

        # perform hierarchical sampling
        if fine_scene is not None:

            # forward prop (fine scene)
            fine_rgb_pred, _, _ = renderer.render_scene(
                target_scene=fine_scene,
                camera=camera,
                num_pixels=num_pixels,
                num_samples=cfg.renderer.num_samples_fine,
                pixel_indices=pixel_indices,
                prev_weights=coarse_weights,
                prev_t_samples=coarse_t_samples,
            )

            # compute loss
            fine_loss = loss_fn(fine_rgb_pred, rgb_target.to(fine_rgb_pred))
            total_loss += fine_loss

            if "fine_loss" not in loss_dict:
                loss_dict["fine_loss"] = fine_loss.item()
            else:
                loss_dict["fine_loss"] += fine_loss.item()

        if "loss" not in loss_dict:
            loss_dict["loss"] = total_loss.item()
        else:
            loss_dict["loss"] += total_loss.item()

        # backward prop
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # update iteration bar
        pbar.update(cfg.data.batch_size)

    # compute average losses over all batches
    for loss in loss_dict:
        loss_dict[loss] /= len(loader)

    return loss_dict


@torch.no_grad()
def validate_one_epoch(
    epoch: int,
    cfg: DictConfig,
    default_scene: module_scene.BasePrimitive,
    renderer: module_renderer.VolumeRenderer,
    dataset: torch.utils.data.Dataset,
    loader: torch.utils.data.DataLoader,
    fine_scene: module_scene.BasePrimitive = None,
) -> Tuple[Dict, torch.Tensor]:
    """
    Validation routine for one epoch of the model.

    Returns:
        metric_dict (Dict): A dictionary containing several metrics.
        val_images (torch.Tensor): A tensor containing validation images.
    """
    # get device
    device_idx = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_idx}")

    # initialize metric dictionary
    lpips = LPIPS(net_type="vgg", normalize=True).to(device)
    psnr = PSNR().to(device)
    ssim = SSIM().to(device)

    val_images = []
    metric_dict = {}

    num_sample = 0
    eval_pbar = tqdm(
        range(cfg.trainer.validation.num_batch),
        desc=f"Validating [Epoch {epoch}]",
        leave=False,
    )

    for batch_index, batch in enumerate(loader):
        if batch_index >= cfg.trainer.validation.num_batch:
            break

        # parse batch
        img_data, camera_pose = batch  # still loads in cpu
        img_data = img_data.squeeze()
        camera_pose = camera_pose.squeeze()
        batch_size = len(camera_pose)

        # count the number of samples used for validation
        num_sample += batch_size

        # create camera for current viewpoint
        img_height, img_width = dataset.img_height, dataset.img_width
        camera = module_renderer.Camera(
            camera_to_world=camera_pose,
            f_x=dataset.focal_length,
            f_y=dataset.focal_length,
            c_x=img_width / 2,
            c_y=img_height / 2,
            near=cfg.renderer.t_near,
            far=cfg.renderer.t_far,
            image_width=img_width,
            image_height=img_height,
            device=device,
        )

        # render all pixels in coarse scene
        img_pred, coarse_weights, coarse_t_samples = renderer.render_scene(
            target_scene=default_scene,
            camera=camera,
            num_samples=cfg.renderer.num_samples_coarse,
            num_ray_batch=cfg.trainer.num_ray_batch,
        )

        # render all pixesl in fine scene
        if fine_scene is not None:
            img_pred, _, _ = renderer.render_scene(
                target_scene=fine_scene,
                camera=camera,
                num_samples=cfg.renderer.num_samples_fine,
                num_ray_batch=cfg.trainer.num_ray_batch,
                prev_weights=coarse_weights,
                prev_t_samples=coarse_t_samples,
            )

        # transform images: (H * W, C) -> (1, C, H, W)
        img_pred = img_pred.reshape(img_height, img_width, -1)  # (H, W, C)
        img_pred = img_pred.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        img_data = img_data.reshape(img_height, img_width, -1)  # (H, W, C)
        img_data = img_data.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        # clamp values to [0, 1]
        img_pred = torch.clamp(img_pred, 0.0, 1.0)
        img_data = torch.clamp(img_data, 0.0, 1.0)

        # collect images
        img_all = torch.cat([img_pred.cpu(), img_data], dim=3)
        val_images.append(img_all)

        # compute metrics
        lpips_val = lpips(img_pred, img_data.to(img_pred)).item()
        if "lpips" not in metric_dict:
            metric_dict["lpips"] = lpips_val
        else:
            metric_dict["lpips"] += lpips_val

        psnr_val = psnr(img_pred, img_data.to(img_pred)).item()
        if "psnr" not in metric_dict:
            metric_dict["psnr"] = psnr_val
        else:
            metric_dict["psnr"] += psnr_val

        ssim_val = ssim(img_pred, img_data.to(img_pred)).item()
        if "ssim" not in metric_dict:
            metric_dict["ssim"] = ssim_val
        else:
            metric_dict["ssim"] += ssim_val

        # update iteration bar by 1
        eval_pbar.update(cfg.data.batch_size)

    # compute average metrics over all batches
    for metric in metric_dict:
        metric_dict[metric] /= num_sample

    return metric_dict, torch.cat(val_images, dim=0)


@hydra.main(
    version_base=None,
    config_path="../configs",  # config file search path is relative to this script
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    """The entry point of training code."""

    # resume training setup
    if cfg.resume is not None:
        log_dir = Path(cfg.resume.log_dir)
        assert log_dir.exists(), f"Provided log directory {log_dir} does not exist."

        cfg_dir = log_dir / ".hydra"
        assert (
            log_dir / ".hydra"
        ).exists(), f"Provided log directory {log_dir} does not contain .hydra directory for config."

        # update loaded cfg and use that for training consistency
        old_cfg = OmegaConf.load(cfg_dir / "config.yaml")
        for key, value in cfg.resume["update"].items():
            OmegaConf.update(old_cfg, key, value)
        cfg = old_cfg

        # copy runtime config to log dir
        src_dir = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
        tgt_dir = "_".join(
            [src_dir.name, src_dir.parents[1].name, src_dir.parents[0].name]
        )
        shutil.copytree(src_dir, log_dir / tgt_dir)

    else:
        # hydra's default output directory
        log_dir = Path(HydraConfig.get().runtime.output_dir)

    # initialize logger: Tensorboard writer
    tb_log_dir = log_dir / "tensorboard"
    writer = init_tensorboard(tb_log_dir)

    # initialize wandb logger
    if cfg.use_wandb:
        wandb.init(project="radiance-fields", config=OmegaConf.to_container(cfg))
    else:
        wandb.init(mode="disabled")

    # initialize device: RNG seed, CUDA device, and PyTorch CPU threads
    init_device(cfg)

    # initialize renderer
    renderer = init_renderer(cfg)

    # initialize dataset and data loader
    train_dataset, train_loader = init_dataset_and_loader(cfg, "train")
    val_dataset, val_loader = init_dataset_and_loader(cfg, "val")

    # initialize scene and network parameters
    default_scene, fine_scene = init_scene(cfg)

    # initialize optimizer and learning rate scheduler
    optimizer, scheduler = init_optimizer_and_scheduler(cfg, default_scene, fine_scene)

    # initialize objective function
    loss_fn = init_loss_function(cfg)

    # load if checkpoint exists
    start_epoch, load_msg = load_checkpoint(
        ckpt_dir=log_dir / "ckpt",
        default_scene=default_scene,
        fine_scene=fine_scene,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    total_epoch = cfg.trainer.num_iter // len(train_dataset)
    if start_epoch >= total_epoch:
        print(
            "Training has already completed.",
            "Try increasing num_iter in default config for further training.",
        )
        return

    print("===========================================")
    print(f"Dataset type / Scene name: {cfg.data.type} / {cfg.data.args.scene_name}")
    print(f"Number of training data: {len(train_dataset)}")
    print(f"Image resolution: ({train_dataset.img_height}, {train_dataset.img_width})")
    print(f"Loading checkpoint: {load_msg}")
    print("===========================================")

    # initialize progress bar
    start_iter = (start_epoch - 1) * len(train_dataset)  # epoch counter starts from 1
    max_iter = total_epoch * len(train_dataset)
    with tqdm(total=max_iter, initial=start_iter) as train_pbar:
        for epoch in range(start_epoch, total_epoch + 1):
            train_pbar.set_description(f"Training [Epoch {epoch}/{total_epoch}]")

            # train one epoch
            train_loss_dict = train_one_epoch(
                epoch,
                cfg,
                default_scene,
                renderer,
                train_dataset,
                train_loader,
                loss_fn,
                optimizer,
                scheduler,
                fine_scene,
                train_pbar,
            )
            for loss_name, value in train_loss_dict.items():
                writer.add_scalar(f"train/{loss_name}", value, epoch)
                wandb.log({f"train/{loss_name}": value}, step=epoch)

            # validate every configured number of epochs or at the end of training
            if (
                epoch % cfg.trainer.validation.validate_every == 0
                or epoch == total_epoch
            ):
                val_metric_dict, val_images = validate_one_epoch(
                    epoch,
                    cfg,
                    default_scene,
                    renderer,
                    val_dataset,
                    val_loader,
                    fine_scene,
                )
                train_pbar.set_postfix(val_metric_dict)

                # log metrics
                for metric_name, value in val_metric_dict.items():
                    writer.add_scalar(f"val/{metric_name}", value, epoch)
                    wandb.log({f"val/{metric_name}": value}, step=epoch)

                # log images
                for index in range(val_images.shape[0]):
                    writer.add_image(f"val/image_{index}", val_images[index], epoch)
                    wandb.log(
                        {f"val/image_{index}": wandb.Image(val_images[index].cpu())},
                        step=epoch,
                    )

            # save checkpoint every configured number of epochs or at the end of training
            if epoch % cfg.trainer.save_epoch_ckpt == 0 or epoch == total_epoch:
                save_checkpoint(
                    log_dir / "ckpt",
                    epoch,
                    default_scene,
                    optimizer,
                    fine_scene,
                    scheduler,
                )


if __name__ == "__main__":
    main()
