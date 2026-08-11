#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import json
import time

import torch

from scene import Scene, DeformModel, GaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.image_utils import psnr
from utils.metrics import SSIM, LPIPS
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import imageio
import numpy as np


def parse_iteration(value):
    if value == "best":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("iteration must be an integer, -1, or 'best'") from exc


def build_gs_feature(gaussians):
    xyz = gaussians.get_xyz.detach()
    sh = gaussians.get_features[:, 0, :].detach()
    r = gaussians.get_rotation.detach()
    s = gaussians.get_scaling.detach()
    a = gaussians.get_opacity.detach()
    return torch.cat([xyz, sh.view(sh.size(0), -1), r, s, a], dim=1)


def scalarize(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.mean())
    return float(value)


def append_metrics_log(log_path, record):
    if not log_path:
        return

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_set(
    model_path,
    load2gpu_on_the_fly,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    deform,
    static_threshold,
    metrics_only=False,
):
    render_root = os.path.join(model_path, name, f"ours_{iteration}")
    render_path = os.path.join(render_root, "renders")
    gts_path = os.path.join(render_root, "gt")
    depth_path = os.path.join(render_root, "depth")
    speed_path = os.path.join(render_root, "speed.txt")

    if not metrics_only:
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)

    total_time = 0.0
    inter_psnr = outer_psnr = 0.0
    inter_ssim = outer_ssim = 0.0
    inter_lpips = outer_lpips = 0.0
    inter_cet = outer_cet = 0
    metric_ssim, metric_lpips = SSIM(), LPIPS()

    with torch.no_grad():
        feature = build_gs_feature(gaussians)
        dxyz0 = deform.step(feature, 0.0)["d_xyz"]
        static_mask = torch.ones(feature.shape[0], dtype=torch.bool, device=feature.device)

        sampled_time = torch.linspace(0, 0.75, steps=75, device=feature.device)
        for t in sampled_time:
            dxyz_t = deform.step(feature, t)["d_xyz"]
            displacement = (dxyz_t - dxyz0).norm(dim=-1)
            static_mask &= displacement < static_threshold

        motion_mask = ~static_mask

    d_xyz = torch.zeros_like(gaussians.get_xyz)
    d_rotation = torch.zeros_like(gaussians.get_rotation)
    d_scaling = torch.zeros_like(gaussians.get_scaling)
    deform_pkg = deform.step(feature, 0.0)
    d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = (
        deform_pkg["d_xyz"][static_mask],
        deform_pkg["d_rotation"][static_mask],
        deform_pkg["d_scaling"][static_mask],
    )

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    renderings = [] if not metrics_only else None

    def update_metrics(rendering, gt, inter_flag):
        nonlocal inter_psnr, inter_ssim, inter_lpips, inter_cet
        nonlocal outer_psnr, outer_ssim, outer_lpips, outer_cet
        psnr_val = psnr(rendering, gt)
        ssim_val = metric_ssim(rendering.unsqueeze(0), gt.unsqueeze(0))
        lpips_val = metric_lpips(rendering.unsqueeze(0) * 2.0 - 1.0, gt.unsqueeze(0) * 2.0 - 1.0)
        if inter_flag:
            inter_psnr += psnr_val
            inter_ssim += ssim_val
            inter_lpips += lpips_val
            inter_cet += 1
        else:
            outer_psnr += psnr_val
            outer_ssim += ssim_val
            outer_lpips += lpips_val
            outer_cet += 1

    feature = build_gs_feature(gaussians)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid

        t_start = time.time()
        deform_pkg = deform.step(feature, fid, True)
        d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = (
            deform_pkg["d_xyz"][motion_mask],
            deform_pkg["d_rotation"][motion_mask],
            deform_pkg["d_scaling"][motion_mask],
        )
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)
        total_time += time.time() - t_start

        rendering, depth = results["render"], results["depth"]
        depth = depth / (depth.max() + 1e-5)
        gt = view.original_image[0:3, :, :]

        update_metrics(rendering, gt, inter_flag=(fid <= 0.75))

        if not metrics_only:
            torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))
            torchvision.utils.save_image(depth, os.path.join(depth_path, f"{idx:05d}.png"))

            render_np = rendering.detach().cpu().numpy().transpose(1, 2, 0)
            renderings.append(to8b(render_np))

    if not metrics_only:
        video_path = os.path.join(render_path, "video.mp4")
        imageio.mimwrite(video_path, renderings, fps=30, quality=8)

    def safe_divide(num, den):
        return num / den if den > 0 else np.array([0.0, 0.0])

    fps_val = safe_divide(len(views), total_time)
    all_psnr = safe_divide(inter_psnr + outer_psnr, inter_cet + outer_cet)
    all_ssim = safe_divide(inter_ssim + outer_ssim, inter_cet + outer_cet)
    all_lpips = safe_divide(inter_lpips + outer_lpips, inter_cet + outer_cet)

    summary = {
        "all_psnr": scalarize(all_psnr.mean()),
        "all_ssim": scalarize(all_ssim.mean()),
        "all_lpips": scalarize(all_lpips.mean()),
        "interp_psnr": scalarize(safe_divide(inter_psnr, inter_cet).mean()),
        "extrap_psnr": scalarize(safe_divide(outer_psnr, outer_cet).mean()),
        "interp_ssim": scalarize(safe_divide(inter_ssim, inter_cet).mean()),
        "extrap_ssim": scalarize(safe_divide(outer_ssim, outer_cet).mean()),
        "interp_lpips": scalarize(safe_divide(inter_lpips, inter_cet).mean()),
        "extrap_lpips": scalarize(safe_divide(outer_lpips, outer_cet).mean()),
        "fps": scalarize(fps_val),
        "gaussian_num": int(len(gaussians.get_xyz.detach())),
        "num_views": int(len(views)),
        "interp_count": int(inter_cet),
        "extrap_count": int(outer_cet),
        "metrics_only": bool(metrics_only),
    }

    print("all", summary["all_psnr"], summary["all_ssim"], summary["all_lpips"])
    print("PSNR:", summary["interp_psnr"], summary["extrap_psnr"])
    print("SSIM:", summary["interp_ssim"], summary["extrap_ssim"])
    print("LPIPS:", summary["interp_lpips"], summary["extrap_lpips"])
    print("FPS:", summary["fps"])
    print("Gaussian num:", summary["gaussian_num"])

    if not metrics_only:
        with open(speed_path, "w") as f:
            f.write("FPS: " + str(summary["fps"]))

    return summary


def render_sets(
    dataset: ModelParams,
    iteration,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_val: bool,
    skip_test: bool,
    mode: str,
    static_threshold=0.00,
    fps=60,
    eval_name=None,
    metrics_log=None,
    metrics_only=False,
):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=iteration,
        shuffle=False,
        skip_train=skip_train,
        skip_val=skip_val,
        skip_test=skip_test,
    )
    deform = DeformModel(dataset.encoder_config)
    deform.load_weights(dataset.model_path, iteration=iteration)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    metrics = None

    if not skip_test:
        eval_views = scene.getTestCameras()
        if not skip_val:
            eval_views = eval_views + scene.getValCameras()
        run_name = eval_name or "test"

        with torch.no_grad():
            metrics = render_set(
                dataset.model_path,
                dataset.load2gpu_on_the_fly,
                run_name,
                scene.loaded_iter,
                eval_views,
                gaussians,
                pipeline,
                background,
                deform,
                static_threshold,
                metrics_only=metrics_only,
            )

    if metrics is not None:
        record = {
            "model_path": dataset.model_path,
            "source_path": dataset.source_path,
            "iteration": scene.loaded_iter,
            "mode": mode,
            "eval_name": eval_name or "test",
            "skip_train": bool(skip_train),
            "skip_val": bool(skip_val),
            "skip_test": bool(skip_test),
        }
        record.update(metrics)
        append_metrics_log(metrics_log, record)
        if metrics_log:
            print(f"Metrics log saved to {metrics_log}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=parse_iteration)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default="render", choices=["render"])
    parser.add_argument("--static_threshold", default=0.00, type=float)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--eval_name", type=str, default=None)
    parser.add_argument("--metrics_log", type=str, default=None)
    parser.add_argument("--metrics_only", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_val,
        args.skip_test,
        args.mode,
        args.static_threshold,
        args.fps,
        eval_name=getattr(args, "eval_name", None),
        metrics_log=getattr(args, "metrics_log", None),
        metrics_only=args.metrics_only,
    )
