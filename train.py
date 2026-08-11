import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state
import uuid
import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, merge_config
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    deform,
    load2gpu_on_the_fly,
    t_max,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    test_psnr = 0.0

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = []
                psnr_test = []
                psnr_fid_reconstruction = []
                psnr_fid_extrapolation = []
                for idx, viewpoint in enumerate(config["cameras"]):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz.detach()
                    sh = scene.gaussians.get_features[:, 0, :].detach()
                    r = scene.gaussians.get_rotation.detach()
                    s = scene.gaussians.get_scaling.detach()
                    a = scene.gaussians.get_opacity.detach()
                    feature = torch.cat([xyz, sh.view(sh.size(0), -1), r, s, a], dim=1)
                    deform_pkgs = deform.step(feature, fid)
                    d_xyz, d_rotation, d_scaling = (
                        deform_pkgs["d_xyz"],
                        deform_pkgs["d_rotation"],
                        deform_pkgs["d_scaling"],
                    )
                    image = torch.clamp(
                        renderFunc(
                            viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling
                        )["render"],
                        0.0,
                        1.0,
                    ).cpu()
                    gt_image = torch.clamp(viewpoint.original_image.cpu(), 0.0, 1.0)
                    if load2gpu_on_the_fly:
                        viewpoint.load2device("cpu")
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test.append(l1_loss(image, gt_image).mean().item())
                    psnr_value = psnr(image, gt_image).mean().item()
                    psnr_test.append(psnr_value)

                    if fid < t_max:
                        psnr_fid_reconstruction.append(psnr_value)
                    else:
                        psnr_fid_extrapolation.append(psnr_value)

                l1_test = np.mean(l1_test)
                psnr_test = np.mean(psnr_test)

                psnr_fid_reconstruction_mean = (
                    np.mean(psnr_fid_reconstruction) if psnr_fid_reconstruction else None
                )
                psnr_fid_extrapolation_mean = (
                    np.mean(psnr_fid_extrapolation) if psnr_fid_extrapolation else None
                )

                if config["name"] == "test" or len(validation_configs[0]["cameras"]) == 0:
                    test_psnr = psnr_test

                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                print(f"[ITER {iteration}] PSNR for fid <= {t_max}: {psnr_fid_reconstruction_mean}")
                print(f"[ITER {iteration}] PSNR for fid > {t_max}: {psnr_fid_extrapolation_mean}")

                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


class Trainer:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        self.tb_writer = prepare_output_and_logger(dataset)
        self.gaussians = GaussianModel(dataset.sh_degree)
        self.deform = DeformModel(encoder_config=dataset.encoder_config)
        self.deform.train_setting(opt)

        self.scene = Scene(dataset, self.gaussians, skip_val=True)
        self.gaussians.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")

    def train(self, iters=5000):
        for _ in range(iters):
            self.train_step()

    def train_step(self):
        self.iter_start.record()

        if self.iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        if not self.viewpoint_stack or self.iteration == self.opt.warm_up:
            if self.iteration < self.opt.warm_up:
                self.viewpoint_stack = self.scene.getInitCameras().copy()
            else:
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
                dataset_max_time = max([viewpoint.fid for viewpoint in self.viewpoint_stack])
                if self.opt.gradual:
                    cur_max_fid = (
                        ((self.iteration - self.opt.warm_up) // 1000 + 1.0)
                        / ((10000 - self.opt.warm_up) // 1000 + 1.0)
                        * dataset_max_time
                    )
                    cur_max_fid = min(cur_max_fid, dataset_max_time)
                    self.viewpoint_stack = [
                        viewpoint
                        for viewpoint in self.viewpoint_stack
                        if viewpoint.fid <= cur_max_fid
                    ]

        if self.iteration == self.opt.warm_up:
            self.scene.init_cameras = None

        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        reg = 0.0
        if self.iteration < self.opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            xyz = self.gaussians.get_xyz.detach()
            sh = self.gaussians.get_features[:, 0, :].detach()
            r = self.gaussians.get_rotation.detach()
            s = self.gaussians.get_scaling.detach()
            a = self.gaussians.get_opacity.detach()
            feature = torch.cat([xyz, sh.view(sh.size(0), -1), r, s, a], dim=1)

            use_div_loss = self.opt.lambda_div > 0.0
            deform_pkgs = self.deform.step(
                feature, fid, compute_div_loss=use_div_loss, div_mode=self.opt.div_mode
            )
            d_xyz = deform_pkgs["d_xyz"]
            d_rotation = deform_pkgs["d_rotation"]
            d_scaling = deform_pkgs["d_scaling"]
            if self.iteration < self.opt.warm_up + 2000:
                d_rotation = 0.0
                d_scaling = 0.0

        render_pkg_re = render(
            viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg_re["render"],
            render_pkg_re["viewspace_points"],
            render_pkg_re["visibility_filter"],
            render_pkg_re["radii"],
        )

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        if (
            self.iteration >= self.opt.warm_up
            and self.opt.lambda_div > 0.0
            and "div_loss" in deform_pkgs
        ):
            reg = deform_pkgs["div_loss"]
            loss = loss + self.opt.lambda_div * deform_pkgs["div_loss"]
        loss.backward()

        self.iter_end.record()

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device("cpu")

        with torch.no_grad():

            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix(
                    {
                        "Loss": f"{self.ema_loss_for_log:.{7}f}",
                        "pts": len(self.gaussians.get_xyz),
                        "reg": f"{reg:.{4}f}",
                    }
                )

                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            self.gaussians.max_radii2D[visibility_filter] = torch.max(
                self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )

            cur_psnr = training_report(
                self.tb_writer,
                self.iteration,
                Ll1,
                loss,
                self.iter_start.elapsed_time(self.iter_end),
                self.testing_iterations,
                self.scene,
                render,
                (self.pipe, self.background),
                self.deform,
                self.dataset.load2gpu_on_the_fly,
                self.dataset.max_time,
            )
            if self.iteration in self.testing_iterations:
                cur_psnr = float(cur_psnr)
                if cur_psnr >= self.best_psnr:
                    self.best_psnr = cur_psnr
                    self.scene.save(self.iteration, True)
                    self.deform.save_weights(self.args.model_path, self.iteration, True)
                    print("Best: {} PSNR: {}".format(self.iteration, self.best_psnr))

            if self.iteration in self.saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)

            if self.iteration < self.opt.densify_until_iter and not (
                self.opt.warm_up <= self.iteration < self.opt.warm_up + self.opt.netwarm
            ):
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (
                    self.iteration > self.opt.densify_from_iter
                    and self.iteration % self.opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if self.iteration > self.opt.opacity_reset_interval else None
                    )
                    self.gaussians.densify_and_prune(
                        self.opt.densify_grad_threshold,
                        self.opt.opacity,
                        self.scene.cameras_extent,
                        size_threshold,
                    )
                    self.deform.OdeTransformer.encoder.cache = None

                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                    self.dataset.white_background and self.iteration == self.opt.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

            if self.iteration % 500 == 0:
                xyz = self.gaussians.get_xyz.detach()
                sh = self.gaussians.get_features[:, 0, :].detach()
                r = self.gaussians.get_rotation.detach()
                s = self.gaussians.get_scaling.detach()
                a = self.gaussians.get_opacity.detach()
                feature = torch.cat([xyz, sh.view(sh.size(0), -1), r, s, a], dim=1)

                _disp = d_xyz.detach() if isinstance(d_xyz, torch.Tensor) else None
                if _disp is not None and _disp.shape[0] != feature.shape[0]:
                    _disp = None

                _do_vis = self.args.vis_fps and (self.iteration % self.args.vis_fps_interval == 0)
                if _do_vis:
                    self.deform.OdeTransformer.encoder.vis_fps = True

                self.deform.OdeTransformer.encoder.refresh(feature, displacements=_disp)

                if _do_vis:
                    self.deform.OdeTransformer.encoder.vis_fps = False
                    fps_data = self.deform.OdeTransformer.encoder.get_last_fps_data()
                    fps_dir = os.path.join(self.args.model_path, "fps_vis")
                    os.makedirs(fps_dir, exist_ok=True)
                    snap = {"all_xyz": feature[:, :3].cpu().numpy()}
                    snap["fps_centers"] = fps_data["centers"]
                    if fps_data["high_idx"] is not None:
                        snap["high_idx"] = fps_data["high_idx"]
                    if _disp is not None:
                        snap["displacements"] = _disp.cpu().numpy()
                    np.savez_compressed(
                        os.path.join(fps_dir, f"iter_{self.iteration:06d}.npz"), **snap
                    )
                    print(f"[FPS vis] saved snapshot iter={self.iteration} → {fps_dir}")

            if self.iteration < self.opt.iterations:
                if (
                    self.iteration > self.opt.warm_up + self.opt.netwarm - 200
                    or self.iteration < self.opt.warm_up
                ):
                    self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
                self.deform.update_learning_rate(self.iteration)

        self.iteration += 1


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[3000, 5000, 6000, 7000] + list(range(8000, 80001, 1000)),
    )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--vis_fps",
        action="store_true",
        default=False,
        help="save FPS centers + all Gaussians for visualization",
    )
    parser.add_argument(
        "--vis_fps_interval",
        type=int,
        default=1000,
        help="save FPS snapshot every N iterations (default 1000)",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.conf is not None and os.path.exists(args.conf):
        print("Find Config:", args.conf)
        args = merge_config(args, args.conf)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer = Trainer(
        args=args,
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
    )
    trainer.train(op.extract(args).iterations)

    print("\nTraining complete.")
