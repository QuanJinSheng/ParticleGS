#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path


SCENES = {
    "bat": ("dynobjects", "dataset/DynObjects/data/bat"),
    "fan": ("dynobjects", "dataset/DynObjects/data/fan"),
    "shark": ("dynobjects", "dataset/DynObjects/data/shark"),
    "darkroom": ("dynamic_indoor", "dataset/DynIndoorScene/data/darkroom"),
    "chessboard": ("dynamic_indoor", "dataset/DynIndoorScene/data/chessboard"),
}
REQUIRED_FILES = (
    "cfg_args",
    "deform/iteration_best/deform.pth",
    "deform/iteration_best/iter.txt",
    "point_cloud/iteration_best/point_cloud.ply",
    "point_cloud/iteration_best/iter.txt",
)


def parse_checkpoint(value):
    scene, separator, path = value.partition("=")
    if not separator or scene not in SCENES:
        raise argparse.ArgumentTypeError("expected SCENE=/path/to/checkpoint")
    return scene, Path(path).expanduser().resolve()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_config(source, destination, model_path, dataset_path):
    config = source.read_text(encoding="utf-8")
    config, source_count = re.subn(
        r"source_path='[^']*'", f"source_path='{dataset_path}'", config, count=1
    )
    config, model_count = re.subn(
        r"model_path='[^']*'", f"model_path='{model_path}'", config, count=1
    )
    if source_count != 1 or model_count != 1:
        raise ValueError(f"could not sanitize paths in {source}")
    destination.write_text(config, encoding="utf-8")


def read_best_iteration(scene_root):
    values = []
    for relative_path in (
        "deform/iteration_best/iter.txt",
        "point_cloud/iteration_best/iter.txt",
    ):
        match = re.fullmatch(
            r"Best iter:\s*(\d+)\s*", (scene_root / relative_path).read_text(encoding="utf-8")
        )
        if not match:
            raise ValueError(f"invalid iteration marker: {scene_root / relative_path}")
        values.append(int(match.group(1)))
    if len(set(values)) != 1:
        raise ValueError(f"Gaussian and deformation iterations differ for {scene_root.name}")
    return values[0]


def main():
    parser = argparse.ArgumentParser(description="Package ParticleGS release checkpoints")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-tag", default="v1.0.0")
    parser.add_argument("--license", type=Path, default=Path("LICENSE"))
    args = parser.parse_args()

    checkpoints = dict(args.checkpoint)
    missing_scenes = set(SCENES) - set(checkpoints)
    if missing_scenes:
        parser.error(f"missing checkpoints: {', '.join(sorted(missing_scenes))}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_root = output_dir / "staging"
    if staging_root.exists():
        raise FileExistsError(f"staging directory already exists: {staging_root}")
    staging_root.mkdir()

    manifest = {"release": args.release_tag, "license": "CC-BY-NC-SA-4.0", "scenes": {}}
    archives = []

    try:
        for scene, (family, dataset_path) in SCENES.items():
            source_root = checkpoints[scene]
            for relative_path in REQUIRED_FILES:
                if not (source_root / relative_path).is_file():
                    raise FileNotFoundError(source_root / relative_path)

            best_iteration = read_best_iteration(source_root)
            relative_model_path = Path("checkpoints") / family / scene
            staged_scene = staging_root / relative_model_path
            staged_scene.mkdir(parents=True)

            sanitize_config(
                source_root / "cfg_args",
                staged_scene / "cfg_args",
                relative_model_path.as_posix(),
                dataset_path,
            )
            for relative_path in REQUIRED_FILES[1:]:
                destination = staged_scene / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / relative_path, destination)
            shutil.copy2(args.license, staged_scene / "LICENSE")

            asset_name = f"particlegs-{scene}-best-{args.release_tag}.tar.gz"
            archive_path = output_dir / asset_name
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(staged_scene, arcname=relative_model_path)
            archives.append(archive_path)

            files = {}
            for path in sorted(staged_scene.rglob("*")):
                if path.is_file():
                    files[path.relative_to(staged_scene).as_posix()] = {
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
            manifest["scenes"][scene] = {
                "family": family,
                "best_iteration": best_iteration,
                "dataset_path": dataset_path,
                "archive": asset_name,
                "files": files,
            }

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        checksum_paths = archives + [manifest_path]
        checksum_text = "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths)
        (output_dir / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")
    finally:
        shutil.rmtree(staging_root)


if __name__ == "__main__":
    main()
