#!/usr/bin/env bash
set -euo pipefail

release_tag="${PARTICLEGS_RELEASE_TAG:-v1.0.0}"
release_base="https://github.com/QuanJinSheng/ParticleGS/releases/download/${release_tag}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 {bat|fan|shark|darkroom|chessboard|all}" >&2
    exit 2
fi

if [ "$1" = "all" ]; then
    scenes=(bat fan shark darkroom chessboard)
else
    scenes=("$1")
fi

case "${scenes[0]}" in
    bat|fan|shark|darkroom|chessboard) ;;
    *)
        echo "Unknown scene: ${scenes[0]}" >&2
        exit 2
        ;;
esac

tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT

curl -fL "${release_base}/SHA256SUMS" -o "${tmp_dir}/SHA256SUMS"

for scene in "${scenes[@]}"; do
    asset="particlegs-${scene}-best-${release_tag}.tar.gz"
    curl -fL "${release_base}/${asset}" -o "${tmp_dir}/${asset}"
    (
        cd "${tmp_dir}"
        grep "  ${asset}$" SHA256SUMS | sha256sum --check -
    )
    tar -xzf "${tmp_dir}/${asset}"
done

echo "Checkpoints extracted under checkpoints/."
