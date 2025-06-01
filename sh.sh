#!/bin/bash

# 基础数据路径
base_data_dir="/hdd2/PhysData/NVFi_datasets/InDoorObj/data"
conf_dir="./arguments/nvfiobj"
datasets=( "bat" )

base_data_dir="/hdd2/PhysData/NVFi_datasets/InDoorSeg/data"
conf_dir="./arguments/nvfiseg"
datasets=( "chessboard" )

base_data_dir="/hdd2/PhysData/data/"
conf_dir="./arguments/panopticSports"
datasets=( "boxes" )

for dataset in "${datasets[@]}"; do
    # 原始数据路径
    source_dir="${base_data_dir}/${dataset}"
    # 输出目录路径
    model_dir="./output/${dataset}"
    # 配置文件路径
    config_file="${conf_dir}/${dataset}.py"

#     检查数据目录是否存在
    if [ ! -d "${source_dir}" ]; then
        echo "错误：数据目录 ${source_dir} 不存在"
        continue
    fi

    # 检查配置文件是否存在
    if [ ! -f "${config_file}" ]; then
        echo "错误：配置文件 ${config_file} 不存在"
        continue
    fi

    # 创建输出目录（如果不存在）
    mkdir -p "${model_dir}"

    # 运行训练命令
    echo "正在训练 ${dataset}..."
#    python train.py -s "${source_dir}" -m "${model_dir}" --conf "${config_file}"
    python render.py --conf "${config_file}" -m "${model_dir}"  --iteration best

    echo "${dataset} 训练完成！"
    echo "-------------------------"
done
