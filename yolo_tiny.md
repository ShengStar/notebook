# yolo tiny

## 指令

> conda activate pointpillars_python36
>
> export PYTHONPATH=/home/lixusheng/yolov4_pointpillar/yolov4/:$PYTHONPATH
>
> python create_data.py create_kitti_info_file --data_path=/mnt/KITTI_DATASET_ROOT/
>
> python create_data.py create_reduced_point_cloud --data_path=/mnt/KITTI_DATASET_ROOT/
>
> python create_data.py create_groundtruth_database --data_path=/mnt/KITTI_DATASET_ROOT/
>
> python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=data/model1
>
> 

