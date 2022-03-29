# PointPillars的量化

## 指令

* conda activate pointpillars_python36

* 文件夹：/home/lixusheng/yolov4_pointpillar/pointpillar

* 环境变量：export PYTHONPATH=/home/lixusheng/yolov4_pointpillar/pointpillar/:$PYTHONPATH

* 创建数据集：python create_data.py create_kitti_info_file --data_path=/mnt/KITTI_DATASET_ROOT/

* 创建点云数据：python create_data.py create_reduced_point_cloud --data_path=/mnt/KITTI_DATASET_ROOT/

* 创建点云 python create_data.py create_groundtruth_database --data_path=/mnt/KITTI_DATASET_ROOT/

* 训练：python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=data/model20211012

  > 量化
  >
  > coors_range：[  0.   -39.68  -3.    69.12  39.68   1.  ]
  >
  > grid_size = [432.00003 496.        1.     ]
  >
  > grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
  >
  > =[432 496   1]
  >
  > grid_size = np.round(grid_size, 0, grid_size).astype(np.int8)
  >
  > [-80 -16   1]
  >
  > 