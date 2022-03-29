# nuScenes数据集处理

## 设备

![img](https://img-blog.csdnimg.cn/20181103225533690.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDk5NDkxMw==,size_16,color_FFFFFF,t_70)

特点：包含camera、radar和ladar数据，有1000个场景组成，每个scenes长度为20秒，每个scenes包含40个关键帧（key frames），也就是每秒钟有2个关键帧，其他的帧为sweeps。关键帧经过手工的标注，每一帧中都有了若干个annotation，标注的形式为bounding box。不仅标注了大小、范围、还有类别、可见程度等。

## 1.1 Setup

metadata and annotation 是必须要下载的，其他数据可以根据需要下载。下载后的数据是 tbz2 格式的压缩文件，需要经过两次解压缩。在 [nuScenes-devkit github page](https://github.com/nutonomy/nuscenes-devkit) 上面有一个可以用来visualization的开发工具。

### 1.1.1 nuScenes devkit

> conda create --name nuScenes_dvkit python=3.6
>
> conda activate nuScenes_dvkit
>
> pip install nuscenes-devkit



