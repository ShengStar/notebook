# CenterPoint

## 1、安装

> mkdir CenterPoint
>
> conda create --name centerpoint python=3.6
> conda activate centerpoint
>
> pip install torch==1.4.0 torchvision==0.5.0
>
> pip install -r requirements.txt
>
> /home/lixusheng/CenterPoint

### 1.1 apex

> git clone https://github.com/NVIDIA/apex
> cd apex
> git checkout 5633f6  # recent commit doesn't build in our system 
> pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

### 1.2 spconv(pytorch 1.4.0 cmake 3.13.12)

> git clone https://github.com/traveller59/spconv.git --recursive
>
> cd spconv && git checkout 7342772
> python setup.py bdist_wheel
>
> cd ./dist && pip install *

### 安装cmake3.13.2

查看cmake版本：cmake --version

卸载旧版本： sudo apt-get autoremove cmake

下载：

```text
wget https://cmake.org/files/v3.13/cmake-3.13.2.tar.gz
tar xvf cmake-3.13.2.tar.gz
cd cmake-3.13.2
```

安装：

./bootstrap --prefix=/usr

make

sudo make install

检查版本：

```text
cmake --version
```

## 训练

### 数据准备

