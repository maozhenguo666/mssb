# 2024-全球校园人工智能算法精英大赛-算法挑战赛-基于无人机的人体行为识别

### 依赖库

python ：`numpy tqdm`



## 数据准备（UAV-human骨架数据预处理）

### 注意
1. 在验证中，我们发现Windows多线程处理存在一些问题（包括内存占用异常增加），或者系统盘与工作区在一个存储介质上，也会存在系统IO耗尽导致死机卡顿。因此默认启用单线程
2. 为加快处理速度，也可以启用多线程，通常可带来2-4倍的性能提升。

**多线程处理**：在以下提到的命令中，添加`--use_mp True`即可。

### 流程

1. 解压数据集：将`data.zip`在`data`目录下解压
2. 数据集处理出bone模态数据（可选）：运行`python gen_modal.py --modal bone`得到bone模态数据
3. 数据集处理出motion模态数据（可选）：运行`python gen_modal.py --modal motion`得到motion模态的数据
4. bone模态与joint模态合并（可选）：运行`python gen_modal.py --modal jmb`得到合并模态的数据
5. 最终你会得到如下所展示的目录结构与文件
```
└─data
    ├── train_label.pkl
    ├── train_bone_motion.npy
    ├── train_bone.npy
    ├── train_joint_motion.npy
    ├── train_joint.npy
    ├── val_label.pkl
    ├── val_bone_motion.npy
    ├── val_bone.npy
    ├── val_joint_motion.npy
    ├── val_joint.npy
    ├── test_bone_motion.npy
    ├── test_bone.npy
    ├── test_joint_bone.npy
    ├── test_joint.npy
```
