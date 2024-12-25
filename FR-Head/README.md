# FR-Head-HDGCN复现

## 环境配置
- 建议Linux系统
- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX
- 在`cd FE-Head` Run `conda env create -f mix_GCN.yml` 创建环境
- Run `pip install -e torchlight` 
- Run `pip install torch_topological` 
## 数据预处理
data文件夹有data.zip压缩包，压缩后得到5个文件：tarin_joint.npy,train_label.npy,val_joint.npy,val_label.npy,test_joint.npy。都是关节模态的数据，在Process中生成骨骼(bone)和他们对应的运动（motion）数据。简称如下：Joint，Bone，Joint_motion，Bone_motion。
- 如何处理得到4种数据在Process文件夹的readme文件里
## 运行
### Train:
#### Joint
`python main.py --config config/uav/jhd.yaml --work-dir results/uav/jhd --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0`
#### Bone
`python main.py --config config/uav/bhd.yaml --work-dir results/uav/bhd --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0`
#### Joint_motion
`python main.py --config config/uav/jmhd.yaml --work-dir results/uav/jmhd --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0`
#### Bone_motion
`python main.py --config config/uav/bmhd.yaml --work-dir results/uav/bmhd --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0`

### Test:
#### Joint
`python main.py --config /home/featurize/work/block/FR-Head/config/uav/jhd.yaml --work-dir results/uav/jhd --phase test --save-score True --weights /home/featurize/work/block/FR-Head/results/uav/jhd/runs-54-56430.pt --device 0`
#### Bone
`python main.py --config /home/featurize/work/block/FR-Head/config/uav/bhd.yaml --work-dir results/uav/bhd --phase test --save-score True --weights /home/featurize/work/block/FR-Head/results/uav/bhd/runs-54-56430.pt --device 0`
#### Joint_motion
`python main.py --config /home/featurize/work/block/FR-Head/config/uav/jmhd.yaml --work-dir results/uav/jmhd --phase test --save-score True --weights /home/featurize/work/block/FR-Head/results/uav/jmhd/runs-55-57475.pt --device 0`
#### Bone_motion
`python main.py --config /home/featurize/work/block/FR-Head/config/uav/bmhd.yaml --work-dir results/uav/bmhd --phase test --save-score True --weights /home/featurize/work/block/FR-Head/results/uav/bmhd/runs-55-57475.pt --device 0`

### 注意：权重地址需要改变，文件地址也是,config文件中需要更改的地方：test_feeder_args-----data_path、label_path，如果是训练这里改成val的数据，看训练效果，测试集这里改成test的数据，但是测试集没有标签文件，这是比赛的题目，我们需要得到测试集的预测文件提交比赛。

## 贝叶斯优化
### 验证集得到的得分文件进行贝叶斯优化得到集成比例。4种数据对应4种得分文件，需要知道集成比例，因此使用贝叶斯优化，看每种数据要多少的比例。
需要集成什么文件在weight.py里面设置，可以调节集成范围，[0.2,1.2]。设置好文件地址就可以运行了（需要设置四种得分文件的地址和val_label.npy的地址）。最后run`python weight.py `