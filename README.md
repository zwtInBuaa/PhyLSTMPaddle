## PhyLSTM: 物理信息引导的长短期记忆网络

#### 1\. 项目简介

我们引入了一种创新的物理知识LSTM框架，用于对缺乏数据的非线性结构系统进行元建模。基本概念是将可用但尚不完整的物理知识（如物理定律、科学原理）整合到深度长短时记忆（LSTM）网络中，该网络在可行的解决方案空间内限制和促进学习。物理约束嵌入在损失函数中，以强制执行模型训练，即使在可用训练数据集非常有限的情况下，也能准确地捕捉潜在的系统非线性。特别是对于动态结构，考虑运动方程的物理定律、状态依赖性和滞后本构关系来构建物理损失。嵌入式物理可以缓解过拟合问题，减少对大型训练数据集的需求，并提高训练模型的鲁棒性，使其具有外推能力，从而进行更可靠的预测。因此，物理知识指导的深度学习范式优于传统的非物理指导的数据驱动神经网络。

我们基于原始的模型实现，使用PaddleScience框架复现，实现了PhyLSTM2、PhyLSTM3模型的迁移和复现。

论文：https://arxiv.org/pdf/2002.10253

代码：https://github.com/zhry10/PhyLSTM

算法链接：https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/phylstm/

paddle配置：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/quick_start_cn.html

#### 2.问题定义

结构系统的元建模旨在开发低保真度（或低阶）模型，以有效地捕捉潜在的非线性输入－输出行为。元模型可以在从高保真度模拟或实际系统感知获得的数据集上进行训练。为了更好地说明，我们考虑一个建筑类型结构并假设地震动力学由低保真度非线性运动方程（EOM）支配：

$$
\mathbf{M} \ddot{\mathbf{u}}+\underbrace{\mathbf{C} \dot{\mathbf{u}}+\lambda \mathbf{K} \mathbf{u}+(1-\lambda) \mathbf{K r}}_{\mathbf{h}}=-\mathbf{M} \Gamma a_g
$$


其中 $M$ 是质量矩阵；$C$ 为阻尼矩阵；$K$ 为刚度矩阵。
控制方程可以改写成一个更一般的形式：

$$
\ddot{\mathbf{u}}+\mathrm{g}=-\Gamma a_g
$$

#### 3\. 快速开始

##### 3.1 环境依赖

注：目前飞桨支持 Python 3.6 ~ 3.9 版本，pip3 要求 20.2.2 或更高版本，请提前安装对应版本的 Python 和 pip 工具。

```shell
pip3 install -r requirements.txt
```

```txt
paddlepaddle==3.0
numpy==1.23.5
scipy==1.10.1
hydra-core==1.3.2
omegaconf==2.3.0
matplotlib==3.7.1
```

##### 3.2 数据准备

```shell
# Linux系统下载数据
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
# Windows系统下载数据
curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat -o data_boucwen.mat
```

##### 3.3 代码结构

```
.
├── PhyLSTM2.py
├── PhyLSTM2_w_Graph.py
├── PhyLSTM3.py
├── PhyLSTM3_w_Graph.py
├── README.md
├──output
│   ├──训练输出、checkpoint保存、可视化结果 
├── conf
│   ├── phylstm2.yaml
│   ├── phylstm3.yaml
│   └── phylstm3_new.yaml
├── data
│   └── data_boucwen.mat
├── draw_loss.py
├── functions.py
└── requirements.txt
```

##### 3.4 模型训练

```shell
# 简单复现，注意在配置文件中设置模式为train，其余参数可自己设置
python PhyLSTM3.py
# 带模型保存，train log日志，train/val/pred结果可视化
python PhyLSTM3_w_Graph.py
```

