import numpy as np
import paddle
import ppsci
from ppsci.utils import logger
import os
from os import path as osp
import matplotlib.pyplot as plt

debug = False


# 绘图功能实现
def plot_loss_curves(train_loss, val_loss, best_loss, save_path):
    """绘制训练和验证损失曲线（保持不变）"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='train', color='blue', linewidth=2)
    plt.plot(val_loss, label='valid', color='red', linewidth=2)

    # 标记最佳损失点
    min_idx = np.argmin(val_loss)
    plt.scatter(min_idx, val_loss[min_idx],
                color='green', marker='*', s=200, label=f'最佳损失: {best_loss:.6f}')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss Curves', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(osp.join(save_path, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_series(true_data, pred_data, title, time_axis, dof=0, save_path=None):
    """绘制时间序列对比图（修改为每个样本单独绘图）"""
    # 确保 save_path 存在
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # 处理单样本和多样本情况
    if len(true_data.shape) == 3:
        # 3D数据: [样本数, 时间步, 自由度]
        num_samples = true_data.shape[0]

        for sample_idx in range(num_samples):
            plt.figure(figsize=(12, 7))
            plt.plot(time_axis, true_data[sample_idx, :, dof],
                     label='Reference', color='blue', linestyle='-', linewidth=2)
            plt.plot(time_axis, pred_data[sample_idx, :, dof],
                     label='PhyLSTM3', color='red', linestyle='--', linewidth=2)

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('time [s]', fontsize=14)
            plt.ylabel(title.split('_')[-1], fontsize=14)
            plt.title(f'{title} (sample {sample_idx + 1})', fontsize=16)
            plt.legend(fontsize=12)
            plt.tight_layout()

            if save_path:
                title_safe = title.replace(':', '').replace(' ', '_')
                plt.savefig(osp.join(save_path, f'{title_safe}_sample{sample_idx + 1}.png'),
                            dpi=300, bbox_inches='tight')
            plt.close()
            if debug:
                break
    else:
        # 2D数据: [时间步, 自由度]
        plt.figure(figsize=(12, 7))
        plt.plot(time_axis, true_data[:, dof], label='Reference', color='blue', linestyle='-', linewidth=2)
        plt.plot(time_axis, pred_data[:, dof], label='PhyLSTM3', color='red', linestyle='--', linewidth=2)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('time [s]', fontsize=14)
        plt.ylabel(title.split('_')[-1], fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()

        if save_path:
            title_safe = title.replace(':', '').replace(' ', '_')
            plt.savefig(osp.join(save_path, f'{title_safe}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_hysteresis(true_y, true_g, pred_y, pred_g, title, save_path=None):
    """绘制磁滞回线图（修改为每个样本单独绘图）"""
    # 确保 save_path 存在
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # 处理单样本和多样本情况
    if len(true_y.shape) == 3:
        # 3D数据: [样本数, 时间步, 自由度]
        num_samples = true_y.shape[0]

        for sample_idx in range(num_samples):
            plt.figure(figsize=(10, 8))
            plt.plot(true_y[sample_idx, :, 0], true_g[sample_idx, :, 0],
                     label='Reference', color='blue', linewidth=2)
            plt.plot(pred_y[sample_idx, :, 0], pred_g[sample_idx, :, 0], label='PhyLSTM3', color='red', linestyle='--',
                     linewidth=2)

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Displacement [m]', fontsize=14)
            plt.ylabel('Normalized \n Restoring Force [N]', fontsize=14)
            plt.title(f'{title} (sample{sample_idx + 1})', fontsize=16)
            plt.legend(fontsize=12)
            plt.tight_layout()

            if save_path:
                title_safe = title.replace(':', '').replace(' ', '_')
                plt.savefig(osp.join(save_path, f'{title_safe}_sample{sample_idx + 1}.png'),
                            dpi=300, bbox_inches='tight')
            plt.close()
            if debug:
                break
    else:
        # 2D数据: [时间步, 自由度]
        plt.figure(figsize=(10, 8))
        plt.plot(true_y[:, 0], true_g[:, 0], label='Reference', color='blue', linewidth=2)
        plt.plot(pred_y[:, 0], pred_g[:, 0], label='PhyLSTM3', color='red', linestyle='--', linewidth=2)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Displacement [m]', fontsize=14)
        plt.ylabel('Normalized \n Restoring Force [N]', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()

        if save_path:
            title_safe = title.replace(':', '').replace(' ', '_')
            plt.savefig(osp.join(save_path, f'{title_safe}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_all_results(cfg, mode, true_y, pred_y, true_yt, pred_yt, true_g, pred_g, time):
    """绘制所有结果图表（保持逻辑不变，但每个样本单独绘图）"""
    save_path = osp.join(cfg.output_dir, "plots")
    os.makedirs(save_path, exist_ok=True)

    # 1. 绘制损失曲线（保持不变）
    # plot_loss_curves(train_loss, val_loss, best_loss, save_path)

    # 2. 绘制训练集位移对比（每个样本单独绘图）
    train_displacement_path = osp.join(save_path, f"{mode}/displacement")
    plot_time_series(true_y, pred_y, '[' + mode + '] Displacement(u)', time, dof=0,
                     save_path=train_displacement_path)

    # 3. 绘制训练集速度对比
    train_velocity_path = osp.join(save_path, f"{mode}/velocity")
    plot_time_series(true_yt, pred_yt, '[' + mode + '] Velocity (u_t)', time, dof=0,
                     save_path=train_velocity_path)

    # 4. 绘制训练集加速度对比
    # train_acceleration_path = osp.join(save_path, "train_acceleration")
    # plot_time_series(train_true_ytt, train_pred_ytt, '训练集加速度 (u_tt) 对比', time_train, dof=0,
    #                  save_path=train_acceleration_path)

    # 5. 绘制训练集力对比
    train_force_path = osp.join(save_path, f"{mode}/restoring_force")
    plot_time_series(true_g, pred_g, '[' + mode + '] Restoring Force (g)', time, dof=0,
                     save_path=train_force_path)

    # 6. 绘制训练集磁滞回线
    train_hysteresis_path = osp.join(save_path, f"{mode}/hysteresis_curves")
    plot_hysteresis(true_y, true_g, pred_y, pred_g, '[' + mode + '] Hysteresis Curves',
                    save_path=train_hysteresis_path)

    logger.info(f"{[mode]}模式的所有图表已保存至: {save_path}")


def transform_in(_in):
    """输入数据转换函数，重命名输入键名"""
    transformed = {k: v for k, v in _in.items()}
    if "phi_t" in transformed:
        transformed["phi"] = transformed.pop("phi_t")  # 重命名phi_t为phi
    return transformed


def transform_out(_in, _out):
    """输出数据转换函数，此处无需转换"""
    return _out


class Dataset:
    """PhyLSTM模型的数据集类，处理训练和验证数据"""

    def __init__(self, eta, eta_t, g, ag, ag_c, lift, phi_t):
        self.eta = eta
        self.eta_t = eta_t
        self.g = g
        self.ag = ag
        self.ag_c = ag_c
        self.lift = lift
        self.phi_t = phi_t
        # self.eta = paddle.to_tensor(eta)
        # self.eta_t = paddle.to_tensor(eta_t)
        # self.g = paddle.to_tensor(g)
        # self.ag = paddle.to_tensor(ag)
        # self.ag_c = paddle.to_tensor(ag_c)
        # self.lift = paddle.to_tensor(lift)
        # self.phi_t = paddle.to_tensor(phi_t)

    def get(self, n_train):
        """获取训练和验证数据，按比例分割数据集"""
        n_total = self.eta.shape[0]
        indices = np.random.permutation(n_total)
        split_idx = min(int(n_total * 0.8), n_train)  # 80%用于训练，不超过指定训练样本数

        # 训练数据
        input_dict_train = {
            "eta": self.eta[indices[:split_idx]],
            "eta_t": self.eta_t[indices[:split_idx]],
            "g": self.g[indices[:split_idx]],
            "ag": self.ag[indices[:split_idx]],
            "ag_c": self.ag_c,
            # "lift": self.lift,
            "phi_t": self.phi_t[indices[:split_idx]],
        }

        label_dict_train = {
            "eta": self.eta[indices[:split_idx]],
            "eta_t": self.eta_t[indices[:split_idx]],
            "g": self.g[indices[:split_idx]],
            "lift": self.lift,
        }

        # 验证数据
        input_dict_val = {
            "eta": self.eta[indices[split_idx:]],
            "eta_t": self.eta_t[indices[split_idx:]],
            "g": self.g[indices[split_idx:]],
            "ag": self.ag[indices[split_idx:]],
            "ag_c": self.ag_c,
            # "lift": self.lift,
            "phi_t": self.phi_t[indices[split_idx:]],
        }

        label_dict_val = {
            "eta": self.eta[indices[split_idx:]],
            "eta_t": self.eta_t[indices[split_idx:]],
            "g": self.g[indices[split_idx:]],
            "lift": self.lift,
        }

        return input_dict_train, label_dict_train, input_dict_val, label_dict_val


def train_loss_func2(output_dict, label_dict, weight_dict=None):
    """训练损失函数，包含物理约束和预测误差"""
    # 提取预测值
    eta_pred = output_dict["eta_pred"]
    eta_dot_pred = output_dict["eta_dot_pred"]
    g_pred = output_dict["g_pred"]
    eta_t_pred_c = output_dict["eta_t_pred_c"]
    eta_dot_pred_c = output_dict["eta_dot_pred_c"]
    lift_pred_c = output_dict["lift_pred_c"]

    # 提取标签
    eta_true = label_dict["eta"]
    eta_t_true = label_dict["eta_t"]
    g_true = label_dict["g"]
    lift_true = label_dict["lift"]  # 确保label_dict中有"lift"键

    # 预测值与真实值的损失
    loss_u = paddle.mean((eta_true - eta_pred) ** 2)
    loss_udot = paddle.mean((eta_t_true - eta_dot_pred) ** 2)
    loss_g = paddle.mean((g_true - g_pred) ** 2)

    # 配置点处的物理约束损失
    loss_ut_c = paddle.mean((eta_t_pred_c - eta_dot_pred_c) ** 2)
    loss_e = paddle.mean((lift_true - lift_pred_c) ** 2)

    # 总损失（包含预测误差和物理约束）
    # 由于位移损失量级太小，因此需要乘以一个较大的因子保证可以有较好的拟合效果
    total_loss = 200 *loss_u + loss_udot + loss_g + loss_ut_c + loss_e

    return {"total_loss": total_loss}


def train_loss_func3(output_dict, label_dict, weight_dict=None):
    """扩展的训练损失函数，包含更多约束项"""
    # 提取预测值
    eta_pred = output_dict["eta_pred"]
    eta_dot_pred = output_dict["eta_dot_pred"]
    g_pred = output_dict["g_pred"]
    eta_t_pred_c = output_dict["eta_t_pred_c"]
    eta_dot_pred_c = output_dict["eta_dot_pred_c"]
    lift_pred_c = output_dict["lift_pred_c"]
    g_t_pred_c = output_dict["g_t_pred_c"]
    g_dot_pred_c = output_dict["g_dot_pred_c"]

    # 提取标签
    eta_true = label_dict["eta"]
    eta_t_true = label_dict["eta_t"]
    g_true = label_dict["g"]
    lift_true = label_dict["lift"]

    # 预测值与真实值的损失
    loss_u = paddle.mean((eta_true - eta_pred) ** 2)
    loss_udot = paddle.mean((eta_t_true - eta_dot_pred) ** 2)
    loss_g = paddle.mean((g_true - g_pred) ** 2)

    # 配置点处的物理约束损失
    loss_ut_c = paddle.mean((eta_t_pred_c - eta_dot_pred_c) ** 2)
    loss_gt_c = paddle.mean((g_t_pred_c - g_dot_pred_c) ** 2)
    loss_e = paddle.mean((lift_true - lift_pred_c) ** 2)

    # 总损失（包含更多约束项）
    total_loss = 200 * loss_u + loss_udot + loss_g + loss_ut_c + loss_gt_c + loss_e
    # print(f"loss_u: {loss_u.item():.6f} | loss_udot: {loss_udot.item():.6f} | loss_g: {loss_g.item():.6f} | loss_ut_c: {loss_ut_c.item():.6f} | loss_gt_c: {loss_gt_c.item():.6f} | loss_e: {loss_e.item():.6f} | total_loss: {total_loss.item():.6f}")

    return {"total_loss": total_loss}


def metric_expr(output_dict, label_dict):
    """评估指标计算函数，计算相对L2误差"""
    # 提取预测值
    eta_pred = output_dict["eta_pred"]
    eta_dot_pred = output_dict["eta_dot_pred"]
    g_pred = output_dict["g_pred"]

    # 提取标签
    eta_true = label_dict["eta"]
    eta_t_true = label_dict["eta_t"]
    g_true = label_dict["g"]

    # 计算相对L2误差
    error_eta = paddle.norm(eta_true - eta_pred) / paddle.norm(eta_true)
    error_eta_t = paddle.norm(eta_t_true - eta_dot_pred) / paddle.norm(eta_t_true)
    error_g = paddle.norm(g_true - g_pred) / paddle.norm(g_true)

    return {
        "eta_error": error_eta,
        "eta_t_error": error_eta_t,
        "g_error": error_g,
    }
