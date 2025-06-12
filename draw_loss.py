import os
import re
import hydra
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


@hydra.main(version_base=None, config_path="./conf", config_name="phylstm2")
# @hydra.main(version_base=None, config_path="./conf", config_name="phylstm2")
def main(cfg):
    """主函数：训练完成后绘制损失曲线"""
    # 训练过程...（原有代码省略）

    # 绘制训练损失曲线
    plot_train_loss(cfg)


def plot_train_loss(cfg):
    """从日志文件提取损失数据并绘制曲线"""
    # 日志文件路径
    log_file = os.path.join(cfg.output_dir, "train.log")

    # 检查文件存在性
    if not os.path.exists(log_file):
        print(f"错误：未找到日志文件 {log_file}")
        return

    # 提取Epoch和对应损失值
    epochs, losses = extract_loss_data(log_file, cfg.TRAIN.epochs, cfg.TRAIN.iters_per_epoch)

    # 绘制曲线
    draw_loss_curve(epochs, losses, cfg.output_dir)


def extract_loss_data(log_file, total_epochs, iters_per_epoch):
    """从日志中提取每个Epoch的最终损失"""
    epochs = []
    losses = []

    with open(log_file, 'r') as f:
        for line in f:
            # 正则匹配训练日志格式：[Train][Epoch X/X][Iter Y/Y] lr:..., loss:..., sup_train:...
            match = re.search(r'\[Train\]\[Epoch\s+(\d+)/(\d+)\]\[Iter\s+(\d+)/(\d+)\].*loss: ([\d.]+)', line)

            if match:
                epoch = int(match.group(1))
                iter_num = int(match.group(3))
                loss = float(match.group(5))

                # 仅保留每个Epoch的最后一个Iteration（Iter iters_per_epoch/iters_per_epoch）
                if iter_num == iters_per_epoch:
                    epochs.append(epoch)
                    losses.append(loss)

    # 验证数据完整性（应与配置的epochs数量一致）
    if len(epochs) != total_epochs:
        print(f"警告：提取到{len(epochs)}个Epoch的损失数据，配置文件中epochs为{total_epochs}")

    return epochs, losses


def draw_loss_curve(epochs, losses, output_dir):
    """绘制损失曲线"""
    # 设置中文字体（macOS系统，Windows可改为SimHei）
    font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, label="训练损失")

    # 美化图表
    plt.title("训练损失曲线", fontproperties=font, fontsize=16)
    plt.xlabel("Epoch", fontproperties=font, fontsize=14)
    plt.ylabel("Loss", fontproperties=font, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(prop=font)

    # 标注每个数据点
    # for epoch, loss in zip(epochs, losses):
    #     plt.text(epoch, loss, f"({epoch}, {loss:.4f})",
    #              fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8))

    # 调整坐标轴
    interval = 10
    xticks = list(range(min(epochs) - 1, max(epochs) + 1, interval))
    plt.xticks(xticks)
    # plt.xticks(epochs)
    plt.xlim(min(epochs) - 1, max(epochs) + 1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(output_dir, "train_loss_curve.png")
    plt.savefig(plot_path, dpi=300)
    print(f"损失曲线已保存至: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
