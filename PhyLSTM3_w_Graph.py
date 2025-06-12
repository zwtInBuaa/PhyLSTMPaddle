# PhyLSTM3_w_Graph.py
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""
PhyLSTM (Physics-Informed LSTM) 模型实现
Reference: https://github.com/zhry10/PhyLSTM.git
"""

import os
import paddle
import numpy as np
import ppsci
import scipy.io
import yaml

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

from os import path as osp
from omegaconf import OmegaConf

import hydra
import functions
from omegaconf import DictConfig
from ppsci import arch, constraint, validate, optimizer, solver
from ppsci.utils import logger, misc
from ppsci.loss import FunctionalLoss
from ppsci.metric import FunctionalMetric



def train(cfg: DictConfig):
    """模型训练函数（保持不变）"""
    # 设置随机种子
    misc.set_random_seed(cfg.seed)
    # 初始化日志
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")
    logger.info(f"Training with config: {cfg}")

    # 加载数据（保持不变）
    mat = scipy.io.loadmat(cfg.DATA_FILE_PATH)
    t = mat["time"]
    dt = 0.02
    n1 = int(dt / 0.005)
    t = t[::n1]

    # 处理输入数据（保持不变）
    ag_data = mat["input_tf"][:, ::n1].astype(np.float32)
    u_data = mat["target_X_tf"][:, ::n1].astype(np.float32)
    ut_data = mat["target_Xd_tf"][:, ::n1].astype(np.float32)
    utt_data = mat["target_Xdd_tf"][:, ::n1].astype(np.float32)
    ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
    u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
    ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
    utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])

    # 处理预测数据（保持不变）
    ag_pred = mat["input_pred_tf"][:, ::n1].astype(np.float32)
    u_pred = mat["target_pred_X_tf"][:, ::n1].astype(np.float32)
    ut_pred = mat["target_pred_Xd_tf"][:, ::n1].astype(np.float32)
    utt_pred = mat["target_pred_Xdd_tf"][:, ::n1].astype(np.float32)
    ag_pred = ag_pred.reshape([ag_pred.shape[0], ag_pred.shape[1], 1])
    u_pred = u_pred.reshape([u_pred.shape[0], u_pred.shape[1], 1])
    ut_pred = ut_pred.reshape([ut_pred.shape[0], ut_pred.shape[1], 1])
    utt_pred = utt_pred.reshape([utt_pred.shape[0], utt_pred.shape[1], 1])

    # 构建物理约束矩阵（保持不变）
    N = u_data.shape[1]
    phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([N - 3])]).astype(np.float32)
    temp1 = np.concatenate([-1 / 2 * np.eye(N - 2), np.zeros([N - 2, 2])], axis=1).astype(np.float32)
    temp2 = np.concatenate([np.zeros([N - 2, 2]), 1 / 2 * np.eye(N - 2)], axis=1).astype(np.float32)
    phi2 = temp1 + temp2
    phi3 = np.concatenate([np.zeros([N - 3]), np.array([1 / 2, -2, 3 / 2])]).astype(np.float32)
    phi_t0 = (1 / dt * np.concatenate([
        phi1.reshape(1, -1),
        phi2,
        phi3.reshape(1, -1)
    ], axis=0)).reshape(1, N, N).astype(np.float32)

    # 数据准备（保持不变）
    ag_star = ag_data
    eta_star = u_data
    eta_t_star = ut_data
    eta_tt_star = utt_data
    ag_c_star = np.concatenate([ag_data, ag_pred[0:53]])
    lift_star = -ag_c_star

    eta = eta_star
    ag = ag_star
    lift = lift_star
    eta_t = eta_t_star
    eta_tt = eta_tt_star
    g = -eta_tt - ag
    g_pred = -utt_pred - ag_pred
    ag_c = ag_c_star
    phi_t = np.repeat(phi_t0, ag_c_star.shape[0], axis=0)

    # 创建模型（保持不变）
    model = arch.DeepPhyLSTM(
        cfg.MODEL.input_size,
        eta.shape[2],
        cfg.MODEL.hidden_size,
        cfg.MODEL.model_type,
    )
    model.register_input_transform(functions.transform_in)
    model.register_output_transform(functions.transform_out)

    # 准备数据集（保持不变）
    dataset_obj = functions.Dataset(eta, eta_t, g, ag, ag_c, lift, phi_t)
    input_dict_train, label_dict_train, input_dict_val, label_dict_val = dataset_obj.get(cfg.TRAIN.epochs)

    # 定义约束条件（保持不变）
    sup_constraint_pde = constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_train,
                "label": label_dict_train,
            },
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
            "batch_size": 1,
            "num_workers": 0,
        },
        FunctionalLoss(functions.train_loss_func3),
        {
            "eta_pred": lambda out: out["eta_pred"],
            "eta_dot_pred": lambda out: out["eta_dot_pred"],
            "g_pred": lambda out: out["g_pred"],
            "eta_t_pred_c": lambda out: out["eta_t_pred_c"],
            "eta_dot_pred_c": lambda out: out["eta_dot_pred_c"],
            "lift_pred_c": lambda out: out["lift_pred_c"],
            "g_t_pred_c": lambda out: out["g_t_pred_c"],
            "g_dot_pred_c": lambda out: out["g_dot_pred_c"],
        },
        name="sup_train",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    # 定义验证器（保持不变）
    sup_validator_pde = validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_val,
                "label": label_dict_val,
            },
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
        },
        FunctionalLoss(functions.train_loss_func3),
        {
            "eta_pred": lambda out: out["eta_pred"],
            "eta_dot_pred": lambda out: out["eta_dot_pred"],
            "g_pred": lambda out: out["g_pred"],
            "eta_t_pred_c": lambda out: out["eta_t_pred_c"],
            "eta_dot_pred_c": lambda out: out["eta_dot_pred_c"],
            "lift_pred_c": lambda out: out["lift_pred_c"],
            "g_t_pred_c": lambda out: out["g_t_pred_c"],
            "g_dot_pred_c": lambda out: out["g_dot_pred_c"],
        },
        metric={"metric": FunctionalMetric(functions.metric_expr)},
        name="sup_valid",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # 初始化优化器和求解器（保持不变）
    optimizer_obj = optimizer.Adam(cfg.TRAIN.learning_rate)(model)
    solver_obj = solver.Solver(
        model,
        constraint_pde,
        cfg.output_dir,
        optimizer_obj,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        validator=validator_pde,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # 开始训练（保持不变）
    logger.info("Starting training...")
    solver_obj.train()

    # 提取训练和验证数据的真实值（从输入字典和处理后的数据中获取）
    # ---------------------------- 训练集真实值 ----------------------------
    train_true_y = input_dict_train["eta"]  # eta真实值，形状: [batch, time]
    train_true_yt = input_dict_train["eta_t"]  # eta_t真实值（一阶导数）
    # train_true_g = -eta_tt - ag  # g真实值，公式: g = -eta_tt - ag
    train_true_g = label_dict_train["g"]

    # ---------------------------- 验证集真实值 ----------------------------
    val_true_y = input_dict_val["eta"]
    val_true_yt = input_dict_val["eta_t"]
    val_true_g = label_dict_val["g"]

    # ---------------------------- 测试集真值 ----------------------------
    pred_true_y = u_pred
    pred_true_yt = ut_pred
    pred_true_g = g_pred # -utt_pred - ag_pred

    # ---------------------------- 模型推理获取预测值 ----------------------------
    # 训练集预测
    with paddle.no_grad():
        train_pred = solver_obj.predict(input_dict_train)  # 使用训练好的模型推理训练集
    train_pred_y = train_pred["eta_pred"].numpy().astype(np.float32)  # eta预测值
    train_pred_yt = train_pred["eta_dot_pred"].numpy().astype(np.float32)  # eta_t预测值（一阶导数）
    train_pred_g = train_pred["g_pred"].numpy().astype(np.float32)  # g预测值
    # print(train_pred_y.type)

    # 验证集预测
    with paddle.no_grad():
        val_pred = solver_obj.predict(input_dict_val)  # 推理验证集
    val_pred_y = val_pred["eta_pred"].numpy().astype(np.float32)
    val_pred_yt = val_pred["eta_dot_pred"].numpy().astype(np.float32)
    val_pred_g = val_pred["g_pred"].numpy().astype(np.float32)


    '''
    input_dict_train = {
            "eta": self.eta[indices[:split_idx]],
            "eta_t": self.eta_t[indices[:split_idx]],
            "g": self.g[indices[:split_idx]],
            "ag": self.ag[indices[:split_idx]],
            "ag_c": self.ag_c,
            "lift": self.lift,
            "phi_t": self.phi_t[indices[:split_idx]],
        }
    '''
    # 准备预测数据的推理
    input_dict_pred = {
        "ag": paddle.to_tensor(ag_pred),
        "ag_c": paddle.to_tensor(ag_c_star),
        "phi": paddle.to_tensor(phi_t[:ag_pred.shape[0]]),
        # "eta": paddle.to_tensor(u_pred),
        # "eta_t": paddle.to_tensor(ut_pred),
        # "g": paddle.to_tensor(-utt_pred - ag_pred)
    }

    '''
    model_predict return:
    return {
            "eta_pred": eta_pred,
            "eta_dot_pred": eta_dot_pred,
            "g_pred": g_pred,
            "eta_t_pred_c": eta_t_pred_c,
            "eta_dot_pred_c": eta_dot_pred_c,
            "lift_pred_c": lift_pred_c,
            "g_t_pred_c": g_t_pred_c,
            "g_dot_pred_c": g_dot_pred_c,
        }
    '''

    # 测试集合
    with paddle.no_grad():
        pred_pred = solver_obj.predict(input_dict_pred)

    pred_pred_y = pred_pred["eta_pred"].numpy()
    pred_pred_yt = pred_pred["eta_dot_pred"].numpy()
    pred_pred_g = pred_pred["g_pred"].numpy()

    # 生成时间轴（保持不变）
    time_train = np.arange(train_true_y.shape[1]) * dt
    time_val = np.arange(val_true_y.shape[1]) * dt
    time_pred = np.arange(pred_true_y.shape[1]) * dt

    functions.plot_all_results(cfg, 'train', train_true_y, train_pred_y, train_true_yt, train_pred_yt, train_true_g, train_pred_g, time_train)
    functions.plot_all_results(cfg, 'val', val_true_y, val_pred_y, val_true_yt, val_pred_yt, val_true_g, val_pred_g, time_val)
    functions.plot_all_results(cfg, 'pred', pred_true_y, pred_pred_y, pred_true_yt, pred_pred_yt, pred_true_g, pred_pred_g, time_pred)

    # 保存模型（保持不变）
    model_save_dir = osp.join(cfg.output_dir, cfg.TRAIN.model_save_dir)
    final_model_path = osp.join(model_save_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    paddle.save(model.state_dict(), osp.join(final_model_path, "model.pdparams"))
    with open(osp.join(final_model_path, "config.yaml"), 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)
    logger.info(f"Training completed. Model saved to {final_model_path}")


def evaluate(cfg: DictConfig):
    """模型评估函数（保持不变）"""
    # 设置随机种子
    misc.set_random_seed(cfg.seed)
    # 初始化日志
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")
    logger.info(f"Evaluation with config: {cfg}")

    # 加载数据
    mat = scipy.io.loadmat(cfg.DATA_FILE_PATH)
    t = mat["time"]
    dt = 0.02
    n1 = int(dt / 0.005)
    t = t[::n1]

    # 处理输入数据
    ag_data = mat["input_tf"][:, ::n1].astype(np.float32)
    u_data = mat["target_X_tf"][:, ::n1].astype(np.float32)
    ut_data = mat["target_Xd_tf"][:, ::n1].astype(np.float32)
    utt_data = mat["target_Xdd_tf"][:, ::n1].astype(np.float32)
    ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
    u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
    ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
    utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])

    # 处理预测数据
    ag_pred = mat["input_pred_tf"][:, ::n1].astype(np.float32)
    u_pred = mat["target_pred_X_tf"][:, ::n1].astype(np.float32)
    ut_pred = mat["target_pred_Xd_tf"][:, ::n1].astype(np.float32)
    utt_pred = mat["target_pred_Xdd_tf"][:, ::n1].astype(np.float32)
    ag_pred = ag_pred.reshape([ag_pred.shape[0], ag_pred.shape[1], 1])
    u_pred = u_pred.reshape([u_pred.shape[0], u_pred.shape[1], 1])
    ut_pred = ut_pred.reshape([ut_pred.shape[0], ut_pred.shape[1], 1])
    utt_pred = utt_pred.reshape([utt_pred.shape[0], utt_pred.shape[1], 1])

    # 构建物理约束矩阵
    N = u_data.shape[1]
    phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([N - 3])]).astype(np.float32)
    temp1 = np.concatenate([-1 / 2 * np.eye(N - 2), np.zeros([N - 2, 2])], axis=1).astype(np.float32)
    temp2 = np.concatenate([np.zeros([N - 2, 2]), 1 / 2 * np.eye(N - 2)], axis=1).astype(np.float32)
    phi2 = temp1 + temp2
    phi3 = np.concatenate([np.zeros([N - 3]), np.array([1 / 2, -2, 3 / 2])]).astype(np.float32)
    phi_t0 = (1 / dt * np.concatenate([
        phi1.reshape(1, -1),
        phi2,
        phi3.reshape(1, -1)
    ], axis=0)).reshape(1, N, N).astype(np.float32)

    # 数据准备
    ag_star = ag_data
    eta_star = u_data
    eta_t_star = ut_data
    eta_tt_star = utt_data
    ag_c_star = np.concatenate([ag_data, ag_pred[0:53]])
    lift_star = -ag_c_star

    eta = eta_star
    ag = ag_star
    lift = lift_star
    eta_t = eta_t_star
    eta_tt = eta_tt_star
    g = -eta_tt - ag
    ag_c = ag_c_star
    phi_t = np.repeat(phi_t0, ag_c_star.shape[0], axis=0)

    # 创建模型
    model = arch.DeepPhyLSTM(
        cfg.MODEL.input_size,
        eta.shape[2],
        cfg.MODEL.hidden_size,
        cfg.MODEL.model_type,
    )
    model.register_input_transform(functions.transform_in)
    model.register_output_transform(functions.transform_out)

    # 准备验证数据集
    dataset_obj = functions.Dataset(eta, eta_t, g, ag, ag_c, lift, phi_t)
    _, _, input_dict_val, label_dict_val = dataset_obj.get(1)

    # 定义验证器
    sup_validator_pde = validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_val,
                "label": label_dict_val,
            },
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
        },
        FunctionalLoss(functions.train_loss_func3),
        {
            "eta_pred": lambda out: out["eta_pred"],
            "eta_dot_pred": lambda out: out["eta_dot_pred"],
            "g_pred": lambda out: out["g_pred"],
            "eta_t_pred_c": lambda out: out["eta_t_pred_c"],
            "eta_dot_pred_c": lambda out: out["eta_dot_pred_c"],
            "lift_pred_c": lambda out: out["lift_pred_c"],
            "g_t_pred_c": lambda out: out["g_t_pred_c"],
            "g_dot_pred_c": lambda out: out["g_dot_pred_c"],
        },
        metric={"metric": FunctionalMetric(functions.metric_expr)},
        name="sup_valid",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # 加载预训练模型
    if cfg.EVAL.pretrained_model_path:
        model_path = cfg.EVAL.pretrained_model_path
        logger.info(f"Loading model from {model_path}")
        try:
            state_dict = paddle.load(osp.join(model_path, "model.pdparams"))
            model.set_dict(state_dict)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    else:
        logger.warning("No pretrained model specified. Using initialized model.")

    # 初始化求解器
    solver_obj = solver.Solver(
        model,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        validator=validator_pde,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # 评估模型
    logger.info("Starting evaluation...")
    solver_obj.eval()


def infer(cfg: DictConfig):
    """模型推理函数（保持不变）"""
    # 初始化日志
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "infer.log"), "info")
    logger.info(f"Inference with config: {cfg}")

    # 确保指定了预训练模型路径
    if not cfg.EVAL.pretrained_model_path:
        raise ValueError("pretrained_model_path must be specified for inference")

    # 加载配置文件
    config_path = osp.join(cfg.EVAL.pretrained_model_path, "config.yaml")
    if osp.exists(config_path):
        with open(config_path, 'r') as f:
            model_cfg = yaml.safe_load(f)
        cfg = {**cfg, **model_cfg}
        cfg = DictConfig(cfg)
        logger.info("Loaded model configuration")
    else:
        logger.warning("No config.yaml found, using provided config")

    # 加载数据
    mat = scipy.io.loadmat(cfg.DATA_FILE_PATH)
    t = mat["time"]
    dt = 0.02
    n1 = int(dt / 0.005)
    t = t[::n1]

    ag_data = mat["input_tf"][:, ::n1]
    ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])

    N = ag_data.shape[1]
    phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([N - 3])])
    temp1 = np.concatenate([-1 / 2 * np.eye(N - 2), np.zeros([N - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([N - 2, 2]), 1 / 2 * np.eye(N - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate([np.zeros([N - 3]), np.array([1 / 2, -2, 3 / 2])])
    phi_t0 = (1 / dt * np.concatenate([
        phi1.reshape(1, -1),
        phi2,
        phi3.reshape(1, -1)
    ], axis=0)).reshape(1, N, N)
    phi_t = np.repeat(phi_t0, ag_data.shape[0], axis=0)

    # 创建模型
    model = arch.DeepPhyLSTM(
        cfg.MODEL.input_size,
        1,
        cfg.MODEL.hidden_size,
        cfg.MODEL.model_type,
    )
    model.register_input_transform(functions.transform_in)
    model.register_output_transform(functions.transform_out)

    # 加载模型参数
    model_path = cfg.EVAL.pretrained_model_path
    logger.info(f"Loading inference model from {model_path}")
    try:
        state_dict = paddle.load(osp.join(model_path, "model.pdparams"))
        model.set_dict(state_dict)
        model.eval()
        logger.info("Inference model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load inference model: {e}")
        raise

    # 准备推理输入数据
    input_data = {
        "ag": ag_data,
        "ag_c": ag_data,
        "phi_t": phi_t,
        "eta": np.zeros_like(ag_data),
        "eta_t": np.zeros_like(ag_data),
        "g": np.zeros_like(ag_data),
    }

    # 进行推理
    logger.info("Starting inference...")
    with paddle.no_grad():
        inputs = functions.transform_in(input_data)
        outputs = model(inputs)
        results = functions.transform_out(outputs)

    # 保存推理结果
    results_dir = osp.join(cfg.output_dir, "inference_results")
    os.makedirs(results_dir, exist_ok=True)
    result_file = osp.join(results_dir, "inference_results.mat")
    scipy.io.savemat(result_file, results)
    logger.info(f"Inference completed. Results saved to {result_file}")


@hydra.main(version_base=None, config_path="./conf", config_name="phylstm3")
def main(cfg: DictConfig):
    """主函数（保持不变）"""
    # paddle.set_default_dtype("float64")
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "main.log"), "info")
    logger.info(f"Running in {cfg.mode} mode with config: {cfg}")

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "infer":
        infer(cfg)
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}. Expected 'train', 'eval', or 'infer'")


if __name__ == "__main__":
    main()
