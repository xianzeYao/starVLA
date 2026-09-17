# ARX CoT 部署客户端

本目录面向同一套双臂 ARX LeRobot 记录约定，不区分 v1/v2。三路图按
`camera_l, camera_r, camera_h` 发送；服务端负责动作反归一化，客户端仅沿用
旧 ARX 客户端的 RGB 转换、双臂动作重排和 chunk 边界混合。模型输出的
连续夹爪值不在这里重新映射或限幅。

## GPU 服务器

模型目录需包含 `final_model/pytorch_model.pt`、`config.yaml`、
`dataset_statistics.json`。目前 v2 80k 权重放在
`/root/data/yxz/models/arx-cot-v2-80k/`。发布的配置原件保存在
`config.hf-original.yaml`；本机的 `config.yaml` 仅调整了两处：

- `framework.qwenvl.base_vlm` 指向本机的 Qwen3.5-4B。
- `datasets.vla_data.obs_image_size: [224, 224]`，使服务端输入预处理
  和训练加载器一致。客户端仍发送原始 480×640 RGB 图像。

在仓库根目录启动现有 CoT 服务端：

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m deployment.model_server.server_policy \
  --ckpt_path /root/data/yxz/models/arx-cot-v2-80k/final_model/pytorch_model.pt \
  --port 10093 --use_bf16
```

## 数据回放 smoke

这是独立入口，不导入 ROS，不创建 ARX 环境，也不发送机器人动作。
`--dataset_root` 可指向任何相同 ARX schema 的 LeRobot 数据集。下面只是当前
v2 数据的例子，换成 v1 或后续同本体数据无需改代码：

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m deployment.model_server.arx4cot.client_policy_arx_smoke \
  --dataset_root /root/data/yxz/datasets/arx_cot_sweep_v2_lerobot \
  --episode_index 0 --policy_host 127.0.0.1 --policy_port 10093 \
  --execute_horizon 20 --max_episode_steps 20 \
  --output_dir /tmp/arx4cot-smoke
```

结果写入 `summary.json` 和 `records.json`。任务提示词从 episode 元数据读取，
与训练一致。输出的 MAE 只用于检查数据路径和形状，不是策略质量评测。

## Mac 真机端

把本目录复制到已验证的 Mac 工程 `Deployment/model_server/arx4cot/`；
旧 `tools`、`ARXRobotEnv` 和 ROS2 环境不改。确保旧工具包和 ROS2 依赖可用，
并先让 Mac 能连接 GPU 服务器（必要时使用 SSH 端口转发）。真机执行入口：

```bash
python -m Deployment.model_server.arx4cot.client_policy_arx \
  --policy_host 127.0.0.1 --policy_port 10093 \
  --task_prompt "Sweep the green cub into the U-shaped target area." \
  --execute_horizon 20 --max_episode_steps 200
```

`execute_horizon` 是每次实际下发的步数；模型 chunk 长度从服务端握手读取。
当前模型预测 30 步、默认只执行前 20 步再重新规划。真机脚本会调用
`ARXRobotEnv.reset`、`step_lift`、`step_raw_joint`；只有 smoke 入口保证不动机器人。
