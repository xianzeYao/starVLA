# ARX CoT 部署设计

## 目标与边界

在当前 CoT 分支新增 `deployment/model_server/arx4cot/`，让双臂 ARX 的
QwenCoTv2_arx 策略先用录制数据验证，再由用户在 Mac 真机端验证。
上传的 `deployment/arx4cot-reference/` 保持原样，不纳入提交；原有
`arx`、`arx4process`、WebSocket `tools` 和 `ARXRobotEnv` 均不修改。
自动化 smoke test 不向机器人下发命令。

## 服务端与反归一化

沿用当前 CoT 的 `deployment/model_server/server_policy.py`，不复制旧
`server_policy_arx.py`。服务端根据 checkpoint 旁的
`dataset_statistics.json` 和训练时的 ARX transform 对全部 14 维动作
反归一化，响应 `data.actions` 为原始单位，形状为 `[1, chunk, 14]`。
新客户端只读取该字段，不再次反归一化，也不对夹爪做二值化或重映射。

沿用旧客户端的夹爪连续值处理和通用 chunk 边界混合；不额外限幅、
阈值化或加入夹爪专用逻辑。当前 CoT 服务端的 min-max inverse
不裁剪输入，真机调试时仍需留意实际输出范围。

## 观察输入与双臂动作

只支持双臂。每次请求是
`examples=[{"image": [camera_l, camera_r, camera_h], "lang": task}]`。
三路图按训练顺序排列，由旧客户端完成 BGR→RGB，发送原始的
480×640 RGB `uint8`；不传 state 或深度。训练数据加载器会先把图像
缩放到 224×224，再交给 Qwen 图像处理器。发布配置缺少
`datasets.vla_data.obs_image_size`，因此部署用的本地配置需补上
`obs_image_size: [224, 224]`，由现有模型输入预处理在服务端完成
同样的缩放，客户端不再重复缩放。`task` 使用数据集原始任务文本，由保存的模型配置套
`Your task is {instruction}.` 模板，客户端不重复套模板。

连接时读取服务端 `action_chunk_size`；每次检查响应状态、batch=1、
形状 `[chunk,14]` 和有限数值。
模型动作顺序是 `[左臂6, 右臂6, 左夹爪, 右夹爪]`，下发前转为
`[左臂6, 左夹爪, 右臂6, 右夹爪]`。沿用上传的 `arx/client_utils.py`
里的取图、BGR→RGB、重排、chunk 边界混合和 ROS payload 逻辑。

## 预测 30 步、执行 20 步

当前两次训练的 action chunk 均为 30 步；部署参数
`execute_horizon` 是每次实际执行的步数，可以小于服务端的 chunk。
按本次讨论默认设为 20：执行前 20 步，丢弃剩余 10 步，重新取图
并预测下一段。保持 20 Hz（`control_dt=0.05`）和旧 ARX
客户端的 3 步 chunk 边界混合；允许把 `execute_horizon` 设为
1～服务端公布的 chunk 大小。
同步查询会在 chunk 边界等待服务端，smoke 需记录推理延迟，
真机连通测试需观察可能的控制停顿。

## 数据回放与真机客户端

`arx4cot` 包含共享协议辅助代码、数据回放 smoke 客户端、真机执行
客户端、启动示例和 README。Mac 端继续用已有的
`Deployment.model_server.tools.WebsocketClientPolicy`；只需把新目录
复制到 `Deployment/model_server/arx4cot/`，不用替换 `tools`。

smoke 接受任意同一 ARX schema 的 LeRobot 数据集根目录，v1 和 v2
只是两份验证样本。读取器统一使用 `meta/episodes.jsonl`、
逐 episode Parquet 和三路视频；旧
`arx4process` fallback reader 使用另一种 episodes Parquet 布局，
不能直接复制。新 reader 读取该 episode 的任务、对齐的 RGB 帧和
原始 14 维 action，不读取深度或 UVD。预测动作转为机器人顺序后
与 GT 比较，保存逐维误差与请求耗时。

“真机执行客户端”指实例化 `ARXRobotEnv` 并调用
`reset`、`step_lift`、`step_raw_joint` 等现有方法的脚本，不是
模型服务端。沿用旧 ARX 的类实例化和动作循环，不修改 ROS2
环境实现；真机执行与 smoke 使用不同入口。smoke 是独立
入口，不导入真机模块，不构造 ARX 环境或发送动作。

验证顺序：请求／响应及维度单元测试；v1、v2 真实数据读取；
真实 checkpoint 加单帧 WebSocket 推理；可选整段数据回放。
以上均不初始化 ROS2。通过后由用户在 Mac 上先测网络连通，
最后才显式运行真机动作。离线成功不等于物理安全验证通过。

## 权重文件

先只下载 `yaoxianze/arx-cot-v2-80k` 的
`final_model/pytorch_model.pt`（80k，约 10.7 GB）、`config.yaml`
和 `dataset_statistics.json` 到共享盘
`/root/data/yxz/models/arx-cot-v2-80k/`。不下载
`checkpoints/steps_40000_pytorch_model.pt` 或
`checkpoints/steps_60000_pytorch_model.pt`；中间权重仅在后续比较时
按需获取。`config.full.yaml` 和训练日志不是本次推理必需文件。

模型加载器要求 `.pt` 位于 run 根目录下一层，同时在 run 根目录
读取 `config.yaml` 与 `dataset_statistics.json`；后者是服务端
反归一化必需的。发布配置里的 `framework.qwenvl.base_vlm`
仍指向训练机，部署副本需改为已有的
`/root/data/yxz/models/Qwen3.5-4B`，保留原始配置备查。
本机 `/home/yxz/CoT` 工作盘仅约 4.2 GB 可用，不放权重。

## 交付范围

交付新 `arx4cot` 目录、验证结果和准确的服务端、smoke、
真机启动命令。若 Mac 到 GPU 服务器无法直连，用 SSH 隧道
解决网络连通，不改 WebSocket 协议。
