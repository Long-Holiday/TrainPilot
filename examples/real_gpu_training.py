"""Real PyTorch GPU training simulation with TrainPilot HITL closed-loop recovery."""

import argparse
import logging
import os
import sys
import time
import torch
import torch.nn as nn

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from trainpilot.agent import StopTrainingException, TrainPilotClient, TrainingGuardian
from trainpilot.common.gateway import resolve_gateway_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealGPUTraining")


class SimpleMLP(nn.Module):
    """Lightweight neural network for GPU training demonstration."""

    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_training(
    gateway_url: str,
    api_token: str | None = None,
    task_id: str = "real-gpu-mlp",
    total_steps: int = 8,
    anomaly_step: int = 5,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"\n🚀 启动 PyTorch 模型训练任务: {task_id}")
    print(f"🖥️ 运行设备: {device_name} | 🔗 网关地址: {gateway_url}\n")

    model = SimpleMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 初始化 TrainPilot Client 与 Guardian
    client = TrainPilotClient(gateway_url=gateway_url, task_id=task_id, api_token=api_token)
    guardian = TrainingGuardian(client=client)

    loss_val = 0.0
    for step in range(1, total_steps + 1):
        inputs = torch.randn(64, 128, device=device)
        targets = torch.randint(0, 10, (64,), device=device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 阶段性上报 Checkpoint 里程碑
        if step == 3:
            vram = round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2) if device.type == "cuda" else 0.0
            print(f"💾 [Step {step}] Checkpoint 保存成功，上报里程碑...")
            client.notify_milestone(
                message=f"已保存 Step {step} 阶段性 Checkpoint",
                step=step,
                epoch=1,
                metrics={"loss": round(loss.item(), 4), "vram_mb": vram},
            )

        # 模拟第 5 步出现 Loss NaN 异常
        if step == anomaly_step:
            print(f"\n⚠️ [Step {step}] 模拟训练异常: 注入 NaN Loss！")
            loss = torch.tensor(float("nan"), device=device)
        else:
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        # 核心：优雅一行拦截异常，自动冻结现场并通知网关，挂起等待飞书决策（或服务端30s自动决策）并自动 ACK 恢复
        guardian.check_and_handle_loss(loss.item(), step=step)
        print(f"📊 [Step {step}/{total_steps}] 迭代完成 - Loss: {loss_val:.4f}")
        time.sleep(0.3)

    # 上报完成事件
    client.notify_event(
        event_type="completed",
        message="PyTorch 模型训练任务顺利完成！",
        step=total_steps,
        epoch=1,
        metrics={"final_loss": round(loss_val, 4)},
    )
    print("\n🎉 训练任务全部圆满完成！\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrainPilot Real GPU Training Example")
    parser.add_argument("--gateway", default=resolve_gateway_url(), help="TrainPilot Gateway URL")
    parser.add_argument("--token", default=os.environ.get("TRAINPILOT_API_TOKEN"), help="API Token")
    parser.add_argument("--task-id", default=f"real-gpu-{int(time.time())}", help="Unique Task ID")
    parser.add_argument("--steps", type=int, default=8, help="Total training steps")
    parser.add_argument("--anomaly-step", type=int, default=5, help="Step to simulate anomaly")
    args = parser.parse_args()

    try:
        run_training(
            gateway_url=args.gateway,
            api_token=args.token,
            task_id=args.task_id,
            total_steps=args.steps,
            anomaly_step=args.anomaly_step,
        )
    except StopTrainingException:
        print("\n🛑 收到飞书操作人员决策【停止训练】，任务已安全退出。")

