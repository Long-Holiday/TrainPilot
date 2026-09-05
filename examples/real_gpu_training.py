"""Real PyTorch GPU training simulation with TrainPilot HITL closed-loop recovery.

Demonstrates:
1. Real CUDA tensor operations and neural network training on NVIDIA GPU.
2. Real-time GPU VRAM telemetry reporting to TrainPilot Gateway.
3. Checkpoint snapshotting on GPU.
4. Loss NaN anomaly detection & live freezing.
5. Feishu interactive card dispatch to real Feishu chat.
6. HITL human decision polling, state rollback, LR decay, and ACK recovery.
7. Post-recovery training continuation to successful completion.
"""

import argparse
import copy
import logging
import os
import sys
import threading
import time
from typing import Optional

# Add src to python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import requests
import torch
import torch.nn as nn

from trainpilot.agent import TrainPilotClient, TrainingGuardian

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RealGPUTraining")


class SimpleMLP(nn.Module):
    """A lightweight neural network running on CUDA."""

    def __init__(self, in_features: int = 128, hidden_features: int = 256, out_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_gpu_vram_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2)
    return 0.0


def _auth_headers(api_token: Optional[str]) -> dict:
    """无 Token 时返回空 headers (开放开发模式), 避免发出 'Bearer None'。"""
    if api_token:
        return {"Authorization": f"Bearer {api_token}"}
    return {}


def run_real_gpu_training(
    gateway_url: str,
    api_token: Optional[str],
    task_id: str,
    total_steps: int = 8,
    checkpoint_step: int = 3,
    anomaly_step: int = 5,
    feishu_wait_seconds: float = 20.0,
):
    print("\n" + "=" * 60)
    print(f"🚀 TrainPilot 真实 GPU 训练与闭环测试启动")
    print(f"🔗 网关地址: {gateway_url}")
    if api_token:
        masked_token = '*' * (len(api_token) - 4) + api_token[-4:] if len(api_token) > 4 else '***'
    else:
        masked_token = '(未设置, 网关为开放开发模式)'
    print(f"🔑 鉴权 Token: {masked_token}")
    print(f"🏷️  任务 ID: {task_id}")
    print("=" * 60 + "\n")

    # 1. 检查 GPU 设备
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA GPU，本脚本要求真实 GPU 环境！")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram_mb = round(torch.cuda.get_device_properties(device).total_memory / (1024 * 1024), 2)
    print(f"🖥️  GPU 设备: {gpu_name} (总显存: {total_vram_mb} MB)")
    print(f"⚡ PyTorch 版本: {torch.__version__}, CUDA: {torch.version.cuda}\n")

    # 2. 初始化网络、优化器与数据
    model = SimpleMLP().to(device)
    initial_lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    criterion = nn.CrossEntropyLoss()

    # 固定伪数据流
    torch.manual_seed(42)
    batch_size = 64
    x_train = torch.randn(total_steps * 2, batch_size, 128, device=device)
    y_train = torch.randint(0, 10, (total_steps * 2, batch_size), device=device)

    # 3. 初始化 TrainPilot Client 与 Guardian
    client = TrainPilotClient(
        gateway_url=gateway_url,
        task_id=task_id,
        api_token=api_token,
        timeout=10,
    )
    guardian = TrainingGuardian(
        client=client,
        poll_interval=1.5,
        poll_timeout=120.0,
    )

    # 保存 Checkpoint 状态字典
    saved_checkpoint = {
        "step": 0,
        "model_state": None,
        "opt_state": None,
        "lr": initial_lr,
    }

    # 注册恢复动作回调
    def handle_reduce_lr_rollback(payload):
        print(f"\n[Guardian Callback] 📉 执行恢复策略: reduce_lr_rollback")
        if saved_checkpoint["model_state"] is not None:
            model.load_state_dict(saved_checkpoint["model_state"])
            optimizer.load_state_dict(saved_checkpoint["opt_state"])
            old_lr = saved_checkpoint["lr"]
            new_lr = old_lr * 0.5
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            saved_checkpoint["lr"] = new_lr
            current_step = saved_checkpoint["step"]
            print(f"[Guardian Callback] 🔄 已将模型与优化器回滚至 Step {current_step} 权重")
            print(f"[Guardian Callback] 📉 学习率从 {old_lr:.2e} 下调至 {new_lr:.2e}\n")
            return {
                "rollback_step": current_step,
                "new_lr": new_lr,
                "recovered_gpu_vram_mb": get_gpu_vram_mb(device),
            }
        else:
            print("[Guardian Callback] ⚠️ 未找到保存的 Checkpoint，保持当前模型并下调 LR")
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5
            return {"fallback": "reduced_lr_only"}

    guardian.register_action_handler("reduce_lr_rollback", handle_reduce_lr_rollback)

    # 4. 开始训练循环
    step = 0
    anomaly_triggered = False

    while step < total_steps:
        step += 1
        current_lr = optimizer.param_groups[0]["lr"]
        inputs = x_train[step - 1]
        targets = y_train[step - 1]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 检查是否保存 Checkpoint
        if step == checkpoint_step:
            saved_checkpoint["step"] = step
            saved_checkpoint["model_state"] = copy.deepcopy(model.state_dict())
            saved_checkpoint["opt_state"] = copy.deepcopy(optimizer.state_dict())
            saved_checkpoint["lr"] = current_lr
            vram_mb = get_gpu_vram_mb(device)
            print(f"💾 [Step {step}] Checkpoint 保存成功！当前 GPU 显存占用: {vram_mb} MB")

            # 上报阶段里程碑 (网关将向飞书推送绿色只读卡片)
            print(f"📤 [Step {step}] 向 TrainPilot 网关上报 Milestone (触发飞书里程碑通知)...")
            client.notify_milestone(
                message=f"已保存 Step {step} 阶段性 Checkpoint",
                step=step,
                epoch=1,
                metrics={
                    "loss": round(loss.item(), 4),
                    "lr": current_lr,
                    "gpu_vram_mb": vram_mb,
                    "gpu_device": gpu_name,
                },
            )

        # 检查是否触发异常 (仅触发一次)
        if step == anomaly_step and not anomaly_triggered:
            anomaly_triggered = True
            print("\n" + "!" * 60)
            print(f"⚠️  [Step {step}] 模拟训练异常发生: 注入 NaN Loss！")
            print(f"⚠️  现场将被冻结，向网关发送告警并触发飞书交互式卡片...")
            print("!" * 60)

            loss_val = float("nan")

            # 启动辅助决策线程：
            # 如果配置了 feishu_wait_seconds > 0，先等待用户在真实飞书中点击卡片按钮；
            # 若在设定等待时间内用户未点击，则自动通过网关决策接口注入决策，确保测试顺畅闭环。
            def decision_watcher():
                if feishu_wait_seconds > 0:
                    print(f"\n[飞书交互提醒] 📲 告警卡片已推送到飞书群！")
                    print(f"[飞书交互提醒] 您可以在飞书群中点击卡片上的【回滚Checkpoint并降LR】按钮进行真实决策。")
                    print(f"[飞书交互提醒] 将等待 {feishu_wait_seconds:.0f} 秒... 如未点击将自动兜底注入决策。\n")

                start_wait = time.time()
                while time.time() - start_wait < feishu_wait_seconds:
                    # 检查是否已经有人在飞书点击了
                    try:
                        resp = requests.get(
                            f"{gateway_url}/api/tasks/{task_id}/status",
                            headers=_auth_headers(api_token),
                            timeout=3,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("state") in ("RESOLVED", "RECOVERING", "RUNNING"):
                                print(f"\n[飞书交互成功] 🎉 检测到已在飞书端或外部下发决策！状态: {data.get('state')}")
                                return
                    except Exception:
                        pass
                    time.sleep(2.0)

                # 超时后自动兜底提交决策
                print("\n[自动兜底决策] 🤖 等待超时，自动向网关注入 'reduce_lr_rollback' 决策以闭环恢复...")
                try:
                    res = requests.post(
                        f"{gateway_url}/api/tasks/{task_id}/decision",
                        headers=_auth_headers(api_token),
                        json={
                            "task_id": task_id,
                            "action": "reduce_lr_rollback",
                            "operator": "Auto-Test-Fallback (or Feishu Timeout)",
                        },
                        timeout=5,
                    )
                    if res.status_code == 200:
                        print("[自动兜底决策] ✅ 决策注入成功，状态已流转至 RESOLVED！")
                    else:
                        print(f"[自动兜底决策] ⚠️ 注入响应状态码: {res.status_code}")
                except Exception as err:
                    print(f"[自动兜底决策] ❌ 决策注入请求异常: {err}")

            t = threading.Thread(target=decision_watcher, daemon=True)
            t.start()
        else:
            loss.backward()
            optimizer.step()
            loss_val = loss.item()

        # 通过 Guardian 进行异常拦截与闭环监控
        vram_mb = get_gpu_vram_mb(device)
        guardian.check_and_handle_loss(
            loss_val=loss_val,
            step=step,
            epoch=1,
            extra_metrics={
                "lr": current_lr,
                "gpu_vram_mb": vram_mb,
                "gpu_device": gpu_name,
            },
        )

        # 若发生了异常回滚，将当前 step 重置为回滚的目标 step
        if anomaly_triggered and step == anomaly_step:
            print(f"🔄 [Step {step}] 异常恢复完毕，将 step 从 {step} 重置为 Checkpoint 步骤 {saved_checkpoint['step']}")
            step = saved_checkpoint["step"]
            # 恢复后继续训练，跳过该步的正常打印
            continue

        # 正常训练步
        print(f"📊 [Step {step}/{total_steps}] 正常训练步完成 - Loss: {loss_val:.4f}, LR: {current_lr:.2e}, GPU显存: {vram_mb} MB")
        time.sleep(0.4)

    # 5. 完成训练上报
    vram_mb = get_gpu_vram_mb(device)
    print(f"\n📤 上报训练完成事件 (Completed)...")
    client.notify_event(
        event_type="completed",
        message="GPU 深度学习模型训练任务顺利完成！",
        step=step,
        epoch=1,
        metrics={
            "final_loss": round(loss_val, 4),
            "final_lr": optimizer.param_groups[0]["lr"],
            "final_vram_mb": vram_mb,
            "gpu_device": gpu_name,
        },
    )

    print("\n" + "=" * 60)
    print("🎉 真实 GPU 训练与闭环恢复流程执行完毕！")
    print("=" * 60)

    # 6. 查询网关任务最终全量状态与事件流
    time.sleep(1.0)
    status_resp = requests.get(
        f"{gateway_url}/api/tasks/{task_id}/status",
        headers=_auth_headers(api_token),
        timeout=5,
    )
    if status_resp.status_code == 200:
        summary = status_resp.json()
        print("\n📋 【网关任务最终状态报告】")
        print(f"• 任务 ID: {summary.get('task_id')}")
        print(f"• 最终状态: {summary.get('state')}")
        print(f"• 事件总数: {summary.get('events_count')}")
        print(f"• 当前步数: {summary.get('current_step')}")
        print(f"• 最新指标: {summary.get('latest_metrics')}")
        print(f"• 创建时间: {summary.get('created_at')}")
        print(f"• 更新时间: {summary.get('updated_at')}")

    events_resp = requests.get(
        f"{gateway_url}/api/tasks/{task_id}/events",
        headers=_auth_headers(api_token),
        timeout=5,
    )
    if events_resp.status_code == 200:
        events_data = events_resp.json()
        print(f"\n📜 【网关记录的审计事件流 (共 {events_data.get('total')} 条)】")
        for idx, ev in enumerate(events_data.get("events", []), 1):
            ev_type = ev.get("event_type")
            ev_msg = ev.get("message")
            ev_step = ev.get("step")
            ev_metrics = ev.get("metrics")
            print(f"  {idx}. [{ev_type.upper()}] Step {ev_step}: {ev_msg} (Metrics: {ev_metrics})")


if __name__ == "__main__":
    from trainpilot.common.gateway import resolve_gateway_url

    parser = argparse.ArgumentParser(description="TrainPilot Real GPU Training Test")
    parser.add_argument(
        "--gateway",
        default=resolve_gateway_url(),
        help="TrainPilot Gateway URL (默认: $TRAINPILOT_GATEWAY_URL > http://$TRAINPILOT_HOST:$TRAINPILOT_PORT; GPU 侧设置 TRAINPILOT_HOST=Web公网IP 即可)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("TRAINPILOT_API_TOKEN"),
        help="TrainPilot API Token (建议通过环境变量 TRAINPILOT_API_TOKEN 传入)",
    )
    parser.add_argument(
        "--task-id",
        default=f"real-gpu-rtx3050-{int(time.time())}",
        help="Unique task ID for this test",
    )
    parser.add_argument(
        "--feishu-wait",
        type=float,
        default=15.0,
        help="Seconds to wait for Feishu card manual click before auto-fallback",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=7,
        help="Total training steps",
    )

    args = parser.parse_args()
    run_real_gpu_training(
        gateway_url=args.gateway,
        api_token=args.token,
        task_id=args.task_id,
        total_steps=args.steps,
        checkpoint_step=3,
        anomaly_step=5,
        feishu_wait_seconds=args.feishu_wait,
    )
