# TrainPilot 代码库精简与架构统一设计规范

- **日期**：2026-09-06
- **状态**：已批准 (Approved)
- **目标**：整理 TrainPilot 项目，消除冗余兜底逻辑与多种冗余接入方式，使代码干净易读、心智模型清晰。

---

## 1. 背景与核心问题

在 TrainPilot 的历史迭代中，由于兼顾不同阶段的需求，系统积累了多处重叠实现和多层冗余兜底：

1. **训练端接入方式多头并存**：
   - 同时存在 `TrainPilotClient`、`TrainingGuardian` 和 `TrainPilotPyTorchHook` 三层抽象；
   - `TrainPilotPyTorchHook` 内部夹带了基于静态规则的 `build_default_agent_note` 兜底生成器，与外部 AI 智能体自主撰写点评的设计初衷背道而驰。
2. **多端超时兜底竞争与抢跑**：
   - 控制面服务端具备 30s 自动 `self_resolve` 定时器；
   - 客户端 `TrainingGuardian` 又设置了本地 `timeout_fallback_action="self_resolve"`；
   - 示例脚本 `real_gpu_training.py` 中还启动了外挂后台线程 `decision_watcher` 主动调用接口注入兜底；
   - 三处兜底逻辑互相重叠，产生状态竞争与维护负担。
3. **恢复通知通道冗余**：
   - `/api/tasks/notify` 包含 `event_type="recovery"` 通道；
   - `/api/tasks/{task_id}/ack` 同样支持携带 `solution` 触发飞书恢复卡片。
4. **服务端存储双轨冗余**：
   - `TaskMailboxManager` 引入 SQLiteStorage 后，仍然保留了整套 `_fallback_memory_events` 内存列表，所有方法均充斥着 `if self._storage: ... else: ...`。
5. **CLI 工具与示例代码臃肿**：
   - `skills/trainpilot/scripts/trainpilot_tool.py` 存在大量重复的序列化与环境检测代码；
   - `examples/real_gpu_training.py` 近 400 行，包含过多非核心的模拟线程与轮询。

---

## 2. 总体架构与数据流设计

经过精简后，系统确立“**服务端单一决策源 + 客户端标准通用 Guardian**”的高内聚架构：

```text
[ GPU 训练节点 (无公网IP) ]                     [ 公网控制面网关 (FastAPI + SQLite) ]             [ 飞书客户端 / 人类工程师 ]
         │                                                      │                                              │
1. 异常拦截 (NaN/Inf)                                           │                                              │
   guardian.check_and_handle_loss()                             │                                              │
         │─── POST /api/tasks/notify (alert) ──────────────────►│                                              │
         │    (现场挂起，进入阻塞长轮询)                          │─── 推送告警交互卡片 ────────────────────────►│
         │                                                      │    (启动服务端 30s 单点定时器)                  │
         │                                                      │                                              │
         │                                                      │◄── 人工点击决策 (Webhook) ───────────────────│
         │                                                      │    [或 30s 超时由服务端自动决策 self_resolve]   │
         │                                                      │    (就地更新飞书卡片为已处理)                   │
         │                                                      │                                              │
2. 毫秒级唤醒                                                   │                                              │
   GET /instruction (pop=True) ◄────────────────────────────────┤                                              │
         │                                                      │                                              │
3. 执行恢复动作 (self_resolve / stop_training)                   │                                              │
         │─── POST /api/tasks/{task_id}/ack ───────────────────►│                                              │
              (携带 solution 自愈总结)                           │─── 异步推送【异常已自愈】绿色卡片 ─────────────►│
              (任务状态原子流转回 RUNNING)                        │    (记录自愈事件至 SQLite)                    │
```

---

## 3. 详细设计说明

### 3.1 训练端：聚焦通用 TrainingGuardian

1. **废除 `src/trainpilot/agent/hooks/` 目录**：
   - 删除 `pytorch.py`，不再提供针对特定框架的 Hook 外层包装。
   - 训练脚本统一使用 `TrainingGuardian`，仅需在训练主循环中调用一行 `guardian.check_and_handle_loss(loss.item(), step=step)`。
   - 彻底移除 `build_default_agent_note`：里程碑卡片的 `🤖 Agent 智能点评` 严格由外部智能体（`agy` / `opencode` / `claude-code`）或训练人员自主撰写传入。
2. **重构 `TrainingGuardian`**：
   - 移除 `timeout_fallback_action` 参数与本地超时 fallback 执行逻辑。
   - 客户端在 `handle_anomaly` 中调用 `client.poll_instruction(timeout=None)` 或配置合理的长轮询等待，服务端成为唯一的决策来源。
   - 规范自愈总结生成逻辑：`_generate_solution_summary` 在执行完成后提取自愈说明，并随 `client.ack_instruction` 上报服务端。
3. **精简 `TrainPilotClient`**：
   - 移除无用的 `notify_recovery` 接口方法；
   - 保持只依赖标准库与 `requests` 的零第三方框架依赖特性。

### 3.2 控制面：单点超时与纯净存储

1. **纯净存储（剔除内存 events 双轨）**：
   - `TaskMailboxManager` 移除 `self._fallback_memory_events` 字典。
   - 所有事件统一持久化至 `SQLiteStorage`，内存仅保留 `_tasks` 字典用于活跃状态机路由与长轮询 `_instruction_waiters` 事件通知。
   - 清理所有读写事件时的 `if self._storage: ... else: ...` 分支判断。
2. **服务端单点超时自动闭环**：
   - 任务上报 `ALERT` 后，由 `routes/tasks.py` 的 `_schedule_auto_self_resolve` 统一调度 30s 定时器。
   - 若 30s 内未收到外部决策，执行 `default_mailbox.try_auto_resolve`，就地将决策写入信箱，并通过 `default_feishu_client.update_card_to_resolved` 将卡片就地置为“已自动处理”，同时即刻唤醒等待中的 GPU 客户端长轮询。
3. **恢复通知由 ACK 统一驱动**：
   - 客户端自愈完成后调用 `/api/tasks/{task_id}/ack`，服务端在确认 ACK 后异步发送恢复卡片；
   - 从 `EventType` 枚举和路由中清理独立的 `recovery` 事件分支。

### 3.3 示例与外围工具瘦身

1. **重构 `examples/real_gpu_training.py`**：
   - 移除 `decision_watcher` 辅助线程、移除状态轮询及本地兜底注入代码；
   - 采用真实的 PyTorch 简单模型，规范演示 `TrainingGuardian` 优雅的一行拦截；
   - 代码精简至约 100 行。
2. **精简 `examples/mock_training.py`**：
   - 作为无 GPU 环境下的标准自动化演示脚本，结构清晰直观。
3. **精简 `skills/trainpilot/scripts/trainpilot_tool.py`**：
   - 移除脚本内重复定义的 `_sanitize_for_json`、重复的 `.env` 递归解析等冗余代码，直接从 `trainpilot` 导入。
4. **文档同步**：
   - 更新 `README.md` 与 `skills/trainpilot/SKILL.md`，统一移除对 `TrainPilotPyTorchHook` 的引用，强化 `TrainingGuardian` 标准范式。

---

## 4. 验证与测试策略

1. **单元测试与集成测试更新**：
   - `tests/test_agent_client.py`：移除针对 `TrainPilotPyTorchHook` 的测试，增加对 `TrainingGuardian` 异常上报、长轮询等待与 ACK 附带 solution 触发恢复卡片的测试。
   - `tests/test_server_api.py`：验证 `notify`、`instruction`（长轮询）、`ack` 状态流转，确认已移除独立的 `notify(recovery)` 路径。
   - `tests/test_sqlite_storage.py`：验证 SQLite 单一持久化与事件修剪功能完好。
   - `tests/test_e2e_simulation.py`：运行全流程闭环仿真，确保异常捕获 -> 飞书卡片更新 -> 长轮询唤醒 -> ACK 恢复全链路 100% 成功。
2. **全量回归**：
   - 执行 `uv run pytest`，确保测试全部通过。
