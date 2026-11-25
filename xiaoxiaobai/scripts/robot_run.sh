#!/usr/bin/env bash
# Robot side of `make run`: the API + web UI in the foreground; Ctrl-C (or a dropped SSH session)
# stops the control loop with torque off. When the boot service (`make api-enable`) is running,
# it is restarted on the synced code and its log is followed instead.
set -e
port="${1:-8000}"
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || uv sync --frozen
if systemctl is-active --quiet xiaoxiaobai-api 2>/dev/null; then
  echo "开机服务 xiaoxiaobai-api 在运行：重启它以加载新代码。Ctrl-C 只退出日志，机器人不会停。"
  sudo -n systemctl restart xiaoxiaobai-api || echo "重启失败（需要免密 sudo），沿用旧代码。"
  exec tail -n 0 -F /tmp/xiaoxiaobai_api.log
fi
PYTHONPATH=src exec .venv/bin/python -u -m api --port "$port" --display --stop-on-exit
