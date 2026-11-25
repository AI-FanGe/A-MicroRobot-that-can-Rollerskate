#!/usr/bin/env bash
# 小小白 → WorkBuddy (macOS / Linux)
# Installs WorkBuddy's agent (the CodeBuddy Code CLI) as an HTTP gateway that starts at login and
# reports its address to the robot, so the robot can hand it tasks over the LAN. Uses WorkBuddy's own
# login, skills and MCP servers. Copy the command from the robot's web UI (设置 → 电脑助手):
#
#   curl -fsSL http://<robot>/downloads/workbuddy-gateway.sh | XXB_ROBOT=http://<robot> XXB_PASSWORD=<pw> bash
#
# Options: XXB_PORT=8321 (gateway port), XXB_AUTO=1 (no approval prompts; only on a computer you trust
# the robot with), XXB_FOREGROUND=1 (run in this terminal instead of installing), XXB_UNINSTALL=1.
# Without XXB_ROBOT the gateway just runs in this terminal and prints its address and password.
set -euo pipefail

# Piped into bash: all of it is read before anything runs, so no command eats the rest of the script.
main() {
PORT="${XXB_PORT:-8321}"
AUTO="${XXB_AUTO:-0}"
ROBOT="${XXB_ROBOT:-}"
ROBOT="${ROBOT%/}"
PASSWORD="${XXB_PASSWORD:-}"
LABEL="com.xiaoxiaobai.workbuddy"
if [[ "$(uname)" == "Darwin" ]]; then
  APP_DIR="$HOME/Library/Application Support/xiaoxiaobai-workbuddy"
  PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
else
  APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/xiaoxiaobai-workbuddy"
  UNIT="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/xiaoxiaobai-workbuddy.service"
fi

echo
echo "== 小小白 → WorkBuddy 网关 =="

stop_service() {
  if [[ "$(uname)" == "Darwin" ]]; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || launchctl unload "$PLIST" 2>/dev/null || true
  elif command -v systemctl >/dev/null 2>&1; then
    systemctl --user disable --now xiaoxiaobai-workbuddy.service 2>/dev/null || true
  fi
  # An older gateway started by hand in a terminal holds the port too.
  pkill -f -- "--serve --host 0.0.0.0 --port $PORT" 2>/dev/null || true
}

if [[ "${XXB_UNINSTALL:-0}" == "1" ]]; then
  stop_service
  rm -f "${PLIST:-}" "${UNIT:-}"
  rm -rf "$APP_DIR"
  echo "已卸载：电脑不再开机自动连接小小白。"
  exit 0
fi

# WorkBuddy's agent is the CodeBuddy Code CLI. Recent WorkBuddy.app builds no longer ship a
# runnable copy (cli/bin/codebuddy is there, the cli/dist it loads is not), so the standalone
# npm package comes first; old builds with an unpacked cli/dist/codebuddy.js still work.
RUNNER=()
ELECTRON=0
if ! command -v codebuddy >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
  echo "安装 CodeBuddy Code 命令行（WorkBuddy 背后的智能体）……"
  npm install -g @tencent-ai/codebuddy-code
fi
if command -v codebuddy >/dev/null 2>&1; then
  echo "使用 codebuddy 命令：$(command -v codebuddy)"
  RUNNER=("$(command -v codebuddy)")
else
  for app in "/Applications/WorkBuddy.app" "/Applications/WorkBuddy AI.app" "$HOME/Applications/WorkBuddy.app"; do
    js="$app/Contents/Resources/app.asar.unpacked/cli/dist/codebuddy.js"
    if [[ -f "$js" ]]; then
      echo "找到 WorkBuddy：$app"
      exe="$(find "$app/Contents/MacOS" -maxdepth 1 -type f -perm -u+x | head -n 1)"
      ELECTRON=1
      RUNNER=("$exe" "$js")
      break
    fi
  done
fi
if [[ ${#RUNNER[@]} -eq 0 ]]; then
  echo "没有可用的 codebuddy。先安装 Node.js（Mac：brew install node），再运行：npm install -g @tencent-ai/codebuddy-code" >&2
  exit 1
fi

CONFIG_DIR=""
for dir in "$HOME/.workbuddy" "$HOME/.workbuddy-ai"; do
  if [[ -d "$dir" ]]; then
    CONFIG_DIR="$dir"
    echo "使用 WorkBuddy 的登录和配置：$dir"
    break
  fi
done

# Services start with a bare PATH; the npm codebuddy is a `#!/usr/bin/env node` script.
EXTRA_PATH=""
if command -v node >/dev/null 2>&1; then
  EXTRA_PATH="$(dirname "$(command -v node)")"
fi

if [[ -z "$ROBOT" ]]; then
  if command -v ipconfig >/dev/null 2>&1; then
    ips="$(for i in en0 en1 en2; do ipconfig getifaddr "$i" 2>/dev/null || true; done)"
  else
    ips="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' || true)"
  fi
  echo
  echo "没有带机器人地址，只在这个终端里运行。在小小白网页 设置 → 电脑助手 里填写："
  for ip in $ips; do echo "  地址：$ip:$PORT"; done
  echo "  密码：下面启动信息里 Password 后面那一串"
  echo "（更省事：从网页上复制带地址的安装命令，装一次以后开机自动连接。）"
  echo
  [[ -n "$PASSWORD" ]] && export CODEBUDDY_GATEWAY_PASSWORD="$PASSWORD"
  [[ -n "$CONFIG_DIR" ]] && export CODEBUDDY_CONFIG_DIR="$CONFIG_DIR"
  [[ "$ELECTRON" == "1" ]] && export ELECTRON_RUN_AS_NODE=1
  args=(--serve --host 0.0.0.0 --port "$PORT")
  [[ "$AUTO" == "1" ]] && args+=(--permission-mode bypassPermissions)
  exec "${RUNNER[@]}" "${args[@]}"
fi

if [[ -z "$PASSWORD" ]]; then
  echo "缺少 XXB_PASSWORD：请从小小白网页 设置 → 电脑助手 复制完整的安装命令。" >&2
  exit 1
fi

stop_service
mkdir -p "$APP_DIR"
chmod 700 "$APP_DIR"
NAME="$(scutil --get ComputerName 2>/dev/null || hostname)"
NAME="${NAME//\\/}"
NAME="${NAME//\"/}"
(umask 077 && declare -p RUNNER ELECTRON PORT AUTO ROBOT PASSWORD CONFIG_DIR EXTRA_PATH NAME > "$APP_DIR/config.sh")

cat > "$APP_DIR/run.sh" <<'EOF'
#!/bin/bash
# Started at login: runs the WorkBuddy gateway and keeps telling the robot where this computer is.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
. "$DIR/config.sh"
export PATH="${EXTRA_PATH:+$EXTRA_PATH:}/opt/homebrew/bin:/usr/local/bin:$HOME/.npm-global/bin:/usr/bin:/bin:$PATH"
export CODEBUDDY_GATEWAY_PASSWORD="$PASSWORD"
[ -n "$CONFIG_DIR" ] && export CODEBUDDY_CONFIG_DIR="$CONFIG_DIR"
[ "$ELECTRON" = 1 ] && export ELECTRON_RUN_AS_NODE=1
args=(--serve --host 0.0.0.0 --port "$PORT")
[ "$AUTO" = 1 ] && args+=(--permission-mode bypassPermissions)
"${RUNNER[@]}" "${args[@]}" &
gw=$!
trap 'kill "$gw" 2>/dev/null; exit 0' TERM INT
body="{\"port\": $PORT, \"password\": \"$PASSWORD\", \"name\": \"$NAME\"}"
for _ in $(seq 1 90); do
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:$PORT/api/v1/health" -H "X-CodeBuddy-Request: 1" \
    -H "Authorization: Bearer $PASSWORD" 2>/dev/null)"
  [ "$code" != "000" ] && [ -n "$code" ] && break
  kill -0 "$gw" 2>/dev/null || break
  sleep 1
done
reached=""
while kill -0 "$gw" 2>/dev/null; do
  if curl -fsS -m 10 -X POST "$ROBOT/api/v1/workbuddy/register" -H "Content-Type: application/json" \
      --data "$body" >/dev/null 2>&1; then
    [ "$reached" = 1 ] || echo "$(date '+%F %T') 已向 $ROBOT 报到"
    reached=1
  else
    [ "$reached" = 0 ] || echo "$(date '+%F %T') 报到失败：连不上 $ROBOT（会继续重试）"
    reached=0
  fi
  sleep 30 &
  wait $!
done
wait "$gw"
EOF
chmod 700 "$APP_DIR/run.sh"

if [[ "${XXB_FOREGROUND:-0}" == "1" ]]; then
  echo "在这个终端里运行（关掉终端就断开）。"
  exec /bin/bash "$APP_DIR/run.sh"
fi

LOG="$APP_DIR/gateway.log"
if [[ "$(uname)" == "Darwin" ]]; then
  mkdir -p "$(dirname "$PLIST")"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$APP_DIR/run.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF
  launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl load -w "$PLIST"
  echo "已安装为登录项服务：开机登录后自动启动。"
elif command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  mkdir -p "$(dirname "$UNIT")"
  cat > "$UNIT" <<EOF
[Unit]
Description=小小白 → WorkBuddy 网关
After=network-online.target

[Service]
ExecStart=/bin/bash "$APP_DIR/run.sh"
Restart=always
RestartSec=15
StandardOutput=append:$LOG
StandardError=append:$LOG

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now xiaoxiaobai-workbuddy.service
  echo "已安装为 systemd 用户服务：登录后自动启动。"
else
  echo "这台电脑没有可用的开机自启方式，改为在这个终端里运行（关掉终端就断开）。"
  exec /bin/bash "$APP_DIR/run.sh"
fi

echo "等 WorkBuddy 启动，再让机器人回连测试……"
up=0
for _ in $(seq 1 90); do
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:$PORT/api/v1/health" -H "X-CodeBuddy-Request: 1" \
    -H "Authorization: Bearer $PASSWORD" 2>/dev/null || true)"
  if [[ -n "$code" && "$code" != "000" ]]; then
    up=1
    break
  fi
  sleep 1
done
if [[ "$up" != "1" ]]; then
  echo "WorkBuddy 网关 90 秒内没有启动起来，日志在：$LOG" >&2
  tail -n 20 "$LOG" 2>/dev/null || true
  exit 1
fi
reply="$(curl -sS -m 15 -X POST "$ROBOT/api/v1/workbuddy/register" -H "Content-Type: application/json" \
  --data "{\"port\": $PORT, \"password\": \"$PASSWORD\", \"name\": \"$NAME\"}" 2>&1 || true)"
message="$(printf '%s' "$reply" | sed -n 's/.*"message": "\([^"]*\)".*/\1/p' | head -n 1)"
echo
if printf '%s' "$reply" | grep -q '"authorized": true'; then
  echo "✅ 连好了：机器人已经能用这台电脑。以后开机会自动连接，不用再运行这条命令。"
  if [[ "$AUTO" != "1" ]]; then
    echo "需要确认的操作会在 http://127.0.0.1:$PORT 网页里等你批准（想全自动：命令前加 XXB_AUTO=1 重新运行）。"
  fi
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "如果 macOS 弹出“bash / node 想访问桌面、文稿等文件夹”，请点允许，否则 WorkBuddy 碰不到这些文件。"
  fi
else
  echo "⚠️ 网关已经在这台电脑上运行，但机器人还没连上：${message:-$reply}"
  echo "修好后不用重装，电脑每 30 秒会自动再报到一次。"
fi
echo "卸载：curl -fsSL $ROBOT/downloads/workbuddy-gateway.sh | XXB_UNINSTALL=1 bash"
}

main "$@"
