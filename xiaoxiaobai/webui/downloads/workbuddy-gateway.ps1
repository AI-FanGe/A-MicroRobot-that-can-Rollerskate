# 小小白 → WorkBuddy (Windows)
# Installs WorkBuddy's agent (the CodeBuddy Code CLI) as an HTTP gateway that starts at login and
# reports its address to the robot, so the robot can hand it tasks over the LAN. Uses WorkBuddy's own
# login, skills and MCP servers. Copy the command from the robot's web UI (设置 → 电脑助手):
#
#   $env:XXB_ROBOT='http://<robot>'; $env:XXB_PASSWORD='<pw>'; irm http://<robot>/downloads/workbuddy-gateway.ps1 | iex
#
# Options (environment variables, set before running):
#   $env:XXB_PORT = 8321        gateway port
#   $env:XXB_AUTO = 1           run tasks without asking for approval (only on a PC you trust the robot with)
#   $env:XXB_FOREGROUND = 1     run in this window instead of installing
#   $env:XXB_UNINSTALL = 1      remove it
# Without XXB_ROBOT the gateway just runs in this window and prints its address and password.

& {
$ErrorActionPreference = "Stop"
$Port = if ($env:XXB_PORT) { [int]$env:XXB_PORT } else { 8321 }
$Auto = $env:XXB_AUTO -eq "1"
$Robot = "$env:XXB_ROBOT".TrimEnd("/")
$Password = "$env:XXB_PASSWORD"
$AppDir = Join-Path $env:LOCALAPPDATA "xiaoxiaobai-workbuddy"
$Startup = [Environment]::GetFolderPath("Startup")
$Shortcut = Join-Path $Startup "小小白 WorkBuddy.lnk"
$RuleName = "小小白 WorkBuddy $Port"

Write-Host ""
Write-Host "== 小小白 → WorkBuddy 网关 ==" -ForegroundColor Cyan

function Stop-Gateway {
    # The background runner, then any gateway on this port (also one started by hand in a window).
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -like "*xiaoxiaobai-workbuddy*run.ps1*" -or
            ($_.CommandLine -like "*--serve*" -and $_.CommandLine -like "*--port $Port*")) -and $_.ProcessId -ne $PID } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

if ($env:XXB_UNINSTALL -eq "1") {
    Stop-Gateway
    Remove-Item $Shortcut, $AppDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "已卸载：电脑不再开机自动连接小小白。" -ForegroundColor Green
    return
}

function Find-WorkBuddy {
    $roots = @(
        "$env:LOCALAPPDATA\Programs\WorkBuddy",
        "$env:LOCALAPPDATA\Programs\WorkBuddyAI",
        "$env:LOCALAPPDATA\Programs\WorkBuddy AI",
        "$env:ProgramFiles\WorkBuddy",
        "${env:ProgramFiles(x86)}\WorkBuddy"
    )
    $roots += Get-ChildItem "$env:LOCALAPPDATA\Programs", "$env:ProgramFiles" -Directory -Filter "*Buddy*" -ErrorAction SilentlyContinue | ForEach-Object FullName
    foreach ($root in ($roots | Select-Object -Unique)) {
        $js = Join-Path $root "resources\app.asar.unpacked\cli\dist\codebuddy.js"
        if (Test-Path $js) {
            $exe = Get-ChildItem $root -Filter *.exe -File | Where-Object { $_.Name -notmatch "unins|update|elevate|crash" } | Select-Object -First 1
            return @{ Root = $root; Js = $js; Exe = $exe.FullName }
        }
    }
    return $null
}

# WorkBuddy's agent is the CodeBuddy Code CLI: WorkBuddy's own copy when it ships one, else the npm package.
$runner = $null
$electron = $false
$wb = Find-WorkBuddy
$node = Get-Command node.exe -ErrorAction SilentlyContinue
if ($wb) {
    Write-Host "找到 WorkBuddy：$($wb.Root)"
    if ($node) {
        $runner = @($node.Source, $wb.Js)
    } elseif ($wb.Exe) {
        # WorkBuddy is an Electron app: its own exe runs plain Node scripts with this variable set.
        $electron = $true
        $runner = @($wb.Exe, $wb.Js)
    }
}
if (-not $runner) {
    $cli = Get-Command codebuddy.cmd -ErrorAction SilentlyContinue
    if (-not $cli -and (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
        Write-Host "安装 CodeBuddy Code 命令行（WorkBuddy 背后的智能体）……"
        & npm.cmd install -g @tencent-ai/codebuddy-code
        $cli = Get-Command codebuddy.cmd -ErrorAction SilentlyContinue
    }
    if ($cli) {
        Write-Host "使用 codebuddy 命令：$($cli.Source)"
        $runner = @($cli.Source)
    }
}
if (-not $runner) {
    Write-Host "没有可用的 codebuddy。请先安装并登录 WorkBuddy（https://www.workbuddy.cn），或安装 Node.js 后运行：npm install -g @tencent-ai/codebuddy-code" -ForegroundColor Red
    return
}

$configDir = ""
foreach ($dir in @("$HOME\.workbuddy", "$HOME\.workbuddy-ai")) {
    if (Test-Path $dir) {
        $configDir = $dir
        Write-Host "使用 WorkBuddy 的登录和配置：$dir"
        break
    }
}

# The robot connects to this port: let it through Windows Firewall on private networks.
if (-not (Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue)) {
    $rule = "New-NetFirewallRule -DisplayName '$RuleName' -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Private,Domain | Out-Null"
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    try {
        if ($isAdmin) {
            Invoke-Expression $rule
        } else {
            Write-Host "请求管理员权限来放行防火墙端口 $Port（弹窗里点“是”）……"
            Start-Process powershell.exe -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList "-NoProfile", "-Command", $rule
        }
        if (Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue) { Write-Host "已放行防火墙端口 $Port（专用网络）" }
    } catch {
        Write-Host "没能自动放行防火墙。如果弹出 Windows 防火墙提示，请勾选“专用网络”并允许访问。" -ForegroundColor Yellow
    }
}

if (-not $Robot) {
    $ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notmatch "^(127|169\.254)\." -and $_.InterfaceAlias -notmatch "vEthernet|Loopback|VMware|VirtualBox" } |
        ForEach-Object IPAddress
    Write-Host ""
    Write-Host "没有带机器人地址，只在这个窗口里运行。在小小白网页 设置 → 电脑助手 里填写：" -ForegroundColor Green
    foreach ($ip in $ips) { Write-Host "  地址：$ip`:$Port" -ForegroundColor Green }
    Write-Host "  密码：下面启动信息里 Password 后面那一串" -ForegroundColor Green
    Write-Host "（更省事：从网页上复制带地址的安装命令，装一次以后开机自动连接。）"
    Write-Host ""
    if ($Password) { $env:CODEBUDDY_GATEWAY_PASSWORD = $Password }
    if ($configDir) { $env:CODEBUDDY_CONFIG_DIR = $configDir }
    if ($electron) { $env:ELECTRON_RUN_AS_NODE = "1" }
    $gatewayArgs = @("--serve", "--host", "0.0.0.0", "--port", "$Port")
    if ($Auto) { $gatewayArgs += @("--permission-mode", "bypassPermissions") }
    $exe = $runner[0]
    $rest = @($runner | Select-Object -Skip 1) + $gatewayArgs
    & $exe @rest
    return
}
if (-not $Password) {
    Write-Host "缺少 XXB_PASSWORD：请从小小白网页 设置 → 电脑助手 复制完整的安装命令。" -ForegroundColor Red
    return
}

Stop-Gateway
New-Item -ItemType Directory -Force $AppDir | Out-Null
$config = [ordered]@{
    runner = $runner; electron = $electron; port = $Port; auto = $Auto; robot = $Robot
    password = $Password; configDir = $configDir; name = $env:COMPUTERNAME
}
[IO.File]::WriteAllText((Join-Path $AppDir "config.json"), ($config | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))

$run = @'
# Started at login: runs the WorkBuddy gateway and keeps telling the robot where this PC is.
$ErrorActionPreference = "Continue"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$cfg = [IO.File]::ReadAllText((Join-Path $dir "config.json")) | ConvertFrom-Json
$env:CODEBUDDY_GATEWAY_PASSWORD = $cfg.password
if ($cfg.configDir) { $env:CODEBUDDY_CONFIG_DIR = $cfg.configDir }
if ($cfg.electron) { $env:ELECTRON_RUN_AS_NODE = "1" }
$runner = @($cfg.runner)
$gatewayArgs = @($runner | Select-Object -Skip 1) + @("--serve", "--host", "0.0.0.0", "--port", "$($cfg.port)")
if ($cfg.auto) { $gatewayArgs += @("--permission-mode", "bypassPermissions") }
$line = ($gatewayArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join " "
$log = Join-Path $dir "gateway.log"
$body = [Text.Encoding]::UTF8.GetBytes((@{ port = $cfg.port; password = $cfg.password; name = $cfg.name } | ConvertTo-Json -Compress))
$headers = @{ "X-CodeBuddy-Request" = "1"; "Authorization" = "Bearer $($cfg.password)" }
while ($true) {
    $gw = Start-Process -FilePath $runner[0] -ArgumentList $line -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    for ($i = 0; $i -lt 90 -and -not $gw.HasExited; $i++) {
        try { Invoke-RestMethod "http://127.0.0.1:$($cfg.port)/api/v1/health" -Headers $headers -TimeoutSec 2 | Out-Null; break }
        catch { if ($_.Exception.Response) { break }; Start-Sleep 1 }
    }
    while (-not $gw.HasExited) {
        try {
            Invoke-RestMethod -Method Post "$($cfg.robot)/api/v1/workbuddy/register" -ContentType "application/json; charset=utf-8" `
                -Body $body -TimeoutSec 10 | Out-Null
        } catch {}
        for ($i = 0; $i -lt 30 -and -not $gw.HasExited; $i++) { Start-Sleep 1 }
    }
    Start-Sleep 15
}
'@
[IO.File]::WriteAllText((Join-Path $AppDir "run.ps1"), $run, (New-Object Text.UTF8Encoding $true))

$runArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$(Join-Path $AppDir 'run.ps1')`""
if ($env:XXB_FOREGROUND -eq "1") {
    Write-Host "在这个窗口里运行（关掉窗口就断开）。"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $AppDir "run.ps1")
    return
}

$link = (New-Object -ComObject WScript.Shell).CreateShortcut($Shortcut)
$link.TargetPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$link.Arguments = $runArgs
$link.WorkingDirectory = $AppDir
$link.WindowStyle = 7
$link.Save()
Start-Process powershell.exe -ArgumentList $runArgs -WindowStyle Hidden
Write-Host "已加入开机启动：登录后在后台自动运行（没有窗口）。"

Write-Host "等 WorkBuddy 启动，再让机器人回连测试……"
$headers = @{ "X-CodeBuddy-Request" = "1"; "Authorization" = "Bearer $Password" }
$up = $false
for ($i = 0; $i -lt 90; $i++) {
    try { Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/health" -Headers $headers -TimeoutSec 2 | Out-Null; $up = $true; break }
    catch { if ($_.Exception.Response) { $up = $true; break }; Start-Sleep 1 }
}
if (-not $up) {
    Write-Host "WorkBuddy 网关 90 秒内没有启动起来，日志在：$AppDir\gateway.log" -ForegroundColor Red
    Get-Content "$AppDir\gateway.log", "$AppDir\gateway.log.err" -Tail 20 -ErrorAction SilentlyContinue
    return
}
$body = [Text.Encoding]::UTF8.GetBytes((@{ port = $Port; password = $Password; name = $env:COMPUTERNAME } | ConvertTo-Json -Compress))
$reply = $null
$message = ""
try {
    $reply = Invoke-RestMethod -Method Post "$Robot/api/v1/workbuddy/register" -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec 15
    $message = $reply.data.message
} catch {
    $message = $_.ErrorDetails.Message
    try { $message = ($message | ConvertFrom-Json).message } catch {}
    if (-not $message) { $message = $_.Exception.Message }
}
Write-Host ""
if ($reply -and $reply.data.authorized) {
    Write-Host "连好了：机器人已经能用这台电脑。以后开机会自动连接，不用再运行这条命令。" -ForegroundColor Green
    if (-not $Auto) { Write-Host "需要确认的操作会在 http://127.0.0.1:$Port 网页里等你批准（想全自动：先执行 `$env:XXB_AUTO=1 再重新运行）。" }
} else {
    Write-Host "网关已经在这台电脑上运行，但机器人还没连上：$message" -ForegroundColor Yellow
    Write-Host "修好后不用重装，电脑每 30 秒会自动再报到一次。"
}
Write-Host "卸载：`$env:XXB_UNINSTALL=1; irm $Robot/downloads/workbuddy-gateway.ps1 | iex"
}
