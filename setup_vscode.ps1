# 一键写入 ScholarPulse 的 VSCode 运行配置（launch.json + settings.json）
# 用法：在项目根目录的终端执行  powershell -ExecutionPolicy Bypass -File .\setup_vscode.ps1
$ErrorActionPreference = "Stop"

$launch = @'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "ScholarPulse Web 服务 (端口8321)",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["webapp.app:app", "--port", "8321"],
      "cwd": "${workspaceFolder}",
      "python": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "CLI: 教师班级看板",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}\\run.py",
      "args": ["teacher", "--teacher", "T001"],
      "python": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "console": "integratedTerminal"
    },
    {
      "name": "CLI: 高级分析(混淆矩阵/根因/置信区间)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}\\run.py",
      "args": ["advanced", "--sid", "s001"],
      "python": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "console": "integratedTerminal"
    },
    {
      "name": "CLI: LangGraph 状态图演示",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}\\run.py",
      "args": ["graph-demo"],
      "python": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "console": "integratedTerminal"
    },
    {
      "name": "Python: 当前文件",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "python": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "console": "integratedTerminal"
    },
    {
      "name": "pytest: 全量测试",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["-q"],
      "cwd": "${workspaceFolder}",
      "python": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "console": "integratedTerminal"
    }
  ]
}
'@

$settings = @'
{
  "python-envs.defaultEnvManager": "ms-python.python:system",
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
}
'@

Set-Content -Path "$PSScriptRoot\.vscode\launch.json" -Value $launch -Encoding UTF8
Set-Content -Path "$PSScriptRoot\.vscode\settings.json" -Value $settings -Encoding UTF8
Write-Host "OK: .vscode\launch.json 与 settings.json 已写入" -ForegroundColor Green
Write-Host "用法：VSCode 按 Ctrl+Shift+D 打开运行面板 -> 顶部选『ScholarPulse Web 服务 (端口8321)』-> F5" -ForegroundColor Cyan
