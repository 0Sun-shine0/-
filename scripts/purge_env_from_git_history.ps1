#!/usr/bin/env pwsh
<#
.SYNOPSIS
    从 git 全部历史中彻底删除 .env（含已泄露的 API 密钥）。

.DESCRIPTION
    `git rm --cached .env` 只能阻止后续提交，**历史里的旧提交仍然带着密钥**。
    本脚本重写所有提交，把 .env 从每一个提交中剔除。

    ⚠️ 重写历史会改变所有提交的 SHA，必须 force push，且会影响其他协作者。
    因此本脚本默认只做演练（-DryRun 语义），必须显式传 -Execute 才真正执行。

    ⚠️ 更重要的一点：**重写历史无法撤回已经发生的泄露**。
    GitHub 的 fork、缓存、以及任何已经克隆过仓库的人手里都还有旧密钥。
    唯一可靠的处理是到服务商控制台**吊销并轮换密钥**，这一步必须人工完成。

.PARAMETER Execute
    真正执行历史重写。不加此参数时只打印将要执行的命令。

.PARAMETER Remote
    force push 的目标远端，默认 origin。传空字符串则跳过推送。

.EXAMPLE
    # 1) 先看会发生什么
    pwsh scripts/purge_env_from_git_history.ps1

    # 2) 确认无误后执行（会先自动打备份标签）
    pwsh scripts/purge_env_from_git_history.ps1 -Execute

.NOTES
    执行前请务必：① 已吊销旧密钥；② 已通知所有协作者；③ 工作区改动已提交或暂存。
#>
[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$Remote = "origin",
    [string]$PathToRemove = ".env",
    # 只重写历史、不推送。用于"先清除历史、验证无误后再手动推送"的分步流程。
    # 注意不能靠 -Remote "" 来跳过：PowerShell 的 -File 参数绑定会把空字符串
    # 当成"缺少参数"而直接报错。
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

function Write-Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Write-Warn($text) { Write-Host $text -ForegroundColor Yellow }

# 必须从仓库根目录运行
$repoRoot = (git rev-parse --show-toplevel 2>$null)
if (-not $repoRoot) { throw "当前目录不在 git 仓库中" }
Set-Location $repoRoot

Write-Step "仓库状态检查"
Write-Host "仓库根目录: $repoRoot"
Write-Host "当前分支  : $(git rev-parse --abbrev-ref HEAD)"

$dirty = git status --porcelain
if ($dirty) {
    Write-Warn "工作区有未提交改动："
    $dirty | ForEach-Object { Write-Host "  $_" }
    Write-Warn "建议先提交或 stash，否则重写期间可能丢失改动。"
}

Write-Step "受影响的提交"
$affected = git log --all --oneline --name-only -- $PathToRemove
if (-not $affected) {
    Write-Host "历史中没有 $PathToRemove —— 无需重写，收工。"
    exit 0
}
$affected | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" }

$backupTag = "backup-before-purge-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

if (-not $Execute) {
    Write-Step "演练模式（未做任何修改）"
    Write-Host "将执行的操作："
    Write-Host "  1. 为每个本地分支创建备份标签 $backupTag-<分支名>（可随时回退）"
    Write-Host "  2. git filter-branch --index-filter 'git rm --cached --ignore-unmatch $PathToRemove' --prune-empty -- --all"
    Write-Host "  3. 清理 refs/original、reflog、gc"
    if ($NoPush) {
        Write-Host "  4. （-NoPush）只重写本地历史，不推送"
    } elseif ($Remote) {
        Write-Host "  4. git push $Remote --force --all 与 --tags"
    }
    Write-Host "`n确认无误后重跑并加上 -Execute。"
    Write-Warn "`n提醒：请先到 https://platform.deepseek.com 吊销旧密钥——重写历史撤不回已经泄露的密钥。"
    exit 0
}

# ---------------------------------------------------------------------------
Write-Step "① 创建备份标签"
# 为每个本地分支各打一个标签。
# 注意不能用 `git tag <name> --all`：git tag 不接受 --all 来"给所有引用打标签"，
# 那会直接报 `error: unknown option 'all'`（早先版本就因此中止）。
# 由于后续会 expire reflog + gc，只有标签能可靠保住旧提交。
$backupBranches = git for-each-ref --format='%(refname:short)' refs/heads
if (-not $backupBranches) { throw "没有找到任何本地分支，中止" }
foreach ($b in $backupBranches) {
    $tagName = "$backupTag-$b"
    git tag -f $tagName $b 2>$null
    if ($LASTEXITCODE -ne 0) { throw "创建备份标签失败: $tagName（中止，未做任何修改）" }
    Write-Host "  已创建: $tagName -> $b"
}
Write-Host "回退方式: git reset --hard <上面对应的标签名>"

Write-Step "② 重写历史"
$filter = if (Get-Command git-filter-repo -ErrorAction SilentlyContinue) {
    Write-Host "检测到 git-filter-repo，使用它（更快更安全）"
    "git filter-repo --path $PathToRemove --invert-paths --force"
} else {
    Write-Host "未安装 git-filter-repo，使用内置的 filter-branch"
    Write-Warn "（如需更快的方案: pip install git-filter-repo）"
    "git filter-branch --index-filter `"git rm --cached --ignore-unmatch $PathToRemove`" --prune-empty --tag-name-filter cat -- --all"
}
Write-Host "执行: $filter"
Invoke-Expression $filter
if ($LASTEXITCODE -ne 0) { throw "历史重写失败，可用备份标签回退: git reset --hard $backupTag" }

Write-Step "③ 清理残留引用"
# filter-branch 会把原始引用留在 refs/original，必须删掉否则旧对象仍会被 GC 保留
if (Test-Path .git/refs/original) {
    git for-each-ref --format="%(refname)" refs/original | ForEach-Object { git update-ref -d $_ }
    Write-Host "已删除 refs/original"
}
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Step "④ 验证历史中已无 $PathToRemove"
$stillThere = git log --all --oneline --name-only -- $PathToRemove
if ($stillThere) { Write-Warn "仍能找到 $PathToRemove，请检查：" ; $stillThere | Select-Object -First 5 }
else { Write-Host "干净：历史中已不存在 $PathToRemove" }

$keyLeft = git rev-list --objects --all |
    Where-Object { $_ -match '\.(py|json|ya?ml|md|txt|env|cfg|ini)$' } |
    ForEach-Object { git cat-file -p ($_ -split ' ')[0] 2>$null } |
    Select-String -Pattern 'sk-[A-Za-z0-9]{20,}' -ErrorAction SilentlyContinue
if ($keyLeft) { Write-Warn "历史中仍能匹配到密钥字符串，请人工复查" }
else { Write-Host "干净：历史中已匹配不到 sk- 密钥" }

if ($NoPush) {
    Write-Step "⑤ 跳过推送（-NoPush）"
    Write-Host "历史已在本地重写完成。确认无误后可自行推送，例如："
    Write-Host "  git push $Remote --force --all"
    Write-Host "  git push $Remote --force --tags"
} elseif ($Remote) {
    Write-Step "⑤ force push 到 $Remote"
    Write-Warn "这一步会覆盖远端历史，协作者必须重新 clone 或 hard reset。"
    git push $Remote --force --all
    git push $Remote --force --tags
}

Write-Host "`n完成。" -ForegroundColor Green
Write-Warn "再次提醒：历史已重写，但泄露期间密钥可能已被抓取。请确认已在服务商侧吊销旧密钥。"
