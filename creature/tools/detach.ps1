# detach.ps1 -- launch a long-running command DETACHED from the calling shell.
#
# Why: a training run started as a background-shell child dies when that shell's
# session restarts (this killed a P1b run at step 335,872 four minutes in,
# 2026-08-17). WMI's Win32_Process.Create parents the new process under the
# wmiprvse.exe service instead, outside the caller's process tree and any job
# object, so the run's lifetime is its own.
#
# Usage:
#   powershell -NoProfile -File tools\detach.ps1 -Log <file> -PidFile <file> `
#       [-WorkDir <dir>] <exe> <args...>
#
#   powershell -NoProfile -File tools\detach.ps1 `
#       -Log runs\myrun\train.log -PidFile runs\myrun\pid `
#       .venv\Scripts\python.exe tools\train.py --steps 2e7 --out runs/myrun
#
# stdout+stderr go to -Log (via a cmd.exe redirect wrapper). -PidFile receives
# the DEEPEST python/exe PID in the spawned chain -- the venv python.exe is a
# launcher shim that re-execs the base interpreter as a child, so the first
# child is NOT the real trainer. That deepest PID is the one to monitor and,
# if ever necessary, stop (kill only that exact PID; never kill by name).
param(
    [Parameter(Mandatory)] [string]$Log,
    [Parameter(Mandatory)] [string]$PidFile,
    [string]$WorkDir = (Get-Location).Path,
    [Parameter(Mandatory, ValueFromRemainingArguments)] [string[]]$Command
)
$rp = Resolve-Path -ErrorAction SilentlyContinue $WorkDir
if ($rp) { $WorkDir = $rp.Path }
$exe = $Command[0]
if (-not [System.IO.Path]::IsPathRooted($exe)) { $exe = Join-Path $WorkDir $exe }
$argstr = (($Command | Select-Object -Skip 1) | ForEach-Object {
    if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
$logAbs = if ([System.IO.Path]::IsPathRooted($Log)) { $Log } else { Join-Path $WorkDir $Log }

# >> not >: a resume relaunched into the same run dir appends to the same log,
# matching log.jsonl's "the file spans processes" convention.
$inner = '"{0}" {1} >> "{2}" 2>&1' -f $exe, $argstr, $logAbs
$cmdline = 'cmd.exe /c "' + $inner + '"'

$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmdline; CurrentDirectory = $WorkDir }
if ($r.ReturnValue -ne 0) { Write-Error "Win32_Process.Create failed: rv=$($r.ReturnValue)"; exit 1 }

# Walk to the deepest descendant with the target exe's name (venv shim chain).
Start-Sleep -Seconds 5
$name = [System.IO.Path]::GetFileName($exe)
$p = $r.ProcessId
while ($true) {
    $child = Get-CimInstance Win32_Process -Filter "Name='$name' AND ParentProcessId=$p"
    if (-not $child) { break }
    $p = @($child)[0].ProcessId
}
$pidAbs = if ([System.IO.Path]::IsPathRooted($PidFile)) { $PidFile } else { Join-Path $WorkDir $PidFile }
Set-Content -Path $pidAbs -Value $p
if ($p -ne $r.ProcessId) {
    "launched detached: pid $p (cmd wrapper $($r.ProcessId)) -> $pidAbs"
} else {
    "WARNING: no $name child found yet; pid file holds cmd wrapper $($r.ProcessId)"
    exit 2
}
