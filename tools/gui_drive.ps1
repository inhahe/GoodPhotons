# Drive an ftrace GUI window (the groom tool, a viewer) by pid: focus it, send keys, click / drag
# at window-relative fractions, capture it to a PNG, minimize it again. Runs DPI-unaware, so window
# rects are logical while the screen is captured in physical pixels (the rect is scaled by the
# window DPI). How the groom tool is exercised without a hand on the mouse (design.md, 0.330.0):
#   pwsh -NoProfile -File tools/gui_drive.ps1 -ProcId <pid> -Actions "restore;key:n;click:0.69,0.35;shot:png/x.png;min"
# actions: restore | min | key:<SendKeys text, e.g. n or ^s> | keydown:<vk> | keyup:<vk> (17 = Ctrl, 16 = Shift, 18 = Alt)
#          | click:fx,fy | drag:fx0,fy0,fx1,fy1 | move:fx,fy | wheel:<notches, negative = down> | shot:<png path> | wait:<ms>
param([int]$ProcId, [string]$Actions = "restore")
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System; using System.Runtime.InteropServices;
public class GD {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern int GetSystemMetrics(int i);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint data, UIntPtr extra);
  [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
}
'@
$p = Get-Process -Id $ProcId
$h = $p.MainWindowHandle
for ($i = 0; $i -lt 40 -and $h -eq 0; $i++) { Start-Sleep -Milliseconds 500; $p = Get-Process -Id $ProcId; $h = $p.MainWindowHandle }
if ($h -eq 0) { "no window for pid $ProcId"; exit 1 }
$sw = [GD]::GetSystemMetrics(0); $sh = [GD]::GetSystemMetrics(1)      # logical screen
function Rect { $r = New-Object GD+RECT; [GD]::GetWindowRect($h, [ref]$r) | Out-Null; return $r }
function MoveTo($fx, $fy) {
    $r = Rect
    $x = $r.L + $fx * ($r.R - $r.L); $y = $r.T + $fy * ($r.B - $r.T)
    $nx = [uint32]($x / $sw * 65535); $ny = [uint32]($y / $sh * 65535)
    [GD]::mouse_event(0x8001, $nx, $ny, 0, [UIntPtr]::Zero)      # MOVE | ABSOLUTE
    Start-Sleep -Milliseconds 120
}
foreach ($a in $Actions.Split(";")) {
    $a = $a.Trim(); if ($a -eq "") { continue }
    $kv = $a.Split(":", 2); $op = $kv[0]; $arg = if ($kv.Length -gt 1) { $kv[1] } else { "" }
    switch ($op) {
        "restore" { [GD]::ShowWindow($h, 9) | Out-Null; [GD]::SetForegroundWindow($h) | Out-Null; Start-Sleep -Milliseconds 1200; "restored" }
        "min"     { [GD]::ShowWindow($h, 6) | Out-Null; "minimized" }
        "wait"    { Start-Sleep -Milliseconds ([int]$arg) }
        "keydown" { [GD]::keybd_event([byte][int]$arg, 0, 0, [UIntPtr]::Zero); Start-Sleep -Milliseconds 100; "keydown $arg" }
        "keyup"   { [GD]::keybd_event([byte][int]$arg, 0, 2, [UIntPtr]::Zero); Start-Sleep -Milliseconds 100; "keyup $arg" }
        "key"     { [System.Windows.Forms.SendKeys]::SendWait($arg); Start-Sleep -Milliseconds 400; "key $arg" }
        "move"    { $f = $arg.Split(","); MoveTo ([double]$f[0]) ([double]$f[1]); "moved $arg" }
        "wheel"   { $n = [int]$arg; $step = if ($n -lt 0) { [uint32]4294967176 } else { [uint32]120 }   # -120 as unsigned, or +120
                    for ($i = 0; $i -lt [Math]::Abs($n); $i++) { [GD]::mouse_event(0x0800, 0, 0, $step, [UIntPtr]::Zero); Start-Sleep -Milliseconds 60 }
                    Start-Sleep -Milliseconds 300; "wheel $n" }
        "click"   { $f = $arg.Split(","); MoveTo ([double]$f[0]) ([double]$f[1]); Start-Sleep -Milliseconds 150
                    [GD]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero); Start-Sleep -Milliseconds 80
                    [GD]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero); Start-Sleep -Milliseconds 400; "clicked $arg" }
        "drag"    { $f = $arg.Split(","); MoveTo ([double]$f[0]) ([double]$f[1]); Start-Sleep -Milliseconds 200
                    [GD]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero); Start-Sleep -Milliseconds 150
                    $n = 12
                    for ($i = 1; $i -le $n; $i++) { $t = $i / $n; MoveTo ([double]$f[0] + ($([double]$f[2]) - [double]$f[0]) * $t) ([double]$f[1] + ($([double]$f[3]) - [double]$f[1]) * $t) }
                    Start-Sleep -Milliseconds 150; [GD]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero); Start-Sleep -Milliseconds 400; "dragged $arg" }
        "shot"    { Start-Sleep -Milliseconds 300
                    $r = Rect; $s = [GD]::GetDpiForWindow($h) / 96.0
                    $x = [int]($r.L * $s); $y = [int]($r.T * $s); $w = [int](($r.R - $r.L) * $s); $ht = [int](($r.B - $r.T) * $s)
                    $b = New-Object System.Drawing.Bitmap $w, $ht
                    $g = [System.Drawing.Graphics]::FromImage($b); $g.CopyFromScreen($x, $y, 0, 0, $b.Size); $b.Save($arg); "shot $w x $ht -> $arg" }
        default   { "unknown action $a" }
    }
}
