# Look at an ftrace GUI window (the groom tool, a viewer) WITHOUT touching the focus, the mouse or
# the keyboard -- safe to run while the user is working in another app, which tools/gui_drive.ps1
# is not (0.373.0). The window is restored at the BOTTOM of the z-order without being activated,
# drawn into a bitmap with PrintWindow (PW_RENDERFULLCONTENT, which captures its D3D content even
# when other windows cover it), then minimized again without activation. Keys are POSTED to the
# window's own message queue, so they reach nothing else.
#   pwsh -NoProfile -File tools/gui_peek.ps1 -ProcId <pid> -Actions "show;wait:1500;shot:png/a.png;post:112;wait:800;shot:png/b.png;post:112;hide"
# actions: show | hide | wait:<ms> | shot:<png> | post:<vk> (WM_KEYDOWN + WM_KEYUP: 112 = F1, 78 = N,
#          46 = Del, 27 = Esc) | client (prints where the client area starts inside the capture)
# Mouse input cannot be simulated this way: a posted move makes the ImGui backend call
# TrackMouseEvent, which -- the real cursor being elsewhere -- answers with WM_MOUSELEAVE at once,
# and that races the posted button message, so a click lands at "nowhere" more often than not.
# Start the tool with -window-min so it does not come up in front of the user either.
param([int]$ProcId, [string]$Actions = "show;wait:1500;shot:png/peek.png;hide")
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System; using System.Runtime.InteropServices;
public class GP {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint f);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint f);
  [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint code, uint type);
  [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
}
'@
$p = Get-Process -Id $ProcId
$h = $p.MainWindowHandle
if ($h -eq 0) { "no window for pid $ProcId"; exit 1 }
$HWND_BOTTOM = [IntPtr]1
$NOSIZE = 0x1; $NOMOVE = 0x2; $NOACTIVATE = 0x10
foreach ($a in $Actions.Split(";")) {
    $a = $a.Trim(); if ($a -eq "") { continue }
    $kv = $a.Split(":", 2); $op = $kv[0]; $arg = if ($kv.Length -gt 1) { $kv[1] } else { "" }
    switch ($op) {
        "show" { [GP]::SetWindowPos($h, $HWND_BOTTOM, 0, 0, 0, 0, $NOSIZE -bor $NOMOVE -bor $NOACTIVATE) | Out-Null
                 [GP]::ShowWindow($h, 4) | Out-Null                                   # SW_SHOWNOACTIVATE
                 [GP]::SetWindowPos($h, $HWND_BOTTOM, 0, 0, 0, 0, $NOSIZE -bor $NOMOVE -bor $NOACTIVATE) | Out-Null
                 "shown (bottom of the z-order, not activated)" }
        "hide" { [GP]::ShowWindow($h, 7) | Out-Null; "minimized (not activated)" }       # SW_SHOWMINNOACTIVE
        "wait" { Start-Sleep -Milliseconds ([int]$arg) }
        "post" { $vk = [int]$arg; $sc = [GP]::MapVirtualKey([uint32]$vk, 0)
                 $down = [IntPtr](1 -bor ($sc -shl 16)); $up = [IntPtr]([int64]1 -bor ([int64]$sc -shl 16) -bor (3L -shl 30))
                 [GP]::PostMessage($h, 0x0100, [IntPtr]$vk, $down) | Out-Null; Start-Sleep -Milliseconds 120
                 [GP]::PostMessage($h, 0x0101, [IntPtr]$vk, $up) | Out-Null; Start-Sleep -Milliseconds 300; "posted vk $vk" }
        "client" { $r = New-Object GP+RECT; [GP]::GetWindowRect($h, [ref]$r) | Out-Null
                   $pt = New-Object GP+POINT; [GP]::ClientToScreen($h, [ref]$pt) | Out-Null
                   $s = [GP]::GetDpiForWindow($h) / 96.0
                   "client area starts at capture pixel ($([int](($pt.X - $r.L) * $s)), $([int](($pt.Y - $r.T) * $s)))" }
        "shot" { $r = New-Object GP+RECT; [GP]::GetWindowRect($h, [ref]$r) | Out-Null
                 $s = [GP]::GetDpiForWindow($h) / 96.0
                 $w = [int](($r.R - $r.L) * $s); $ht = [int](($r.B - $r.T) * $s)
                 $b = New-Object System.Drawing.Bitmap $w, $ht
                 $g = [System.Drawing.Graphics]::FromImage($b); $hdc = $g.GetHdc()
                 $ok = [GP]::PrintWindow($h, $hdc, 2)                                  # PW_RENDERFULLCONTENT
                 $g.ReleaseHdc($hdc); $b.Save($arg); "shot $w x $ht -> $arg (PrintWindow $ok)" }
        default { "unknown action $a" }
    }
}
