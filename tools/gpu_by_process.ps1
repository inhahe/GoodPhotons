# Per-process GPU engine utilisation. nvidia-smi cannot attribute GPU use per process
# under WDDM (it reports "100% busy" and "N/A" memory), so this is the only way to tell
# "my kernel is slow" from "someone else owns the card".
$s = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples
$rows = $s | Where-Object { $_.CookedValue -gt 0.5 } | ForEach-Object {
    $pid_ = if ($_.InstanceName -match 'pid_(\d+)') { [int]$Matches[1] } else { 0 }
    $eng  = if ($_.InstanceName -match 'engtype_(\w+)') { $Matches[1] } else { '?' }
    $name = (Get-Process -Id $pid_ -ErrorAction SilentlyContinue).ProcessName
    [pscustomobject]@{ PID = $pid_; Name = $name; Engine = $eng
                       Pct = [math]::Round($_.CookedValue, 1) }
}
$rows | Sort-Object Pct -Descending | Select-Object -First 15 | Format-Table -AutoSize
"total busy% = " + [math]::Round((($rows | Measure-Object Pct -Sum).Sum), 1)
"compute-only busy% = " + [math]::Round((($rows | Where-Object { $_.Engine -match 'Compute' } | Measure-Object Pct -Sum).Sum), 1)
