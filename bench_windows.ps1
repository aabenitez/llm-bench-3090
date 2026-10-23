<#
=============================================================================
 bench_windows.ps1 - Benchmark de LLMs locales con llama.cpp + llama-benchy
 Soporta 1 o 2 GPUs (ej. 2x RTX 3090 con o sin NVLink).

 Por cada modelo de modelos.txt y cada modo de GPU:
   1. levanta llama-server.exe (descarga el GGUF de Hugging Face si hace falta)
   2. mide el tiempo de carga
   3. corre llama-benchy (pp / tg / ttft a varias profundidades de contexto)
   4. muestrea las GPUs (VRAM, uso, consumo, temperatura) con nvidia-smi
   5. guarda todo en resultados\<fecha>_<etiqueta>\

 Modos de GPU (-ModosGpu):
   1gpu         un solo GPU (referencia)
   2gpu-layer   2 GPUs, reparto por capas (pipeline). El mas compatible.
   2gpu-tensor  2 GPUs, paralelismo de tensores (experimental). Es el que mas
                depende del enlace entre GPUs: aqui se nota NVLink.
   2gpu-row     2 GPUs, reparto por filas (obsoleto en llama.cpp, opcional)

 NVLink se detecta solo y queda en la etiqueta. Para medir "sin NVLink" hay
 que desactivarlo ANTES (ver LEEME.md) y volver a correr el script.

 Luego:  uv run --with matplotlib generar_graficos.py
 Requisitos: llama-server.exe (llama.cpp build CUDA), uv/uvx, nvidia-smi
 Compatible con Windows PowerShell 5.1 y PowerShell 7.

 Ejemplos:
   .\bench_windows.ps1 -Etiqueta 9950X3D-64GB -Depth "0 8192 16384" -Runs 5
   .\bench_windows.ps1 -ModosGpu "2gpu-layer 2gpu-tensor"
=============================================================================
#>
[CmdletBinding()]
param(
    [string]$Etiqueta     = "",
    [string]$Modelos      = (Join-Path $PSScriptRoot "modelos.txt"),
    [string]$Solo         = "",                 # ej. "qwen3-8b-q4,gpt-oss-20b"
    [string]$Gpus         = "",                 # ej. "0,1" (vacio = autodetectar las 2 primeras)
    [string]$ModosGpu     = "",                 # vacio = "1gpu" o "1gpu 2gpu-layer 2gpu-tensor"
    [ValidateSet("auto", "si", "no")]
    [string]$NvLink       = "auto",
    [switch]$SinP2P,                            # no exportar GGML_CUDA_P2P=1
    [string]$PP           = "512 2048",
    [string]$TG           = "128",
    [string]$Depth        = "0 4096 16384",
    [int]   $Runs         = 3,
    [string]$Concurrencia = "1",
    [int]   $Puerto       = 8080,
    [string]$Servidor     = $(if ($env:LLAMA_SERVER) { $env:LLAMA_SERVER } else { "llama-server.exe" }),
    [string]$LlamaBenchy  = $(if ($env:LLAMA_BENCHY) { $env:LLAMA_BENCHY } else { "llama-benchy" }),
    [int]   $TimeoutCarga = 3600,               # seg. (incluye la descarga la 1a vez)
    [int]   $Enfriamiento = 10
)

$ErrorActionPreference = "Continue"
$Inv = [Globalization.CultureInfo]::InvariantCulture
$Esc = [char]27

function Log  ($m) { Write-Host ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $m) -ForegroundColor Cyan }
function Warn ($m) { Write-Host ("[{0}] AVISO: {1}" -f (Get-Date -Format HH:mm:ss), $m) -ForegroundColor Yellow }
function Err  ($m) { Write-Host ("[{0}] ERROR: {1}" -f (Get-Date -Format HH:mm:ss), $m) -ForegroundColor Red }
function Nums ($s) { @($s -split '\s+' | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ }) }
function GuardarJson ($obj, $ruta) {
    # UTF-8 sin BOM (PowerShell 5.1 agrega BOM con -Encoding UTF8)
    [IO.File]::WriteAllText($ruta, ($obj | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding $false))
}
function NumD ($s) { [double]::Parse($s, $Inv) }

# Lee la celda GPU<a> x GPU<b> de "nvidia-smi topo <opcion>" (ej. NV4, PHB, OK, NS)
function CeldaTopo ([string[]]$opciones, $a, $b) {
    $out = & nvidia-smi topo @opciones 2>$null
    if (-not $out) { return "-" }
    $lineas = @($out | ForEach-Object { $_ -replace "$Esc\[[0-9;]*m", "" })
    $hdr = @($lineas[0] -split '\s+' | Where-Object { $_ })
    $col = [array]::IndexOf($hdr, "GPU$b")
    $fila = $lineas | Select-Object -Skip 1 | Where-Object { (($_ -split '\s+') | Where-Object { $_ } | Select-Object -First 1) -eq "GPU$a" } | Select-Object -First 1
    if ($col -lt 0 -or -not $fila) { return "-" }
    $campos = @($fila -split '\s+' | Where-Object { $_ })
    if ($campos.Count -le $col + 1) { return "-" }
    return $campos[$col + 1]
}

# Quita -ctk/-ctv: el modo tensor no admite cache KV cuantizada
function SinKvCuant ([string[]]$a) {
    $out = @(); $salta = $false
    foreach ($x in $a) {
        if ($salta) { $salta = $false; continue }
        if ($x -in @("-ctk", "-ctv", "--cache-type-k", "--cache-type-v")) { $salta = $true; continue }
        $out += $x
    }
    return ,$out
}

# ----------------------------- Verificaciones --------------------------------
foreach ($cmd in @($Servidor, "uvx", "nvidia-smi")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Err "No se encontro '$cmd'."
        if ($cmd -eq "uvx") { Write-Host '  Instalar uv:  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"' }
        else { Write-Host "  Descargar llama.cpp (build win-cuda) de https://github.com/ggml-org/llama.cpp/releases y usar -Servidor <ruta>\llama-server.exe" }
        exit 1
    }
}
$Servidor = (Get-Command $Servidor).Source
if (-not (Test-Path $Modelos)) { Err "No existe $Modelos"; exit 1 }

# ----------------------------- GPUs y NVLink ---------------------------------
if (-not $Gpus) {
    $Gpus = (@(& nvidia-smi --query-gpu=index --format=csv,noheader) | Select-Object -First 2 | ForEach-Object { $_.Trim() }) -join ","
}
$listaGpus = @($Gpus -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
$nGpus = $listaGpus.Count
$G0 = $listaGpus[0]; $G1 = if ($nGpus -ge 2) { $listaGpus[1] } else { "" }

if (-not $ModosGpu) { $ModosGpu = if ($nGpus -ge 2) { "1gpu 2gpu-layer 2gpu-tensor" } else { "1gpu" } }
$modos = @($ModosGpu -split '\s+' | Where-Object { $_ })
foreach ($m in $modos) {
    if ($m -notin @("1gpu", "2gpu-layer", "2gpu-tensor", "2gpu-row")) { Err "Modo de GPU desconocido: $m"; exit 1 }
}

$topo = "-"; $p2pEstado = "-"; $nvlinkEstado = "no aplica"
if ($nGpus -ge 2) {
    $topo = CeldaTopo @("-m") $G0 $G1
    $p2pEstado = CeldaTopo @("-p2p", "r") $G0 $G1
    if ($NvLink -eq "auto") {
        $nvs = (& nvidia-smi nvlink -s -i $G0 2>$null | Out-String)
        $NvLink = if ($topo -like "NV*" -or $nvs -match 'GB/s') { "si" } else { "no" }
    }
    $nvlinkEstado = $NvLink
} else {
    if ($modos | Where-Object { $_ -like "2gpu*" }) { Warn "Solo hay 1 GPU: se omiten los modos de 2 GPUs." }
    $modos = @("1gpu")
}

# ----------------------------- Info del sistema ------------------------------
$cpuInfo = Get-CimInstance Win32_Processor | Select-Object -First 1
$cs      = Get-CimInstance Win32_ComputerSystem
$osInfo  = Get-CimInstance Win32_OperatingSystem
$ramVel  = (Get-CimInstance Win32_PhysicalMemory | Select-Object -First 1).ConfiguredClockSpeed
$ramGB   = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
$gpuLine = (& nvidia-smi -i $G0 --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits) -split ',\s*'
$llamaVer = (& $Servidor --version 2>&1 | Out-String) -split "`n" | Where-Object { $_ -match 'version' } | Select-Object -First 1

if (-not $Etiqueta) {
    $m = [regex]::Match($cpuInfo.Name, '\d{4}X3D|\d{4}X?')
    $Etiqueta = "{0}-{1}GB" -f $m.Value, [math]::Round($ramGB)
    if ($nGpus -ge 2) { $Etiqueta += $(if ($NvLink -eq "si") { "-NVLink" } else { "-PCIe" }) }
}
$Etiqueta = $Etiqueta -replace '\s', '_'
$Salida = Join-Path $PSScriptRoot ("resultados\{0}_{1}" -f (Get-Date -Format yyyyMMdd_HHmmss), $Etiqueta)
New-Item -ItemType Directory -Force (Join-Path $Salida "logs") | Out-Null

GuardarJson ([ordered]@{
    etiqueta      = $Etiqueta
    fecha         = (Get-Date -Format o)
    so            = "$($osInfo.Caption) $($osInfo.Version)"
    cpu           = $cpuInfo.Name.Trim()
    nucleos       = $cpuInfo.NumberOfCores
    hilos         = $cpuInfo.NumberOfLogicalProcessors
    ram_gb        = $ramGB
    ram_velocidad = "$ramVel MT/s"
    gpu           = $gpuLine[0].Trim()
    n_gpus        = $nGpus
    gpus          = $Gpus
    gpu_vram_mb   = [int]$gpuLine[1]
    gpu_driver    = $gpuLine[2].Trim()
    nvlink        = $nvlinkEstado
    topologia     = $topo
    p2p_lectura   = $p2pEstado
    llama_cpp     = "$llamaVer".Trim()
    parametros    = [ordered]@{ pp = $PP; tg = $TG; depth = $Depth; runs = $Runs; concurrencia = $Concurrencia
                                modos_gpu = ($modos -join ' '); ggml_cuda_p2p = [int](-not $SinP2P) }
}) (Join-Path $Salida "sistema.json")

Log "Etiqueta:  $Etiqueta"
Log ("Sistema:   {0} ({1} nucleos / {2} hilos), {3} GB RAM @ {4} MT/s" -f `
     $cpuInfo.Name.Trim(), $cpuInfo.NumberOfCores, $cpuInfo.NumberOfLogicalProcessors, $ramGB, $ramVel)
Log ("GPUs:      {0}x {1} ({2} MB c/u) - indices {3}" -f $nGpus, $gpuLine[0].Trim(), $gpuLine[1], $Gpus)
if ($nGpus -ge 2) { Log "Enlace:    NVLink=$nvlinkEstado - topologia=$topo - P2P lectura=$p2pEstado" }
Log "Modos:     $($modos -join ' ')"
Log "Salida:    $Salida"

# ----------------------------- Bucle de modelos ------------------------------
$maxPP   = (Nums $PP | Measure-Object -Maximum).Maximum
$maxTG   = (Nums $TG | Measure-Object -Maximum).Maximum
$maxConc = (Nums $Concurrencia | Measure-Object -Maximum).Maximum
$listaSolo = @($Solo -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$ok = 0; $fallos = 0; $saltados = 0
$envOriginal = @{ CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES; GGML_CUDA_P2P = $env:GGML_CUDA_P2P; CUDA_DEVICE_ORDER = $env:CUDA_DEVICE_ORDER }

foreach ($linea in Get-Content $Modelos -Encoding UTF8) {
    $t = $linea.Trim()
    if (-not $t -or $t.StartsWith('#')) { continue }
    $c = @($linea -split '\|' | ForEach-Object { $_.Trim() })
    $nombre = $c[0]; $repo = $c[1]; $tokenizer = $c[2]; $ctx = [int]$c[3]; $ramMin = [double]$c[4]
    $extra     = if ($c.Count -gt 5) { $c[5] } else { "" }
    $gpusMin   = if ($c.Count -gt 6 -and $c[6]) { [int]$c[6] } else { 1 }
    $extra2gpu = if ($c.Count -gt 7) { $c[7] } else { "" }

    if ($listaSolo.Count -and ($listaSolo -notcontains $nombre)) { continue }

    if ($ramGB -lt $ramMin * 0.9) {
        Warn "Se salta ${nombre}: requiere $ramMin GB de RAM y hay $ramGB GB."
        $saltados++; continue
    }

    $depthsOk = @()
    foreach ($d in (Nums $Depth)) {
        if ($d + $maxPP + $maxTG -le $ctx) { $depthsOk += $d } else { Warn "${nombre}: depth $d no entra en contexto $ctx, se omite." }
    }
    if (-not $depthsOk.Count) { Warn "Ninguna profundidad entra en el contexto; se salta."; $saltados++; continue }

    foreach ($modo in $modos) {
        $ng = if ($modo -like "2gpu*") { 2 } else { 1 }
        if ($ng -lt $gpusMin) { Warn "$nombre no entra en $ng GPU: se omite el modo $modo."; continue }

        # Argumentos y entorno segun el modo
        $argsModo = @(); $argsModelo = @($extra -split '\s+' | Where-Object { $_ })
        $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
        if ($ng -eq 1) {
            $env:CUDA_VISIBLE_DEVICES = "$G0"
            Remove-Item Env:GGML_CUDA_P2P -ErrorAction SilentlyContinue
            $monGpus = "$G0"
        } else {
            $env:CUDA_VISIBLE_DEVICES = "$G0,$G1"
            if ($SinP2P) { Remove-Item Env:GGML_CUDA_P2P -ErrorAction SilentlyContinue } else { $env:GGML_CUDA_P2P = "1" }
            $monGpus = "$G0,$G1"
            if ($extra2gpu) { $argsModelo = @($extra2gpu -split '\s+' | Where-Object { $_ -and $_ -ne "-" }) }
            switch ($modo) {
                "2gpu-layer"  { $argsModo = @("-sm", "layer", "-ts", "1,1") }
                "2gpu-row"    { $argsModo = @("-sm", "row", "-ts", "1,1") }
                "2gpu-tensor" { $argsModo = @("-sm", "tensor", "-ts", "1,1", "-fa", "on", "-ctk", "f16", "-ctv", "f16")
                                $argsModelo = SinKvCuant $argsModelo }
            }
        }

        $id = "$nombre@$modo"
        Write-Host ""
        Log "===================== $nombre  -  $modo ====================="

        $vramBase = (@(& nvidia-smi -i $monGpus --query-gpu=memory.used --format=csv,noheader,nounits) | ForEach-Object { [int]$_ } | Measure-Object -Sum).Sum
        $logSrv   = Join-Path $Salida "logs\$id.server.log"
        $gpuCsv   = Join-Path $Salida "logs\$id.gpu.csv"

        # 1) Levantar llama-server (hereda CUDA_VISIBLE_DEVICES / GGML_CUDA_P2P)
        $argsSrv = @("-hf", $repo, "--alias", $nombre, "-c", ($ctx * $maxConc), "-np", $maxConc,
                     "-ngl", "999", "--jinja", "--host", "127.0.0.1", "--port", $Puerto) + $argsModo + $argsModelo
        $srv = Start-Process -FilePath $Servidor -ArgumentList $argsSrv -NoNewWindow -PassThru `
                             -RedirectStandardOutput "$logSrv.out" -RedirectStandardError $logSrv
        $reloj = [Diagnostics.Stopwatch]::StartNew()
        Log "llama-server iniciado (PID $($srv.Id)) con: $(($argsModo + $argsModelo) -join ' ')"
        Log "Esperando carga (la 1a vez descarga el modelo)..."

        $listo = $false
        while (-not $srv.HasExited -and $reloj.Elapsed.TotalSeconds -lt $TimeoutCarga) {
            try {
                $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$Puerto/health"
                if ($r.StatusCode -eq 200) { $listo = $true; break }
            } catch { }
            Start-Sleep -Seconds 1
        }
        $tCarga = [math]::Round($reloj.Elapsed.TotalSeconds, 1)

        $estado = "ok"; $monitor = $null
        if (-not $listo) {
            Err "El servidor no arranco. Ultimas lineas del log:"
            Get-Content $logSrv -Tail 15 -ErrorAction SilentlyContinue
            $estado = "error_carga"
        } else {
            Log "Modelo cargado en ${tCarga}s"

            # 2) Monitor de GPU en segundo plano (1 muestra por segundo, una linea por GPU)
            "muestra,gpu,vram_mb,uso_pct,potencia_w,temp_c" | Set-Content $gpuCsv -Encoding Ascii
            $monitor = Start-Job -ArgumentList $gpuCsv, $monGpus -ScriptBlock {
                param($archivo, $gpus)
                $n = 0
                while ($true) {
                    $n++
                    $ls = & nvidia-smi -i $gpus --query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits
                    foreach ($l in @($ls)) { Add-Content $archivo ("$n," + ($l -replace '\s', '')) -Encoding Ascii }
                    Start-Sleep -Seconds 1
                }
            }

            # 3) Benchmark
            $jsonOut = Join-Path $Salida "$id.json"
            Log "llama-benchy: pp=[$PP] tg=[$TG] depth=[$($depthsOk -join ' ')] runs=$Runs conc=[$Concurrencia]"
            $argsBench = @($LlamaBenchy, "--base-url", "http://127.0.0.1:$Puerto/v1",
                           "--model", $nombre, "--tokenizer", $tokenizer) +
                         @("--pp") + (Nums $PP) + @("--tg") + (Nums $TG) + @("--depth") + $depthsOk +
                         @("--runs", $Runs, "--concurrency") + (Nums $Concurrencia) +
                         @("--latency-mode", "generation", "--skip-coherence",
                           "--save-result", $jsonOut, "--format", "json")
            $env:PYTHONIOENCODING = "utf-8"
            & uvx @argsBench 2>&1 | Tee-Object -FilePath (Join-Path $Salida "logs\$id.bench.log")
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path $jsonOut)) { $estado = "error_benchmark" }
        }

        # Detener monitor y servidor
        if ($monitor) { Stop-Job $monitor; Remove-Job $monitor -Force }
        if (-not $srv.HasExited) {
            Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
            $srv.WaitForExit(30000) | Out-Null
        }

        # 4) Estadisticas de GPU (VRAM y potencia sumadas entre GPUs por muestra)
        $vramMax = 0; $usoProm = 0; $potProm = 0; $potMax = 0; $tempMax = 0; $porGpu = "-"
        if (Test-Path $gpuCsv) {
            $filas = @(Import-Csv $gpuCsv | Where-Object { $_.vram_mb -match '^[\d.]+$' })
            if ($filas.Count) {
                $muestras = $filas | Group-Object muestra
                $vramMax = ($muestras | ForEach-Object { ($_.Group | ForEach-Object { NumD $_.vram_mb } | Measure-Object -Sum).Sum } | Measure-Object -Maximum).Maximum
                $pot = @($muestras | ForEach-Object {
                    $p = @($_.Group | Where-Object { $_.potencia_w -match '^[\d.]+$' } | ForEach-Object { NumD $_.potencia_w })
                    if ($p.Count) { ($p | Measure-Object -Sum).Sum } })
                if ($pot.Count) { $mp = $pot | Measure-Object -Average -Maximum; $potProm = [math]::Round($mp.Average, 1); $potMax = [math]::Round($mp.Maximum, 1) }
                $usoProm = [math]::Round(($filas | ForEach-Object { NumD $_.uso_pct } | Measure-Object -Average).Average, 1)
                $tempMax = ($filas | ForEach-Object { NumD $_.temp_c } | Measure-Object -Maximum).Maximum
                $porGpu = (($monGpus -split ',') | ForEach-Object {
                    $g = $_; ($filas | Where-Object { $_.gpu -eq $g } | ForEach-Object { NumD $_.vram_mb } | Measure-Object -Maximum).Maximum
                }) -join '/'
            }
        }

        GuardarJson ([ordered]@{
            nombre = $nombre; modo_gpu = $modo; n_gpus_usadas = $ng; archivo = "$id.json"
            repo = $repo; tokenizer = $tokenizer; contexto = $ctx; args_extra = (($argsModo + $argsModelo) -join ' ')
            ggml_cuda_p2p = [int]($ng -gt 1 -and -not $SinP2P)
            estado = $estado; tiempo_carga_s = $tCarga
            vram_base_mb = $vramBase; vram_max_mb = [int]$vramMax; vram_max_por_gpu_mb = $porGpu
            gpu_uso_prom_pct = $usoProm; potencia_prom_w = $potProm; potencia_max_w = $potMax; temp_max_c = [int]$tempMax
        }) (Join-Path $Salida "$id.meta.json")

        if ($estado -eq "ok") { $ok++; Log "OK $id  (VRAM max $vramMax MB [$porGpu], $potProm W prom.)" }
        else {
            $fallos++; Err "$id : $estado (ver $Salida\logs\)"
            if ($modo -eq "2gpu-tensor") { Warn "El modo tensor es experimental y no admite muchas arquitecturas MoE; es normal que algunos fallen." }
        }
        Start-Sleep -Seconds $Enfriamiento
    }
}

# Restaurar variables de entorno de la sesion
foreach ($k in $envOriginal.Keys) {
    if ($null -eq $envOriginal[$k]) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } else { Set-Item "Env:$k" $envOriginal[$k] }
}

Write-Host ""
Log "Terminado: $ok ok, $fallos con error, $saltados saltados."
Log "Resultados en: $Salida"
Log "Generar graficos:  uv run --with matplotlib `"$PSScriptRoot\generar_graficos.py`""
