#!/usr/bin/env bash
# =============================================================================
#  bench_linux.sh — Benchmark de LLMs locales con llama.cpp + llama-benchy
#  Soporta 1 o 2 GPUs (ej. 2x RTX 3090 con o sin NVLink).
#
#  Por cada modelo de modelos.txt y cada modo de GPU:
#    1. levanta llama-server (descarga el GGUF de Hugging Face si hace falta)
#    2. mide el tiempo de carga
#    3. corre llama-benchy (pp / tg / ttft a varias profundidades de contexto)
#    4. muestrea las GPUs (VRAM, uso, consumo, temperatura) con nvidia-smi
#    5. guarda todo en resultados/<fecha>_<etiqueta>/
#
#  Modos de GPU (--modos-gpu):
#    1gpu         un solo GPU (referencia)
#    2gpu-layer   2 GPUs, reparto por capas (pipeline). El más compatible.
#    2gpu-tensor  2 GPUs, paralelismo de tensores (experimental). Es el que más
#                 depende del enlace entre GPUs: aquí se nota NVLink.
#    2gpu-row     2 GPUs, reparto por filas (obsoleto en llama.cpp, opcional)
#
#  NVLink se detecta solo (nvidia-smi topo / nvlink) y queda en la etiqueta.
#  Para medir "sin NVLink" hay que desactivarlo ANTES (ver LEEME.md) y volver
#  a correr el script.
#
#  Luego: uv run --with matplotlib generar_graficos.py
#  Requisitos: llama-server (llama.cpp con CUDA), uv/uvx, nvidia-smi, curl
# =============================================================================
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------- Valores por defecto ---------------------------
ETIQUETA=""
MODELOS_FILE="$DIR/modelos.txt"
PP="512 2048"
TG="128"
DEPTHS="0 4096 16384"
RUNS=3
CONCURRENCIA="1"
PUERTO=8080
GPUS=""                 # ej. "0,1" (vacío = autodetectar las 2 primeras)
MODOS_GPU=""            # vacío = "1gpu" con 1 GPU, "1gpu 2gpu-layer 2gpu-tensor" con 2
NVLINK="auto"           # auto | si | no
P2P=1                   # exporta GGML_CUDA_P2P=1 en modos de 2 GPUs
SOLO=""
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
LLAMA_BENCHY="${LLAMA_BENCHY:-llama-benchy}"   # spec de uvx, ej. llama-benchy==0.2.2
TIMEOUT_CARGA=3600                              # seg. (incluye la descarga la 1ª vez)
ENFRIAMIENTO=10                                 # seg. de pausa entre pruebas

uso() {
  cat <<EOF
Uso: $0 [opciones]

  --etiqueta TXT       Nombre de esta configuración (ej. "9950X3D-NVLink").
                       Por defecto: CPU-RAM y, con 2 GPUs, -NVLink o -PCIe.
  --modelos ARCHIVO    Lista de modelos (por defecto: modelos.txt)
  --solo a,b,c         Ejecutar solo estos modelos (por nombre)
  --gpus "0,1"         Índices de GPU a usar (defecto: autodetectar)
  --modos-gpu "..."    Modos a probar: 1gpu 2gpu-layer 2gpu-tensor 2gpu-row
                       (defecto con 2 GPUs: "1gpu 2gpu-layer 2gpu-tensor")
  --nvlink si|no|auto  Forzar el estado de NVLink en la etiqueta (defecto: auto)
  --sin-p2p            No activar GGML_CUDA_P2P (si ves cuelgues o salida corrupta)
  --pp "N N"           Tamaños de prompt en tokens       (defecto: "$PP")
  --tg "N"             Tokens a generar                  (defecto: "$TG")
  --depth "N N N"      Profundidades de contexto previo  (defecto: "$DEPTHS")
  --runs N             Repeticiones por prueba           (defecto: $RUNS)
  --concurrencia "N N" Clientes simultáneos              (defecto: "$CONCURRENCIA")
  --puerto N           Puerto de llama-server            (defecto: $PUERTO)
  --servidor RUTA      Ruta a llama-server               (defecto: $LLAMA_SERVER)
  -h, --ayuda          Mostrar esta ayuda

Ejemplos:
  $0 --etiqueta 7800X3D-32GB                       # 1 GPU
  $0 --modos-gpu "2gpu-layer 2gpu-tensor"          # 2 GPUs, NVLink autodetectado
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --etiqueta)     ETIQUETA="$2"; shift 2 ;;
    --modelos)      MODELOS_FILE="$2"; shift 2 ;;
    --solo)         SOLO="$2"; shift 2 ;;
    --gpus)         GPUS="$2"; shift 2 ;;
    --modos-gpu)    MODOS_GPU="$2"; shift 2 ;;
    --nvlink)       NVLINK="$2"; shift 2 ;;
    --sin-p2p)      P2P=0; shift ;;
    --pp)           PP="$2"; shift 2 ;;
    --tg)           TG="$2"; shift 2 ;;
    --depth)        DEPTHS="$2"; shift 2 ;;
    --runs)         RUNS="$2"; shift 2 ;;
    --concurrencia) CONCURRENCIA="$2"; shift 2 ;;
    --puerto)       PUERTO="$2"; shift 2 ;;
    --servidor)     LLAMA_SERVER="$2"; shift 2 ;;
    -h|--ayuda)     uso; exit 0 ;;
    *) echo "Opción desconocida: $1"; uso; exit 1 ;;
  esac
done

log()  { printf '\e[1;36m[%s]\e[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\e[1;33m[%s] AVISO:\e[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
err()  { printf '\e[1;31m[%s] ERROR:\e[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
trim() { local s="$*"; s="${s#"${s%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"; }
jesc() { local s="$1"; s="${s//\\/\\\\}"; s="${s//\"/\\\"}"; printf '%s' "$s"; }
max_de() { local m=0; for v in "$@"; do (( v > m )) && m=$v; done; echo "$m"; }
sin_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }

# ----------------------------- Verificaciones --------------------------------
for cmd in "$LLAMA_SERVER" uvx nvidia-smi curl awk; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    err "No se encontró '$cmd' en el PATH."
    case "$cmd" in
      uvx) echo "  Instalar uv:  curl -LsSf https://astral.sh/uv/install.sh | sh" ;;
      "$LLAMA_SERVER") echo "  Compilar llama.cpp con CUDA o indicar la ruta con --servidor / LLAMA_SERVER" ;;
    esac
    exit 1
  fi
done
[[ -f "$MODELOS_FILE" ]] || { err "No existe $MODELOS_FILE"; exit 1; }

# ----------------------------- GPUs y NVLink ---------------------------------
if [[ -z "$GPUS" ]]; then
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -2 | paste -sd, -)
fi
IFS=',' read -r -a LISTA_GPUS <<< "$GPUS"
N_GPUS=${#LISTA_GPUS[@]}
G0=${LISTA_GPUS[0]}; G1=${LISTA_GPUS[1]:-}

if [[ -z "$MODOS_GPU" ]]; then
  (( N_GPUS >= 2 )) && MODOS_GPU="1gpu 2gpu-layer 2gpu-tensor" || MODOS_GPU="1gpu"
fi

TOPO="-"; P2P_ESTADO="-"; NVLINK_ESTADO="no aplica"
if (( N_GPUS >= 2 )); then
  # Enlace GPU0<->GPU1 según "nvidia-smi topo -m": NV# = NVLink, PIX/PXB/PHB/NODE/SYS = PCIe
  TOPO=$(nvidia-smi topo -m 2>/dev/null | sin_ansi | awk -v a="GPU$G0" -v b="GPU$G1" '
           NR==1 { for (i=1; i<=NF; i++) if ($i==b) col=i }
           NR>1 && $1==a && col { print $(col+1); exit }')
  P2P_ESTADO=$(nvidia-smi topo -p2p r 2>/dev/null | sin_ansi | awk -v a="GPU$G0" -v b="GPU$G1" '
           NR==1 { for (i=1; i<=NF; i++) if ($i==b) col=i }
           NR>1 && $1==a && col { print $(col+1); exit }')
  if [[ "$NVLINK" == "auto" ]]; then
    if [[ "$TOPO" == NV* ]] || nvidia-smi nvlink -s -i "$G0" 2>/dev/null | grep -q 'GB/s'; then
      NVLINK="si"; else NVLINK="no"; fi
  fi
  NVLINK_ESTADO="$NVLINK"
  for m in $MODOS_GPU; do
    case "$m" in 1gpu|2gpu-layer|2gpu-tensor|2gpu-row) ;; *) err "Modo de GPU desconocido: $m"; exit 1 ;; esac
  done
else
  [[ "$MODOS_GPU" == *2gpu* ]] && warn "Solo hay 1 GPU: se omiten los modos de 2 GPUs."
  MODOS_GPU="1gpu"
fi

# ----------------------------- Info del sistema ------------------------------
CPU=$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +/,"",$2); print $2; exit}')
NUCLEOS=$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l)
HILOS=$(nproc)
RAM_GB=$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)
RAM_VEL=$(sudo -n dmidecode -t memory 2>/dev/null | awk -F: '/Configured Memory Speed/ && $2 !~ /Unknown/ {gsub(/^ +/,"",$2); print $2; exit}')
IFS=',' read -r GPU_NOMBRE GPU_VRAM GPU_DRIVER < <(nvidia-smi -i "$G0" --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
GPU_NOMBRE=$(trim "$GPU_NOMBRE"); GPU_VRAM=$(trim "$GPU_VRAM"); GPU_DRIVER=$(trim "$GPU_DRIVER")
SO=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"; uname -r | tr '\n' ' ')
LLAMA_VER=$("$LLAMA_SERVER" --version 2>&1 | grep -m1 -i version | tr -d '\r')

if [[ -z "$ETIQUETA" ]]; then
  ETIQUETA="$(echo "$CPU" | grep -oE '[0-9]{4}X3D|[0-9]{4}X?' | head -1)-${RAM_GB%.*}GB"
  (( N_GPUS >= 2 )) && { [[ "$NVLINK" == "si" ]] && ETIQUETA+="-NVLink" || ETIQUETA+="-PCIe"; }
fi
ETIQUETA="${ETIQUETA// /_}"
SALIDA="$DIR/resultados/$(date +%Y%m%d_%H%M%S)_${ETIQUETA}"
mkdir -p "$SALIDA/logs"

cat > "$SALIDA/sistema.json" <<EOF
{
  "etiqueta": "$(jesc "$ETIQUETA")",
  "fecha": "$(date -Iseconds)",
  "so": "$(jesc "$(trim "$SO")")",
  "cpu": "$(jesc "$CPU")",
  "nucleos": $NUCLEOS,
  "hilos": $HILOS,
  "ram_gb": $RAM_GB,
  "ram_velocidad": "$(jesc "${RAM_VEL:-desconocida}")",
  "gpu": "$(jesc "$GPU_NOMBRE")",
  "n_gpus": $N_GPUS,
  "gpus": "$GPUS",
  "gpu_vram_mb": $GPU_VRAM,
  "gpu_driver": "$(jesc "$GPU_DRIVER")",
  "nvlink": "$NVLINK_ESTADO",
  "topologia": "$(jesc "${TOPO:--}")",
  "p2p_lectura": "$(jesc "${P2P_ESTADO:--}")",
  "llama_cpp": "$(jesc "$LLAMA_VER")",
  "parametros": {"pp": "$PP", "tg": "$TG", "depth": "$DEPTHS", "runs": $RUNS, "concurrencia": "$CONCURRENCIA", "modos_gpu": "$MODOS_GPU", "ggml_cuda_p2p": $P2P}
}
EOF

log "Etiqueta:  $ETIQUETA"
log "Sistema:   $CPU ($NUCLEOS núcleos / $HILOS hilos), ${RAM_GB} GB RAM"
log "GPUs:      ${N_GPUS}x $GPU_NOMBRE (${GPU_VRAM} MB c/u) · índices $GPUS"
(( N_GPUS >= 2 )) && log "Enlace:    NVLink=$NVLINK_ESTADO · topología=$TOPO · P2P lectura=$P2P_ESTADO"
log "Modos:     $MODOS_GPU"
log "Salida:    $SALIDA"

# ----------------------------- Limpieza --------------------------------------
SERVER_PID=""; MONITOR_PID=""
detener() {
  [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null && wait "$MONITOR_PID" 2>/dev/null
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""; MONITOR_PID=""
}
trap 'echo; warn "Interrumpido"; detener; exit 130' INT TERM

# Quita "-ctk X" / "-ctv X" (y formas largas): el modo tensor no admite caché KV cuantizada
sin_kv_cuant() {
  local out=() salta=0
  for a in $1; do
    if (( salta )); then salta=0; continue; fi
    case "$a" in -ctk|-ctv|--cache-type-k|--cache-type-v) salta=1; continue ;; esac
    out+=("$a")
  done
  echo "${out[*]}"
}

# ----------------------------- Bucle de modelos ------------------------------
MAX_PP=$(max_de $PP); MAX_TG=$(max_de $TG); MAX_CONC=$(max_de $CONCURRENCIA)
mapfile -t LINEAS < "$MODELOS_FILE"
OK=0; FALLOS=0; SALTADOS=0

for linea in "${LINEAS[@]}"; do
  linea="${linea%$'\r'}"
  [[ -z "$(trim "$linea")" || "$(trim "$linea")" == \#* ]] && continue
  IFS='|' read -r NOMBRE REPO TOKENIZER CTX RAM_MIN EXTRA GPUS_MIN EXTRA_2GPU <<< "$linea"
  NOMBRE=$(trim "$NOMBRE"); REPO=$(trim "$REPO"); TOKENIZER=$(trim "$TOKENIZER")
  CTX=$(trim "$CTX"); RAM_MIN=$(trim "$RAM_MIN"); EXTRA=$(trim "${EXTRA:-}")
  GPUS_MIN=$(trim "${GPUS_MIN:-}"); GPUS_MIN=${GPUS_MIN:-1}; EXTRA_2GPU=$(trim "${EXTRA_2GPU:-}")

  if [[ -n "$SOLO" && ",$SOLO," != *",$NOMBRE,"* ]]; then continue; fi

  # ¿Alcanza la RAM? (tolerancia 10 %: 64 GB nominales se ven como ~62 GB)
  if awk -v r="$RAM_GB" -v m="$RAM_MIN" 'BEGIN{exit !(r < m*0.9)}'; then
    warn "Se salta $NOMBRE: requiere ${RAM_MIN} GB de RAM y hay ${RAM_GB} GB."
    SALTADOS=$((SALTADOS+1)); continue
  fi

  # Profundidades que entran en el contexto del modelo
  DEPTHS_OK=""
  for d in $DEPTHS; do
    (( d + MAX_PP + MAX_TG <= CTX )) && DEPTHS_OK+="$d " || warn "$NOMBRE: depth $d no entra en contexto $CTX, se omite."
  done
  DEPTHS_OK=$(trim "$DEPTHS_OK")
  [[ -z "$DEPTHS_OK" ]] && { warn "Ninguna profundidad entra en el contexto; se salta."; SALTADOS=$((SALTADOS+1)); continue; }

  for MODO in $MODOS_GPU; do
    NG=1; [[ "$MODO" == 2gpu-* ]] && NG=2
    if (( NG < GPUS_MIN )); then
      warn "$NOMBRE no entra en $NG GPU: se omite el modo $MODO."; continue
    fi

    # Argumentos y entorno según el modo
    ARGS_MODO=(); ENTORNO=(CUDA_DEVICE_ORDER=PCI_BUS_ID); ARGS_MODELO="$EXTRA"
    if (( NG == 1 )); then
      ENTORNO+=(CUDA_VISIBLE_DEVICES="$G0")
      MON_GPUS="$G0"
    else
      ENTORNO+=(CUDA_VISIBLE_DEVICES="$G0,$G1")
      (( P2P )) && ENTORNO+=(GGML_CUDA_P2P=1)
      MON_GPUS="$G0,$G1"
      [[ -n "$EXTRA_2GPU" ]] && ARGS_MODELO="$EXTRA_2GPU"
      [[ "$ARGS_MODELO" == "-" ]] && ARGS_MODELO=""
      case "$MODO" in
        2gpu-layer)  ARGS_MODO=(-sm layer -ts 1,1) ;;
        2gpu-row)    ARGS_MODO=(-sm row -ts 1,1) ;;
        2gpu-tensor) ARGS_MODO=(-sm tensor -ts 1,1 -fa on -ctk f16 -ctv f16)
                     ARGS_MODELO=$(sin_kv_cuant "$ARGS_MODELO") ;;
      esac
    fi

    ID="${NOMBRE}@${MODO}"
    echo
    log "===================== $NOMBRE  ·  $MODO ====================="

    VRAM_BASE=$(nvidia-smi -i "$MON_GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')

    # 1) Levantar llama-server
    # shellcheck disable=SC2086
    env "${ENTORNO[@]}" "$LLAMA_SERVER" -hf "$REPO" --alias "$NOMBRE" \
        -c $((CTX * MAX_CONC)) -np "$MAX_CONC" -ngl 999 --jinja \
        --host 127.0.0.1 --port "$PUERTO" ${ARGS_MODO[@]+"${ARGS_MODO[@]}"} $ARGS_MODELO \
        > "$SALIDA/logs/$ID.server.log" 2>&1 < /dev/null &
    SERVER_PID=$!
    T0=$(date +%s.%N)
    log "llama-server iniciado (PID $SERVER_PID) con: ${ARGS_MODO[*]} $ARGS_MODELO"
    log "Esperando carga (la 1ª vez descarga el modelo)..."

    LISTO=0
    while true; do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
      if curl -sf "http://127.0.0.1:$PUERTO/health" >/dev/null 2>&1; then LISTO=1; break; fi
      if awk -v t0="$T0" -v now="$(date +%s.%N)" -v lim="$TIMEOUT_CARGA" 'BEGIN{exit !(now-t0 > lim)}'; then break; fi
      sleep 1
    done
    T_CARGA=$(awk -v t0="$T0" -v now="$(date +%s.%N)" 'BEGIN{printf "%.1f", now-t0}')

    ESTADO="ok"
    GPU_CSV="$SALIDA/logs/$ID.gpu.csv"
    if (( ! LISTO )); then
      err "El servidor no arrancó. Últimas líneas del log:"
      tail -n 15 "$SALIDA/logs/$ID.server.log"
      ESTADO="error_carga"
    else
      log "Modelo cargado en ${T_CARGA}s"

      # 2) Monitor de GPU (1 muestra por segundo, una línea por GPU)
      echo "muestra,gpu,vram_mb,uso_pct,potencia_w,temp_c" > "$GPU_CSV"
      ( n=0
        while :; do
          n=$((n+1))
          nvidia-smi -i "$MON_GPUS" --query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu \
                     --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed "s/^/$n,/" >> "$GPU_CSV"
          sleep 1
        done ) &
      MONITOR_PID=$!

      # 3) Benchmark
      log "llama-benchy: pp=[$PP] tg=[$TG] depth=[$DEPTHS_OK] runs=$RUNS conc=[$CONCURRENCIA]"
      # shellcheck disable=SC2086
      if ! uvx "$LLAMA_BENCHY" \
            --base-url "http://127.0.0.1:$PUERTO/v1" \
            --model "$NOMBRE" --tokenizer "$TOKENIZER" \
            --pp $PP --tg $TG --depth $DEPTHS_OK --runs "$RUNS" \
            --concurrency $CONCURRENCIA \
            --latency-mode generation --skip-coherence \
            --save-result "$SALIDA/$ID.json" --format json \
            < /dev/null 2>&1 | tee "$SALIDA/logs/$ID.bench.log"; then
        ESTADO="error_benchmark"
      fi
      [[ -s "$SALIDA/$ID.json" ]] || ESTADO="error_benchmark"
    fi

    detener

    # 4) Estadísticas de GPU (VRAM y potencia sumadas entre GPUs por muestra)
    read -r VRAM_MAX USO_PROM POT_PROM POT_MAX TEMP_MAX VRAM_POR_GPU < <(
      awk -F, -v ids="$MON_GPUS" 'NR>1 && $3 ~ /^[0-9.]+$/ {
          mem[$1]+=$3; if ($3>gm[$2]) gm[$2]=$3; su+=$4; n++
          if ($5 ~ /^[0-9.]+$/) pw[$1]+=$5
          if ($6>tm) tm=$6 }
        END {
          for (s in mem) { if (mem[s]>vm) vm=mem[s] }
          for (s in pw)  { sp+=pw[s]; np++; if (pw[s]>pm) pm=pw[s] }
          k=split(ids, lista, ","); pg=""; for (i=1; i<=k; i++) pg=pg (i>1?"/":"") (gm[lista[i]]+0)
          printf "%d %.1f %.1f %.1f %d %s\n", vm, (n?su/n:0), (np?sp/np:0), pm, tm, (pg?pg:"-") }' \
        "$GPU_CSV" 2>/dev/null || echo "0 0 0 0 0 -")

    cat > "$SALIDA/$ID.meta.json" <<EOF
{
  "nombre": "$(jesc "$NOMBRE")",
  "modo_gpu": "$MODO",
  "n_gpus_usadas": $NG,
  "archivo": "$(jesc "$ID.json")",
  "repo": "$(jesc "$REPO")",
  "tokenizer": "$(jesc "$TOKENIZER")",
  "contexto": $CTX,
  "args_extra": "$(jesc "${ARGS_MODO[*]} $ARGS_MODELO")",
  "ggml_cuda_p2p": $(( NG > 1 && P2P )),
  "estado": "$ESTADO",
  "tiempo_carga_s": $T_CARGA,
  "vram_base_mb": ${VRAM_BASE:-0},
  "vram_max_mb": ${VRAM_MAX:-0},
  "vram_max_por_gpu_mb": "${VRAM_POR_GPU:--}",
  "gpu_uso_prom_pct": ${USO_PROM:-0},
  "potencia_prom_w": ${POT_PROM:-0},
  "potencia_max_w": ${POT_MAX:-0},
  "temp_max_c": ${TEMP_MAX:-0}
}
EOF

    if [[ "$ESTADO" == "ok" ]]; then
      OK=$((OK+1)); log "✔ $ID listo  (VRAM máx ${VRAM_MAX} MB [$VRAM_POR_GPU], ${POT_PROM} W prom.)"
    else
      FALLOS=$((FALLOS+1)); err "✘ $ID: $ESTADO (ver $SALIDA/logs/)"
      [[ "$MODO" == "2gpu-tensor" ]] && warn "El modo tensor es experimental y no admite muchas arquitecturas MoE; es normal que algunos fallen."
    fi
    sleep "$ENFRIAMIENTO"
  done
done

echo
log "Terminado: $OK ok, $FALLOS con error, $SALTADOS saltados."
log "Resultados en: $SALIDA"
log "Generar gráficos:  uv run --with matplotlib \"$DIR/generar_graficos.py\""
