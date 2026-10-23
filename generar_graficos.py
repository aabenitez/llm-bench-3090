#!/usr/bin/env python3
"""
generar_graficos.py - Gráficos analíticos y comparativos (en español) a partir de
los resultados de bench_linux.sh / bench_windows.ps1 (formato JSON de llama-benchy).

Uso:
    uv run --with matplotlib generar_graficos.py                 # todas las corridas
    uv run --with matplotlib generar_graficos.py --base 7800X3D-64GB
    python3 generar_graficos.py --resultados ./resultados --salida ./informe

Si hay varias corridas con la MISMA etiqueta, se usa la más reciente por modelo.
Corridas con etiquetas distintas (ej. 7800X3D-32GB vs 9950X3D-64GB) se comparan entre sí.

Salida (carpeta resultados/informe/ por defecto):
    01..10_*.png        gráficos
    resumen.csv         datos consolidados (separador "," y punto decimal)
    resumen_excel.csv   igual, pero con ";" y coma decimal (Excel en español)
    informe.html        informe con tablas y todos los gráficos
"""
import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# ----------------------------------------------------------------------------- estilo
SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
GRILLA = "#e4e3df"
# Paleta categórica (orden fijo, validada para daltonismo): una por etiqueta de hardware
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AZULES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("azules", AZULES)
VRAM_RTX3090_GB = 24

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": GRILLA, "axes.labelcolor": TEXTO_2, "axes.titlecolor": TEXTO,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRILLA,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
    "font.size": 10, "text.color": TEXTO,
})


def num(v, dec=1):
    """Formato numérico en español: 1.234,5"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = f"{v:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def ctx_txt(d):
    return "sin contexto" if d == 0 else (f"{d // 1024}K" if d % 1024 == 0 else str(d))


# ----------------------------------------------------------------------------- carga
def leer_json(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


MODOS_TXT = {"1gpu": "1 GPU", "2gpu-layer": "2 GPU capas", "2gpu-tensor": "2 GPU tensor", "2gpu-row": "2 GPU filas"}
ORDEN_MODOS = list(MODOS_TXT)


def cargar(dir_res: Path):
    """Devuelve (sistemas por hardware, filas). Cada fila lleva 'etiqueta' = serie a graficar:
    el hardware solo, o "hardware · modo" si hay más de un modo de GPU en los datos."""
    sistemas, filas, metas = {}, {}, {}
    for corrida in sorted(d for d in dir_res.iterdir() if (d / "sistema.json").exists()):
        sis = leer_json(corrida / "sistema.json")
        et = sis["etiqueta"]
        sistemas[et] = sis  # la corrida más reciente gana
        for meta_p in sorted(corrida.glob("*.meta.json")):
            meta = leer_json(meta_p)
            modelo = meta["nombre"]
            modo = meta.get("modo_gpu", "1gpu")
            bench_p = corrida / meta.get("archivo", f"{modelo}.json")
            if meta.get("estado") != "ok" or not bench_p.exists():
                continue
            bench = leer_json(bench_p)
            filas_modelo = []
            for b in bench.get("benchmarks", []):
                if b.get("is_context_prefill_phase"):
                    continue
                m = lambda k: (b.get(k) or {}).get("mean")  # noqa: E731
                s = lambda k: (b.get(k) or {}).get("std")  # noqa: E731
                filas_modelo.append({
                    "hardware": et, "modo_gpu": modo, "modelo": modelo,
                    "concurrencia": b.get("concurrency", 1), "contexto": b.get("context_size", 0),
                    "prompt": b.get("prompt_size"), "generados": b.get("response_size"),
                    "pp_ts": m("pp_throughput"), "pp_std": s("pp_throughput"),
                    "tg_ts": m("tg_throughput"), "tg_std": s("tg_throughput"),
                    "tg_req_ts": m("tg_req_throughput"), "pico_ts": m("peak_throughput"),
                    "ttft_ms": m("e2e_ttft"), "ttft_std": s("e2e_ttft"),
                })
            filas[(et, modelo, modo)] = filas_modelo
            metas[(et, modelo, modo)] = meta
    todas = [f for lista in filas.values() for f in lista]
    varios_modos = len({f["modo_gpu"] for f in todas}) > 1
    for f in todas:
        meta = metas[(f["hardware"], f["modelo"], f["modo_gpu"])]
        f["etiqueta"] = f"{f['hardware']} · {MODOS_TXT.get(f['modo_gpu'], f['modo_gpu'])}" if varios_modos else f["hardware"]
        f["n_gpus"] = meta.get("n_gpus_usadas", 1)
        repo = meta.get("repo", "")
        f["cuantizacion"] = repo.split(":", 1)[1] if ":" in repo else "por defecto"
        f["vram_gb"] = max(0, meta.get("vram_max_mb", 0)) / 1024
        f["vram_por_gpu_mb"] = meta.get("vram_max_por_gpu_mb", "")
        f["tiempo_carga_s"] = meta.get("tiempo_carga_s")
        f["potencia_w"] = meta.get("potencia_prom_w") or None
        f["temp_max_c"] = meta.get("temp_max_c")
        f["args_extra"] = meta.get("args_extra", "")
    return sistemas, todas


# ----------------------------------------------------------------------------- utilidades
def seleccionar(filas, contexto=0, concurrencia=1, prompt=None):
    """Una fila por (etiqueta, modelo) para el escenario pedido."""
    out = {}
    for f in filas:
        if f["contexto"] == contexto and f["concurrencia"] == concurrencia and (prompt is None or f["prompt"] == prompt):
            out[(f["etiqueta"], f["modelo"])] = f
    return out


def color_et(etiquetas, et):
    return SERIES[etiquetas.index(et) % len(SERIES)]


def guardar(fig, carpeta, nombre, graficos):
    ruta = carpeta / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    graficos.append(nombre)
    print(f"  ✔ {nombre}")


def titulos(ax, titulo, subtitulo=""):
    """Título en negrita + subtítulo explicativo más chico, alineados a la izquierda."""
    ax.set_title(titulo, pad=22 if subtitulo else 8)
    if subtitulo:
        ax.annotate(subtitulo, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                    fontsize=9, color=TEXTO_2, va="bottom")


def barras_h(sel, etiquetas, campo, titulo, xlabel, carpeta, nombre, graficos,
             campo_std=None, dec=1, refs=(), subtitulo="", mayor_mejor=True):
    """Barras horizontales agrupadas: un grupo por modelo, una barra por serie (hardware o hardware · modo).
    refs: líneas verticales de referencia [(valor, texto), ...]"""
    modelos = sorted({m for (_, m) in sel})
    def clave(m):
        v = next((sel[(e, m)][campo] for e in etiquetas if (e, m) in sel and sel[(e, m)][campo] is not None), 0)
        return v if mayor_mejor else -v
    modelos.sort(key=clave)  # el mejor queda arriba
    if not modelos:
        return
    n_et = len(etiquetas)
    alto = 0.8 / n_et
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(modelos) * max(1, n_et * 0.7) + 1.6))
    vmax = max((f[campo] or 0) for f in sel.values()) or 1
    for i, et in enumerate(etiquetas):
        for j, m in enumerate(modelos):
            f = sel.get((et, m))
            if not f or f[campo] is None:
                continue
            y = j + (n_et - 1 - i - (n_et - 1) / 2) * alto
            v = f[campo]
            ax.barh(y, v, height=alto * 0.92, color=color_et(etiquetas, et),
                    label=et if j == 0 or (et, modelos[0]) not in sel else None)
            if campo_std and f.get(campo_std):
                ax.errorbar(v, y, xerr=f[campo_std], fmt="none", ecolor=TEXTO_2, elinewidth=1, capsize=2)
            ax.text(v + vmax * 0.01, y, num(v, dec), va="center", fontsize=8, color=TEXTO)
    ax.set_yticks(range(len(modelos)))
    ax.set_yticklabels(modelos)
    ax.grid(axis="y", visible=False)
    nota = "Las líneas finas sobre cada barra indican ± desviación estándar entre repeticiones." if campo_std else ""
    ax.set_xlabel(xlabel + (f"\n{nota}" if nota else ""))
    ax.set_xlim(0, max([vmax] + [r for r, _ in refs]) * 1.15)
    for r, txt in refs:
        ax.axvline(r, color=SERIES[7], linewidth=1.5, linestyle="--")
        ax.text(r, len(modelos) - 0.4, f" {txt}", color=TEXTO, fontsize=8, va="bottom")
    titulos(ax, titulo, subtitulo)
    if n_et > 1:
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend([uniq[e] for e in etiquetas if e in uniq], [e for e in etiquetas if e in uniq],
                  loc="upper left", bbox_to_anchor=(1.01, 1), title="Configuración")
    guardar(fig, carpeta, nombre, graficos)


def mapa_calor(filas, et, campo, titulo, unidad, carpeta, nombre, graficos, dec=1, invertir=False):
    datos = {(f["modelo"], f["contexto"]): f[campo] for f in filas
             if f["etiqueta"] == et and f["concurrencia"] == 1 and f["prompt"] == max_prompt(filas) and f[campo] is not None}
    if not datos:
        return
    modelos = sorted({m for m, _ in datos}, key=lambda m: -(datos.get((m, 0)) or 0) * (-1 if invertir else 1))
    ctxs = sorted({c for _, c in datos})
    z = [[datos.get((m, c), float("nan")) for c in ctxs] for m in modelos]
    vals = [v for fila in z for v in fila if not math.isnan(v)]
    vmin, vmax = min(vals), max(vals)
    fig, ax = plt.subplots(figsize=(2.2 + 1.6 * len(ctxs), 0.5 * len(modelos) + 1.8))
    im = ax.imshow(z, cmap=CMAP.reversed() if invertir else CMAP, aspect="auto", vmin=vmin, vmax=vmax)
    for i, fila in enumerate(z):
        for j, v in enumerate(fila):
            if math.isnan(v):
                continue
            t = (v - vmin) / ((vmax - vmin) or 1)
            oscuro = (1 - t) if invertir else t
            ax.text(j, i, num(v, dec), ha="center", va="center", fontsize=9,
                    color="white" if oscuro > 0.55 else TEXTO)
    ax.set_xticks(range(len(ctxs)))
    ax.set_xticklabels([ctx_txt(c) for c in ctxs])
    ax.set_yticks(range(len(modelos)))
    ax.set_yticklabels(modelos)
    ax.grid(False)
    ax.set_xlabel("Contexto previo en la conversación (tokens)")
    titulos(ax, titulo, f"Hardware: {et}")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(unidad)
    cb.outline.set_visible(False)
    guardar(fig, carpeta, nombre, graficos)


def max_prompt(filas):
    return max((f["prompt"] or 0) for f in filas)


def multiples(filas, etiquetas, eje_x, campo, titulo, ylabel, xlabel, carpeta, nombre, graficos, filtro, fmt_x=str):
    modelos = sorted({f["modelo"] for f in filas if filtro(f)})
    if not modelos:
        return
    cols = min(4, len(modelos))
    filas_g = math.ceil(len(modelos) / cols)
    fig, axes = plt.subplots(filas_g, cols, figsize=(3.4 * cols, 2.7 * filas_g + 0.8), squeeze=False)
    for k, m in enumerate(modelos):
        ax = axes[k // cols][k % cols]
        for et in etiquetas:
            pts = sorted((f[eje_x], f[campo]) for f in filas
                         if f["modelo"] == m and f["etiqueta"] == et and filtro(f) and f[campo] is not None)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(range(len(xs)) if eje_x == "contexto" else xs, ys, color=color_et(etiquetas, et),
                    linewidth=2, marker="o", markersize=5, label=et)
            if eje_x == "contexto":
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels([fmt_x(x) for x in xs])
            else:
                ax.set_xticks(xs)
        ax.set_title(m, fontsize=10, loc="left")
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
    for k in range(len(modelos), filas_g * cols):
        axes[k // cols][k % cols].set_visible(False)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXTO)
    fig.supylabel(ylabel, fontsize=10, color=TEXTO_2)
    fig.supxlabel(xlabel, fontsize=10, color=TEXTO_2)
    if len(etiquetas) > 1:
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), title="Configuración")
    fig.tight_layout()
    guardar(fig, carpeta, nombre, graficos)


def pct_vs(sel, serie, base, campo):
    """{modelo: % de diferencia de 'serie' frente a 'base'} para un escenario ya seleccionado."""
    out = {}
    for (e, m), f in sel.items():
        b = sel.get((base, m))
        if e == serie and b and b[campo] and f[campo]:
            out[m] = round(100 * (f[campo] / b[campo] - 1), 1) + 0.0
    return out


def graficar_difs(paneles, orden_series, titulo, subtitulo, xlabel, carpeta, nombre, graficos,
                  titulo_leyenda="Configuración"):
    """Barras divergentes agrupadas. paneles = [(subtítulo, {serie: {modelo: pct}}), ...]"""
    series = [s for s in orden_series if any(d.get(s) for _, d in paneles)]
    modelos = sorted({m for _, d in paneles for s in series for m in d.get(s, {})})
    if not modelos:
        return
    primera = paneles[0][1].get(series[0], {})
    modelos.sort(key=lambda m: primera.get(m, 0))
    n = len(series)
    alto = 0.8 / n
    fig, axes = plt.subplots(1, len(paneles), sharey=True, squeeze=False,
                             figsize=(6.5 * len(paneles) + 1.5, 0.5 * len(modelos) * max(1, n * 0.6) + 2))
    for ax, (sub, difs) in zip(axes[0], paneles):
        rango = max([abs(v) for s in series for v in difs.get(s, {}).values()] + [1])
        for i, s in enumerate(series):
            etiquetado = False
            for j, m in enumerate(modelos):
                v = difs.get(s, {}).get(m)
                if v is None:
                    continue
                y = j + (n - 1 - i - (n - 1) / 2) * alto
                ax.barh(y, v, height=alto * 0.92, color=SERIES[orden_series.index(s) % len(SERIES)],
                        label=None if etiquetado else s)
                etiquetado = True
                ax.text(v + (rango * 0.02 if v >= 0 else -rango * 0.02), y, f"{'+' if v > 0 else ''}{num(v)} %",
                        va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=TEXTO)
        ax.axvline(0, color=TEXTO_2, linewidth=1)
        ax.set_xlim(-rango * 1.3, rango * 1.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel)
        ax.set_title(sub, fontsize=11)
    axes[0][0].set_yticks(range(len(modelos)))
    axes[0][0].set_yticklabels(modelos)
    fig.suptitle(titulo, x=0.01, y=0.995, ha="left", fontsize=13, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.995, "\n\n" + subtitulo, fontsize=9, color=TEXTO_2, va="top")
    h, l = [], []
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.9), title=titulo_leyenda)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    guardar(fig, carpeta, nombre, graficos)


# ----------------------------------------------------------------------------- gráficos
def generar(sistemas, filas, etiquetas, carpeta):
    g = []
    pmax = max_prompt(filas)
    sel0 = seleccionar(filas, 0, 1, pmax)
    ctx_max = max(f["contexto"] for f in filas)
    sel_max = seleccionar(filas, ctx_max, 1, pmax)
    dual = any(f["n_gpus"] > 1 for f in filas)
    refs_vram = [(VRAM_RTX3090_GB, "1x RTX 3090 (24 GB)")] + ([(2 * VRAM_RTX3090_GB, "2x RTX 3090 (48 GB)")] if dual else [])

    barras_h(sel0, etiquetas, "tg_ts", "Velocidad de generación",
             "Tokens generados por segundo (más es mejor)", carpeta, "01_velocidad_generacion.png", g,
             campo_std="tg_std", subtitulo="Sin contexto previo, 1 usuario. Es la velocidad a la que \"escribe\" el modelo.")
    barras_h(sel0, etiquetas, "pp_ts", "Velocidad de lectura del prompt (prefill)",
             "Tokens de prompt procesados por segundo (más es mejor)", carpeta, "02_velocidad_prompt.png", g,
             campo_std="pp_std", dec=0, subtitulo=f"Prompt de {num(pmax, 0)} tokens, sin contexto previo.")
    if ctx_max > 0:
        barras_h(sel_max, etiquetas, "tg_ts", f"Velocidad de generación con {ctx_txt(ctx_max)} de contexto",
                 "Tokens generados por segundo (más es mejor)", carpeta, "03_generacion_contexto_largo.png", g,
                 campo_std="tg_std", subtitulo="Simula una conversación larga o un documento grande ya cargado.")

    for i, et in enumerate(etiquetas):
        sufijo = f"_{i + 1}" if len(etiquetas) > 1 else ""
        mapa_calor(filas, et, "tg_ts", "Generación (t/s) según el contexto acumulado", "tokens/s",
                   carpeta, f"04_mapa_generacion{sufijo}.png", g)
        filas_ttft = [dict(f, ttft_s=(f["ttft_ms"] or 0) / 1000) for f in filas]
        mapa_calor(filas_ttft, et, "ttft_s", "Tiempo hasta el primer token (s)", "segundos (menos es mejor)",
                   carpeta, f"05_mapa_ttft{sufijo}.png", g, dec=2, invertir=True)

    # Degradación relativa: % de la velocidad sin contexto que se conserva
    rel = []
    for f in filas:
        b = sel0.get((f["etiqueta"], f["modelo"]))
        if b and b["tg_ts"] and f["tg_ts"] and f["concurrencia"] == 1 and f["prompt"] == pmax:
            rel.append(dict(f, tg_rel=100 * f["tg_ts"] / b["tg_ts"]))
    if ctx_max > 0:
        multiples(rel, etiquetas, "contexto", "tg_rel", "¿Cuánto se frena cada modelo al llenar el contexto?",
                  "% de la velocidad sin contexto", "Contexto previo", carpeta, "06_degradacion_contexto.png", g,
                  filtro=lambda f: True, fmt_x=ctx_txt)

    barras_h(sel0, etiquetas, "vram_gb", "Memoria de video utilizada (VRAM)",
             "GB de VRAM máximos durante la prueba (incluye caché KV y lo que ya usaba el sistema)",
             carpeta, "07_vram.png", g, dec=1, refs=refs_vram, mayor_mejor=False,
             subtitulo="Con 2 GPUs es la suma de ambas." if dual else "")

    efic = {k: dict(f, efic=(f["tg_ts"] / f["potencia_w"]) if f["potencia_w"] else None) for k, f in sel0.items()}
    if any(f["efic"] for f in efic.values()):
        barras_h(efic, etiquetas, "efic", "Eficiencia energética",
                 "Tokens por julio = (tokens/s) ÷ vatios promedio de la GPU (más es mejor)",
                 carpeta, "08_eficiencia.png", g, dec=3,
                 subtitulo="Aproximado: la potencia es el promedio de toda la prueba del modelo.")

    barras_h(sel0, etiquetas, "tiempo_carga_s", "Tiempo de carga del modelo",
             "Segundos hasta que el servidor responde (menos es mejor; la 1ª vez incluye la descarga)",
             carpeta, "09_tiempo_carga.png", g, dec=1, mayor_mejor=False)

    # Dispersión: VRAM vs velocidad
    fig, ax = plt.subplots(figsize=(10, 6.5))
    rotulados = set()
    for et in etiquetas:
        pts = [f for (e, _), f in sel0.items() if e == et and f["tg_ts"]]
        ax.scatter([f["vram_gb"] for f in pts], [f["tg_ts"] for f in pts], s=70, color=color_et(etiquetas, et),
                   edgecolor=SUPERFICIE, linewidth=2, label=et, zorder=3)
        for f in pts:
            if f["modelo"] not in rotulados:
                rotulados.add(f["modelo"])
                ax.annotate(f["modelo"], (f["vram_gb"], f["tg_ts"]), xytext=(6, 4), textcoords="offset points",
                            fontsize=8, color=TEXTO)
    for r, _ in refs_vram:
        ax.axvline(r, color=SERIES[7], linewidth=1.5, linestyle="--")
        ax.text(r, ax.get_ylim()[1], f" {r} GB", color=TEXTO, fontsize=8, va="top")
    ax.set_xlabel("VRAM utilizada (GB)")
    ax.set_ylabel("Tokens generados por segundo")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    titulos(ax, "Tamaño en memoria vs. velocidad", "Arriba a la izquierda = rápido y liviano")
    if len(etiquetas) > 1:
        ax.legend(title="Configuración", loc="upper left", bbox_to_anchor=(1.01, 1))
    guardar(fig, carpeta, "10_vram_vs_velocidad.png", g)

    # Concurrencia (solo si se midió más de un nivel)
    if len({f["concurrencia"] for f in filas}) > 1:
        multiples(filas, etiquetas, "concurrencia", "tg_ts", "Rendimiento total con varios usuarios simultáneos",
                  "Tokens/s totales", "Usuarios simultáneos", carpeta, "11_concurrencia.png", g,
                  filtro=lambda f: f["contexto"] == 0 and f["prompt"] == pmax)

    # Comparativa de todas las series contra la base (hardware y/o modo de GPU)
    if len(etiquetas) > 1:
        base = etiquetas[0]
        for campo, nom, archivo in (("tg_ts", "generación", "12_comparativa_generacion.png"),
                                    ("pp_ts", "lectura de prompt", "13_comparativa_prompt.png")):
            difs = {et: pct_vs(sel0, et, base, campo) for et in etiquetas[1:]}
            graficar_difs([(f"Velocidad de {nom}", difs)], etiquetas, f"Comparativa vs. {base}: {nom}",
                          "Diferencia porcentual frente a la configuración base. Barras a la derecha = más rápido.",
                          f"Diferencia vs. {base} (%)", carpeta, archivo, g)

    # Efecto de NVLink: misma máquina y mismo modo de GPU, con vs. sin NVLink
    nv = {hw: sistemas.get(hw, {}).get("nvlink") for hw in {f["hardware"] for f in filas}}
    con = sorted(h for h, v in nv.items() if v == "si")
    sin = sorted(h for h, v in nv.items() if v == "no")
    if con and sin:
        hc, hs = con[0], sin[0]
        if len(con) > 1 or len(sin) > 1:
            print(f"  Aviso: varias corridas con/sin NVLink; se compara {hc} vs. {hs}")
        por_modo = defaultdict(dict)
        for f in filas:
            if f["contexto"] == 0 and f["concurrencia"] == 1 and f["prompt"] == pmax:
                por_modo[(f["hardware"], f["modo_gpu"])][f["modelo"]] = f
        modos = [m for m in ORDEN_MODOS if (hc, m) in por_modo and (hs, m) in por_modo]
        paneles = []
        for campo, nom in (("tg_ts", "Generación"), ("pp_ts", "Lectura de prompt")):
            difs = {}
            for m in modos:
                a, b = por_modo[(hc, m)], por_modo[(hs, m)]
                difs[MODOS_TXT[m]] = {mod: round(100 * (a[mod][campo] / b[mod][campo] - 1), 1) + 0.0
                                      for mod in a if mod in b and a[mod][campo] and b[mod][campo]}
            paneles.append((nom, difs))
        if modos:
            graficar_difs(paneles, [MODOS_TXT[m] for m in ORDEN_MODOS], f"Efecto de NVLink: {hc} vs. {hs}",
                          "Cuánto más rápido es cada modo con NVLink. \"1 GPU\" es el control: debería dar ~0 %.",
                          "Diferencia con NVLink (%)", carpeta, "14_efecto_nvlink.png", g, titulo_leyenda="Modo de GPU")
    return g


# ----------------------------------------------------------------------------- tablas / informe
COLUMNAS = ["etiqueta", "hardware", "modo_gpu", "n_gpus", "modelo", "cuantizacion", "concurrencia", "contexto", "prompt", "generados",
            "pp_ts", "pp_std", "tg_ts", "tg_std", "tg_req_ts", "pico_ts", "ttft_ms", "ttft_std",
            "vram_gb", "vram_por_gpu_mb", "tiempo_carga_s", "potencia_w", "temp_max_c", "args_extra"]


def escribir_csv(filas, carpeta):
    filas = [{k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()} for f in filas]
    with open(carpeta / "resumen.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, COLUMNAS, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    with open(carpeta / "resumen_excel.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(COLUMNAS)
        for r in filas:
            w.writerow([str(r.get(c, "")).replace(".", ",") if isinstance(r.get(c), float) else r.get(c, "")
                        for c in COLUMNAS])


def informe_html(sistemas, filas, etiquetas, graficos, carpeta):
    e = html.escape
    pmax = max_prompt(filas)
    sel0 = seleccionar(filas, 0, 1, pmax)
    ctx_max = max(f["contexto"] for f in filas)
    selm = seleccionar(filas, ctx_max, 1, pmax)

    hws = list(dict.fromkeys(f["hardware"] for f in sorted(filas, key=lambda f: etiquetas.index(f["etiqueta"]))))
    t_sis = "".join(
        f"<tr><td><b>{e(hw)}</b></td><td>{e(s.get('cpu', ''))}</td><td>{s.get('nucleos', '')} / {s.get('hilos', '')}</td>"
        f"<td>{num(s.get('ram_gb'), 0)} GB · {e(str(s.get('ram_velocidad', '')))}</td>"
        f"<td>{s.get('n_gpus', 1)}x {e(s.get('gpu', ''))}</td><td>{e(str(s.get('nvlink', 'no aplica')))}"
        f"{' (' + e(str(s.get('topologia'))) + ')' if s.get('topologia', '-') != '-' else ''}</td>"
        f"<td>{e(s.get('so', ''))}</td><td>{e(s.get('llama_cpp', ''))}</td></tr>"
        for hw, s in ((hw, sistemas.get(hw, {})) for hw in hws))

    filas_t = sorted(sel0.values(), key=lambda f: (etiquetas.index(f["etiqueta"]), -(f["tg_ts"] or 0)))
    t_res = ""
    for f in filas_t:
        fm = selm.get((f["etiqueta"], f["modelo"]), {})
        ef = f["tg_ts"] / f["potencia_w"] if f["potencia_w"] and f["tg_ts"] else None
        t_res += (f"<tr><td>{e(f['etiqueta'])}</td><td><b>{e(f['modelo'])}</b></td><td>{e(f['cuantizacion'])}</td>"
                  f"<td class=n>{num(f['tg_ts'])}</td><td class=n>{num(fm.get('tg_ts'))}</td>"
                  f"<td class=n>{num(f['pp_ts'], 0)}</td><td class=n>{num((fm.get('ttft_ms') or 0) / 1000, 2)}</td>"
                  f"<td class=n>{num(f['vram_gb'])}</td><td class=n>{num(ef, 3)}</td>"
                  f"<td class=n>{num(f['tiempo_carga_s'])}</td></tr>")

    imgs = "".join(f'<figure><img src="{e(n)}" alt="{e(n)}" loading="lazy"></figure>' for n in graficos)
    doc = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Informe de benchmark LLM</title>
<style>
:root{{color-scheme:light;--bg:#fcfcfb;--tx:#0b0b0b;--tx2:#52514e;--ln:#e4e3df;--ac:#2a78d6}}
body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 16px}}
h1{{font-size:28px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 10px}} p.s{{color:var(--tx2);margin:0}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--ln);text-align:left;white-space:nowrap}}
th{{color:var(--tx2);font-weight:600}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
figure{{margin:24px 0}} img{{max-width:100%;height:auto;border:1px solid var(--ln);border-radius:8px}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;color:var(--tx2);font-size:13px}} dt{{font-weight:600;color:var(--tx)}}
</style></head><body><main>
<h1>Informe de benchmark de LLMs locales</h1>
<p class=s>llama.cpp (llama-server) + llama-benchy · prompt de {num(pmax, 0)} tokens · contextos medidos:
{", ".join(ctx_txt(c) for c in sorted({f['contexto'] for f in filas}))}</p>
<h2>Equipos evaluados</h2>
<div class=tw><table><tr><th>Etiqueta</th><th>CPU</th><th>Núcleos / hilos</th><th>RAM</th><th>GPU</th><th>NVLink</th><th>Sistema</th><th>llama.cpp</th></tr>{t_sis}</table></div>
<h2>Resumen</h2>
<div class=tw><table><tr><th>Configuración</th><th>Modelo</th><th>Cuant.</th><th>Gen. t/s</th><th>Gen. t/s @{ctx_txt(ctx_max)}</th>
<th>Prompt t/s</th><th>1er token @{ctx_txt(ctx_max)} (s)</th><th>VRAM GB</th><th>Tokens/julio</th><th>Carga (s)</th></tr>{t_res}</table></div>
<h2>Cómo leer las métricas</h2>
<dl><dt>Gen. t/s (tg)</dt><dd>Velocidad a la que el modelo escribe la respuesta. ~10 t/s ya se lee cómodo; &gt;30 t/s es fluido.</dd>
<dt>Prompt t/s (pp)</dt><dd>Velocidad de lectura de la entrada. Importa con documentos largos, código o RAG.</dd>
<dt>1er token (TTFT)</dt><dd>Espera hasta ver la primera palabra. Crece con el contexto acumulado.</dd>
<dt>VRAM</dt><dd>Si se acerca a 24 GB (o 48 GB con 2 GPUs), no queda lugar para más contexto. Los modelos “híbridos” usan además RAM del sistema y dependen de la CPU.</dd>
<dt>Tokens/julio</dt><dd>Eficiencia energética aproximada de la(s) GPU(s).</dd>
<dt>Modos de GPU</dt><dd><b>1 GPU</b>: referencia. <b>2 GPU capas</b>: cada GPU tiene la mitad de las capas y trabajan por turnos (compatible, poco tráfico entre GPUs). <b>2 GPU tensor</b>: ambas GPUs trabajan a la vez en cada capa (experimental, mucho tráfico entre GPUs: es donde NVLink debería ayudar).</dd></dl>
<h2>Gráficos</h2>{imgs}
</main></body></html>"""
    (carpeta / "informe.html").write_text(doc, encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    raiz = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Genera gráficos comparativos en español a partir de los resultados.")
    ap.add_argument("--resultados", type=Path, default=raiz / "resultados", help="carpeta con las corridas")
    ap.add_argument("--salida", type=Path, default=None, help="carpeta de salida (defecto: resultados/informe)")
    ap.add_argument("--base", default=None, help="serie de referencia para las comparativas (ej. '7800X3D-64GB-PCIe · 1 GPU')")
    ap.add_argument("--etiquetas", default=None, help="solo estas series, separadas por coma (en ese orden)")
    a = ap.parse_args()

    if not a.resultados.exists():
        raise SystemExit(f"No existe {a.resultados}. Ejecutá primero bench_linux.sh o bench_windows.ps1.")
    sistemas, filas = cargar(a.resultados)
    if not filas:
        raise SystemExit("No se encontraron resultados válidos (archivos *.json + *.meta.json con estado ok).")

    orden_modo = lambda m: ORDEN_MODOS.index(m) if m in ORDEN_MODOS else 99  # noqa: E731
    etiquetas = [s for *_, s in sorted({(f["hardware"], orden_modo(f["modo_gpu"]), f["etiqueta"]) for f in filas})]
    if a.etiquetas:
        pedidas = [x.strip() for x in a.etiquetas.split(",")]
        etiquetas = [x for x in pedidas if x in etiquetas]
        filas = [f for f in filas if f["etiqueta"] in etiquetas]
    if a.base:
        if a.base not in etiquetas:
            raise SystemExit(f"La serie base '{a.base}' no existe. Disponibles: {', '.join(etiquetas)}")
        etiquetas.remove(a.base)
        etiquetas.insert(0, a.base)
    if len(etiquetas) > len(SERIES):
        print(f"Aviso: más de {len(SERIES)} series; los colores se repetirán. Usá --etiquetas para filtrar.")

    salida = a.salida or (a.resultados / "informe")
    salida.mkdir(parents=True, exist_ok=True)
    print(f"Series: {', '.join(etiquetas)}  ·  modelos: {len({f['modelo'] for f in filas})}  ·  filas: {len(filas)}")
    graficos = generar(sistemas, filas, etiquetas, salida)
    escribir_csv(filas, salida)
    informe_html(sistemas, filas, etiquetas, graficos, salida)
    print(f"\nInforme: {salida / 'informe.html'}")


if __name__ == "__main__":
    main()
