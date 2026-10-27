"""Dashboard de gestión - Centro de datos ODATA DC BG03 (Tenjo, Cundinamarca).
Ejecutar:  pip install -r requirements.txt  y luego  streamlit run app.py   (requiere Streamlit 1.50 o superior)
Los datos son SIMULADOS con los supuestos del informe (TI promedio 60 MW).
Para usar datos reales, reemplace make_data() por la lectura de su base de series de tiempo.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ODATA DC BG03 - Centro de control", layout="wide")

TEAL, AMBER, RED, INK, GREY = "#0F766E", "#D97706", "#B91C1C", "#1E293B", "#94A3B8"
T_WARN, T_CRIT = 27.0, 32.0          # °C, entrada de aire (guías ASHRAE)
PUE_BASE = 1.6                        # diseño convencional de referencia
TI_PROM_MW = 60                       # carga de TI promedio del escenario simulado (ajústela si su informe usa otra)
HR_WARN, HR_CRIT = (30.0, 60.0), (20.0, 70.0)   # % HR: banda operativa interna y límites críticos
TRM_DEF = 3160                        # COP/USD, referencia mediados de octubre de 2026 (actualizar)

st.markdown(
    "<style>@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap');"
    "html,body,[class*='css']{font-family:'IBM Plex Sans',sans-serif;}"
    "[data-testid='stMetricValue']{font-size:1.7rem;font-weight:600;}</style>",
    unsafe_allow_html=True,
)


@st.cache_data
def make_data(days: int = 30, seed: int = 7, clima=None) -> pd.DataFrame:
    """Serie horaria sintética. Escenario: primeras fases, refrigeración sin agua (circuito cerrado) a 2.600 m s. n. m."""
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range("2026-09-26", periods=n, freq="h")
    h = idx.hour.values
    frente = np.zeros(days)                                   # frentes de lluvia: persistencia de un día a otro
    for i in range(1, days):
        frente[i] = 0.6 * frente[i - 1] + rng.normal(0, 1.3)
    outside = 14.5 + 5.5 * np.sin((h - 10) / 24 * 2 * np.pi) + np.repeat(frente, 24) + rng.normal(0, 0.6, n)
    if clima is not None:                                     # temperatura exterior real (Open-Meteo)
        outside = np.asarray(clima, dtype=float)[:n]
    it_mw = (TI_PROM_MW + np.linspace(-2.5, 2.5, n) + 2.5 * np.sin((h - 14) / 24 * 2 * np.pi)
             - 1.5 * (idx.dayofweek.values >= 5) + rng.normal(0, 0.8, n))   # rampa de clientes y ciclo semanal
    it_mw += TI_PROM_MW - it_mw.mean()
    pue = 1.22 + 0.005 * np.clip(outside - 12, 0, None) - 0.002 * (it_mw - TI_PROM_MW) + rng.normal(0, 0.008, n)
    wue = np.clip(0.04 + 0.004 * np.clip(outside - 14, 0, None) + rng.normal(0, 0.008, n), 0.01, None)
    noise = (46 + 0.12 * (it_mw - TI_PROM_MW) + 0.15 * np.clip(outside - 12, 0, None)
             + 2.2 * np.sin((h - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.6, n))
    hum = 48 - 0.35 * (outside - 14.0) + rng.normal(0, 1.2, n)
    salida = outside + 9 + 0.12 * (it_mw - TI_PROM_MW) + rng.normal(0, 0.6, n)
    df = pd.DataFrame({"hora": idx, "exterior_C": outside, "ti_MW": it_mw, "PUE": pue, "WUE_L_kWh": wue,
                       "ruido_dBA": noise, "humedad_pct": hum, "salida_C": salida})
    for room, off in zip(["Sala A", "Sala B", "Sala C", "Sala D"], [0.0, 0.6, 1.2, 0.3]):
        df[room] = 22 + off + (outside - 14) * 0.15 + rng.normal(0, 0.35, n)
    ev = (df.hora >= "2026-10-16 15:00") & (df.hora <= "2026-10-16 18:00")   # evento: falla de enfriamiento en la Sala C
    k = np.array([6, 9, 10.5, 7])[: ev.sum()]
    df.loc[ev, "Sala C"] += k
    df.loc[ev, "PUE"] += 0.004 * k                 # ventiladores al máximo
    df.loc[ev, "ruido_dBA"] += 0.4 * k
    df.loc[ev, "humedad_pct"] -= 0.7 * k
    hu = (df.hora >= "2026-10-21 02:00") & (df.hora <= "2026-10-21 04:00")   # evento: humedad alta por 3 horas
    df.loc[hu, "humedad_pct"] += np.array([13, 15, 12])[: hu.sum()]
    df["caida_min"] = 0.0                          # minutos sin servicio: 4,2 min en el mes (99,990 %)
    df.loc[df.hora == "2026-10-08 03:00", "caida_min"] = 2.7
    df.loc[df.hora == "2026-10-22 14:00", "caida_min"] = 1.5
    for c, dec in [("exterior_C", 2), ("ti_MW", 2), ("PUE", 3), ("WUE_L_kWh", 3), ("ruido_dBA", 1),
                   ("humedad_pct", 1), ("salida_C", 1), ("Sala A", 2), ("Sala B", 2), ("Sala C", 2), ("Sala D", 2)]:
        df[c] = df[c].round(dec)
    df["energia_total_MWh"] = df.ti_MW * df.PUE
    df["agua_m3"] = df.ti_MW * df.WUE_L_kWh
    return df


@st.cache_data(show_spinner="Consultando el clima de Tenjo...")
def clima_tenjo(inicio: str, fin: str):
    """Temperatura horaria real de Tenjo desde Open-Meteo (API gratuita, sin clave)."""
    import json
    import urllib.request
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=4.917&longitude=-74.167"
           f"&start_date={inicio}&end_date={fin}&hourly=temperature_2m&timezone=America%2FBogota")
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.load(r)
    s = pd.Series(data["hourly"]["temperature_2m"], dtype=float)
    faltantes = int(s.isna().sum())
    return s.interpolate().ffill().bfill().to_numpy(), faltantes


def estado(t: float) -> str:
    return "🔴 Crítico" if t >= T_CRIT else ("🟡 Advertencia" if t >= T_WARN else "🟢 Normal")


def lineas(fig, y, color, texto):
    fig.add_hline(y=y, line_dash="dash", line_color=color, annotation_text=texto,
                  annotation_position="top left")


def estilo(fig, h=340, titulo=None):
    fig.update_layout(height=h, title=titulo, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2), plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig


COLS_REQ = ["hora", "ti_MW", "PUE", "WUE_L_kWh"]
st.sidebar.header("Fuente de datos")
archivo = st.sidebar.file_uploader("Cargar CSV con datos reales (por hora)", type="csv")
usar_clima = st.sidebar.checkbox("Usar clima real de Tenjo (Open-Meteo)",
                                 help="Reemplaza la temperatura exterior simulada. Requiere internet.")
clima = None
if usar_clima:
    try:
        clima, faltantes = clima_tenjo("2026-09-26", "2026-10-25")
        if len(clima) != 720:
            raise ValueError("la API devolvió un número inesperado de horas")
        if faltantes:
            st.sidebar.info(f"{faltantes} horas recientes aún no estaban publicadas y se interpolaron.")
    except Exception as e:
        st.sidebar.warning(f"No se pudo cargar el clima ({e}). Se usa el simulado.")
        clima = None

origen = "simulados"
df_all = make_data(clima=clima)
if archivo is not None:
    try:
        tmp = pd.read_csv(archivo, parse_dates=["hora"])
        faltan = [c for c in COLS_REQ if c not in tmp.columns]
        if faltan:
            raise ValueError("faltan columnas: " + ", ".join(faltan))
        if not any(c.startswith("Sala") for c in tmp.columns):
            raise ValueError("incluya al menos una columna de temperatura llamada 'Sala ...'")
        for c in ["exterior_C", "ruido_dBA", "humedad_pct", "salida_C"]:
            if c not in tmp.columns:
                tmp[c] = np.nan
        tmp["energia_total_MWh"] = tmp.ti_MW * tmp.PUE        # datos horarios: MW x 1 h
        tmp["agua_m3"] = tmp.ti_MW * tmp.WUE_L_kWh
        df_all, origen = tmp.sort_values("hora").reset_index(drop=True), "cargados"
    except Exception as e:
        st.sidebar.error(f"No se pudo leer el CSV: {e}. Se muestran datos simulados.")
plantilla = make_data().head(48)[["hora", "exterior_C", "ti_MW", "PUE", "WUE_L_kWh", "ruido_dBA", "humedad_pct",
                                  "salida_C", "Sala A", "Sala B", "Sala C", "Sala D", "caida_min"]]
st.sidebar.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"),
                           "plantilla_datos.csv", "text/csv")
ROOMS = [c for c in df_all.columns if c.startswith("Sala")]
it_energy = df_all.ti_MW.mean() * 8760                 # MWh/año de TI (medido o simulado)

clase = st.sidebar.radio("Clase de equipo (umbrales de temperatura)",
                         ["A1 a A4 (refrigeración por aire)", "H1 (alta densidad)"])
if clase.startswith("A1"):
    T_WARN, T_CRIT = 27.0, 32.0     # recomendado máx. 27 °C; permitido A1 máx. 32 °C
else:
    T_WARN, T_CRIT = 22.0, 25.0     # H1: recomendado máx. 22 °C; permitido máx. 25 °C
st.sidebar.caption(f"Alertas: advertencia desde {T_WARN:.0f} °C, crítico desde {T_CRIT:.0f} °C (ASHRAE, 2021).")

# ---------------- Barra lateral ----------------
st.sidebar.header("Filtros y supuestos")
d0, d1 = df_all.hora.min().date(), df_all.hora.max().date()
rango = st.sidebar.date_input("Periodo", (d0, d1), min_value=d0, max_value=d1)
if len(rango) == 2:
    df = df_all[(df_all.hora.dt.date >= rango[0]) & (df_all.hora.dt.date <= rango[1])]
else:
    df = df_all
salas = st.sidebar.multiselect("Salas", ROOMS, default=ROOMS)
precio = st.sidebar.slider("Precio de energía (US$/MWh)", 50, 320, 98, 2,
                           help="98 ≈ contrato industrial (jun. 2026). ~299 ≈ precio de bolsa (ago. 2026).")
ref = st.sidebar.selectbox("PUE de referencia (con qué se compara)",
                           ["1,60 - diseño convencional", "1,54 - promedio global (Uptime 2025)",
                            "1,40 - hiperescala nuevo"], index=1,
                           help="Define el ahorro atribuible a las medidas de eficiencia. Cambia mucho el PRI.")
PUE_BASE = float(ref.split(" ")[0].replace(",", "."))
trm = st.sidebar.number_input("TRM (COP por USD)", value=TRM_DEF, step=10)
st.sidebar.caption(f"US$ {precio}/MWh equivale a unos {precio * trm / 1000:,.0f} COP/kWh.")
pue_op = st.sidebar.slider("PUE de operación para el PRI", 1.15, 1.60, 1.30, 0.01)
inv_efic = st.sidebar.number_input("Sobrecosto de eficiencia (US$ M)", value=45.0, step=1.0)
inv_erp = st.sidebar.number_input("Implementación ERP (US$ M)", value=1.5, step=0.1)
ahorro_mant = st.sidebar.number_input("Ahorro mantenimiento e inventario (US$ M/año)", value=1.0, step=0.1)
costo_erp = st.sidebar.number_input("Licencias y soporte ERP (US$ M/año)", value=0.3, step=0.1)
st.sidebar.download_button("Descargar datos (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "odata_dc_bg03_datos.csv", "text/csv")
st.sidebar.caption("Datos simulados con los supuestos del informe. Sustituya por mediciones reales de los sensores IoT.")

# ---------------- Cálculos ----------------
pue_med = df.energia_total_MWh.sum() / df.ti_MW.sum()
wue_med = df.agua_m3.sum() / df.ti_MW.sum()
agua_evap = df.ti_MW.sum() * 1.8                      # m3 si fuera evaporativo (1,8 L/kWh)
ahorro_agua = 1 - df.agua_m3.sum() / agua_evap
cols_t = salas or ["Sala A"]
t_max = df[cols_t].max().max()
ultima = df.iloc[-1]
costo_periodo = df.energia_total_MWh.sum() * precio / 1e6
if "caida_min" in df.columns:
    disp = 100 - df.caida_min.sum() / (len(df) / 24 * 1440) * 100
elif origen == "simulados":
    disp = 100 - 4.2 / (len(df) / 24 * 1440) * 100     # 4,2 min de caída simulada
else:
    disp = None                                          # el CSV no trae minutos de caída
invest = inv_efic + inv_erp
ahorro_energia = (PUE_BASE - pue_op) * it_energy * precio / 1e6
neto = ahorro_energia + ahorro_mant - costo_erp
pri = invest / neto if neto > 0 else float("inf")

# ---------------- Encabezado ----------------
st.title("ODATA DC BG03 - Centro de control")
st.caption("Zona Franca Metropolitana de Tenjo, hasta 120 MW de capacidad de TI (según ODATA). "
           "Las 65 ha de la presentación están por verificar. Indicadores operativos, ambientales, financieros y de gestión.")
if origen == "simulados":
    st.info("Datos simulados con los supuestos del informe. Cargue un CSV en la barra lateral para usar mediciones reales. "
            "Los bloques de residuos electrónicos, cumplimiento y avance son de ejemplo y se editan en el código.")
else:
    st.success("Mostrando datos cargados desde su CSV. Residuos, cumplimiento y avance siguen siendo de ejemplo.")

# ---------------- Semáforo global (lectura rápida de alertas) ----------------
tmax_hora = df[cols_t].max(axis=1)
h_crit = int((tmax_hora >= T_CRIT).sum())
h_warn = int(((tmax_hora >= T_WARN) & (tmax_hora < T_CRIT)).sum())
sala_pico = ultima[cols_t].astype(float).idxmax()
t_ahora = float(ultima[sala_pico])
txt = (f"Última lectura: {sala_pico} marca {t_ahora:.1f} °C. "
       f"En el periodo hubo {h_crit} h en estado crítico y {h_warn} h en advertencia.")
if df["humedad_pct"].notna().any():
    h_hum = int(((df.humedad_pct < HR_WARN[0]) | (df.humedad_pct > HR_WARN[1])).sum())
    txt += f" Humedad fuera de {HR_WARN[0]:.0f}-{HR_WARN[1]:.0f} % HR: {h_hum} h."
if t_ahora >= T_CRIT:
    st.error(f"🔴 Crítico. {txt} Escale al jefe de turno.")
elif t_ahora >= T_WARN:
    st.warning(f"🟡 Advertencia. {txt} Revise el free cooling y el flujo de aire.")
else:
    st.success(f"🟢 Normal. {txt}")

(t_res, t_op, t_amb, t_fin, t_erp, t_soc, t_ges, t_iot, t_ctx) = st.tabs(
    ["Resumen", "Operación", "Ambiente", "Finanzas", "ERP y mantenimiento", "Social y SST",
     "Gestión y cumplimiento", "Arquitectura IoT", "Contexto y fuentes"])

# ---------------- Resumen ----------------
with t_res:
    c = st.columns(6)
    c[0].metric("PUE medio", f"{pue_med:.2f}", f"{pue_med - PUE_BASE:+.2f} vs referencia {PUE_BASE:.2f}", delta_color="inverse")
    c[1].metric("WUE medio", f"{wue_med:.2f} L/kWh", f"{ahorro_agua:.0%} menos agua que evaporativo", delta_color="off")
    c[2].metric("Disponibilidad", f"{disp:.3f} %" if disp is not None else "n/d",
                "agregue la columna caida_min" if disp is None else None, delta_color="off")
    c[3].metric("Temp. máx. de entrada", f"{t_max:.1f} °C", estado(t_max), delta_color="off")
    c[4].metric("Costo de energía", f"US$ {costo_periodo:,.1f} M", "periodo seleccionado", delta_color="off")
    c[5].metric("PRI simple", f"{pri:.1f} años" if np.isfinite(pri) else "sin retorno")

    def _v(col, fn, fmt, unidad):
        x = getattr(df[col], fn)() if df[col].notna().any() else np.nan
        return "n/d" if pd.isna(x) else f"{x:{fmt}} {unidad}"
    c2 = st.columns(4)
    c2[0].metric("Humedad relativa (última)",
                 "n/d" if pd.isna(ultima["humedad_pct"]) else f"{ultima['humedad_pct']:.0f} %",
                 f"banda {HR_WARN[0]:.0f}-{HR_WARN[1]:.0f} %", delta_color="off")
    c2[1].metric("Descarga de aire (máx.)", _v("salida_C", "max", ".1f", "°C"), "aire que sale al ambiente", delta_color="off")
    c2[2].metric("Ruido perimetral (máx.)", _v("ruido_dBA", "max", ".1f", "dB(A)"), "umbral interno 55 dB(A)", delta_color="off")
    c2[3].metric("Energía total del periodo", f"{df.energia_total_MWh.sum():,.0f} MWh", "TI + enfriamiento", delta_color="off")

    st.subheader("Estado actual por sala")
    est = pd.DataFrame({"Sala": cols_t,
                        "Temp. entrada (°C)": [round(ultima[s], 1) for s in cols_t],
                        "Estado": [estado(ultima[s]) for s in cols_t]})
    st.dataframe(est, hide_index=True, width="stretch")

    st.subheader("Alertas del periodo")
    al = []
    for s in cols_t:
        for _, r in df[df[s] >= T_WARN].iterrows():
            al.append({"Hora": r.hora, "Sala": s, "Temp. (°C)": round(r[s], 1), "Nivel": estado(r[s]),
                       "Acción": "Revisar free cooling y flujo de aire" if r[s] < T_CRIT else "Escalar al jefe de turno"})
    if al:
        tabla = pd.DataFrame(al).sort_values("Temp. (°C)", ascending=False)
        st.caption(f"{len(tabla)} lecturas sobre el umbral; se muestran las 200 más altas.")
        st.dataframe(tabla.head(200), hide_index=True, width="stretch")
    else:
        st.success("Sin alertas en el periodo y las salas seleccionadas.")

# ---------------- Operación ----------------
with t_op:
    f = go.Figure()
    for s in cols_t:
        f.add_trace(go.Scatter(x=df.hora, y=df[s], name=s, mode="lines"))
    lineas(f, T_WARN, AMBER, f"Advertencia {T_WARN:.0f} °C")
    lineas(f, T_CRIT, RED, f"Crítico {T_CRIT:.0f} °C")
    st.plotly_chart(estilo(f, 360, "Temperatura de entrada de aire por sala (°C)"), width="stretch")
    zmin, zmax = T_WARN - 6, T_CRIT + 3
    a_, b_ = (T_WARN - zmin) / (zmax - zmin), (T_CRIT - zmin) / (zmax - zmin)
    hm = go.Figure(go.Heatmap(
        x=df.hora, y=cols_t, z=df[cols_t].T.values, zmin=zmin, zmax=zmax, colorbar=dict(title="°C"),
        colorscale=[[0, "#99D5CF"], [a_, "#99D5CF"], [a_, "#FBBF24"], [b_, "#FBBF24"], [b_, RED], [1, RED]],
        hovertemplate="%{y}<br>%{x}<br>%{z:.1f} °C<extra></extra>"))
    hm.update_yaxes(autorange="reversed")
    st.plotly_chart(estilo(hm, 240, "Mapa de calor por sala: verde normal, ámbar advertencia, rojo crítico"),
                    width="stretch")
    g = go.Figure(go.Scatter(x=df.hora, y=df.ti_MW, line=dict(color=TEAL), name="Carga de TI"))
    g.add_hline(y=120, line_dash="dot", line_color=GREY, annotation_text="Capacidad de TI 120 MW")
    st.plotly_chart(estilo(g, 300, "Carga de TI (MW)"), width="stretch")
    if df.humedad_pct.notna().any():
        fh = go.Figure(go.Scatter(x=df.hora, y=df.humedad_pct, line=dict(color=TEAL), name="Humedad relativa"))
        fh.add_hrect(y0=HR_WARN[0], y1=HR_WARN[1], fillcolor=TEAL, opacity=0.08, line_width=0)
        lineas(fh, HR_CRIT[1], RED, f"Crítico {HR_CRIT[1]:.0f} %")
        lineas(fh, HR_CRIT[0], RED, f"Crítico {HR_CRIT[0]:.0f} %")
        st.plotly_chart(estilo(fh, 300, f"Humedad relativa en sala (%): banda operativa {HR_WARN[0]:.0f}-{HR_WARN[1]:.0f} %"),
                        width="stretch")

# ---------------- Ambiente ----------------
with t_amb:
    d = df.set_index("hora").resample("D").agg({"energia_total_MWh": "sum", "ti_MW": "sum", "agua_m3": "sum",
                                                 "ruido_dBA": "mean", "salida_C": "max", "exterior_C": "mean"})
    d["PUE"] = d.energia_total_MWh / d.ti_MW
    a, b = st.columns(2)
    f = go.Figure(go.Scatter(x=d.index, y=d.PUE, line=dict(color=TEAL), name="PUE diario"))
    lineas(f, 1.3, AMBER, "Meta 1,3")
    a.plotly_chart(estilo(f, 300, "PUE diario"), width="stretch")
    f = go.Figure()
    f.add_bar(x=d.index, y=d.agua_m3, name="Agua medida (enfriamiento sin agua)", marker_color=TEAL)
    f.add_bar(x=d.index, y=d.ti_MW * 1.8, name="Escenario evaporativo", marker_color=GREY)
    f.update_layout(barmode="group")
    b.plotly_chart(estilo(f, 300, "Consumo de agua diario (m³)"), width="stretch")
    a, b = st.columns(2)
    f = go.Figure(go.Scatter(x=d.index, y=d.ruido_dBA, line=dict(color=INK), name="Ruido perimetral"))
    lineas(f, 55, AMBER, "Umbral interno 55 dB(A)")
    a.plotly_chart(estilo(f, 300, "Ruido perimetral (dB(A))"), width="stretch")
    ew = pd.DataFrame({"Mes": ["Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre"],
                       "Generado (t)": [3.2, 2.8, 4.1, 3.5, 3.0, 3.9],
                       "Reutilizado (%)": [22, 25, 28, 31, 35, 38],
                       "Valorizado (%)": [58, 60, 59, 61, 62, 63]})
    ew["Disposición final (%)"] = 100 - ew["Reutilizado (%)"] - ew["Valorizado (%)"]
    b.markdown("**Residuos electrónicos (datos simulados)**")
    b.dataframe(ew, hide_index=True, width="stretch")
    a, b = st.columns(2)
    if d.salida_C.notna().any():
        f = go.Figure()
        f.add_trace(go.Scatter(x=d.index, y=d.salida_C, name="Descarga (máx. diaria)", line=dict(color=AMBER)))
        f.add_trace(go.Scatter(x=d.index, y=d.exterior_C, name="Exterior (media diaria)", line=dict(color=GREY)))
        a.plotly_chart(estilo(f, 300, "Temperatura del aire de descarga (°C)"), width="stretch")
    f = go.Figure(go.Bar(x=d.index, y=d.energia_total_MWh, marker_color=TEAL, name="Energía total"))
    b.plotly_chart(estilo(f, 300, "Consumo de energía total por día (MWh)"), width="stretch")

# ---------------- Finanzas ----------------
with t_fin:
    c = st.columns(4)
    c[0].metric("Inversión adicional", f"US$ {invest:.1f} M")
    c[1].metric("Ahorro de energía", f"US$ {ahorro_energia:.2f} M/año")
    c[2].metric("Ahorro neto anual", f"US$ {neto:.2f} M/año")
    c[3].metric("PRI simple", f"{pri:.1f} años" if np.isfinite(pri) else "sin retorno")
    años = np.arange(0, 9)
    f = go.Figure(go.Scatter(x=años, y=-invest + neto * años, mode="lines+markers", line=dict(color=TEAL)))
    f.add_hline(y=0, line_color=INK)
    f.update_xaxes(title="Años", dtick=1)
    f.update_yaxes(title="Flujo acumulado (US$ M)")
    st.plotly_chart(estilo(f, 340, "Flujo de caja acumulado de las medidas de eficiencia y el ERP"),
                    width="stretch")
    esc = {"Contrato bajo": 70, "Contrato industrial (jun. 2026)": 98,
           "Contrato alto": 130, "Precio de bolsa (ago. 2026)": 299}
    sens = pd.DataFrame({"Escenario": list(esc), "Precio (US$/MWh)": list(esc.values())})
    sens["Ahorro neto (US$ M/año)"] = ((PUE_BASE - pue_op) * it_energy * sens["Precio (US$/MWh)"] / 1e6
                                       + ahorro_mant - costo_erp).round(2)
    neto_s = sens["Ahorro neto (US$ M/año)"]
    sens["PRI (años)"] = (invest / neto_s.where(neto_s > 0)).round(1)
    st.markdown("**Sensibilidad al precio de la energía**")
    st.dataframe(sens, hide_index=True, width="stretch")
    st.markdown("**PRI (años) según el PUE de referencia y el precio de la energía**")
    precios = [70, 98, 130, 299]
    filas = {}
    for base in [1.60, 1.54, 1.40]:
        fila = []
        for pr in precios:
            neto_ = max(base - pue_op, 0) * it_energy * pr / 1e6 + ahorro_mant - costo_erp
            fila.append(round(invest / neto_, 1) if neto_ > 0 else None)
        filas[f"Referencia {base:.2f}"] = fila
    st.dataframe(pd.DataFrame(filas, index=[f"US$ {p}/MWh" for p in precios]).T, width="stretch")
    st.caption("El PRI simple no descuenta el dinero en el tiempo. Complete el análisis con VPN y TIR. "
               "Un PRI muy corto con precio de bolsa refleja también una alta exposición al riesgo de precio.")

# ---------------- ERP y mantenimiento ----------------
with t_erp:
    hoy = df_all.hora.max().normalize()
    st.caption("Módulos del ERP en función: Mantenimiento de activos (PM/EAM), Inventario y Finanzas (FI). "
               "Los datos son de ejemplo: edite las tablas o reemplácelas por la exportación del ERP.")

    st.subheader("Mantenimiento preventivo de generadores (PM/EAM)")
    base_gen = pd.DataFrame({
        "Generador": ["G1", "G2", "G3", "G4", "G5", "G6"],
        "Rol": ["Activo"] * 5 + ["Reserva N+1"],
        "Horas de operación": [1840, 2210, 2475, 960, 1530, 310],
        "Último mantenimiento": [(hoy - pd.Timedelta(days=d)).date() for d in [40, 75, 92, 20, 60, 10]],
        "Intervalo (días)": [90] * 6})
    gen = st.data_editor(base_gen, hide_index=True, width="stretch", key="gen_editor",
                         disabled=["Generador", "Rol"],
                         column_config={"Último mantenimiento": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                        "Intervalo (días)": st.column_config.NumberColumn(min_value=7, step=1)})
    prox = pd.to_datetime(gen["Último mantenimiento"]) + pd.to_timedelta(gen["Intervalo (días)"], unit="D")
    gen_c = gen.assign(**{"Próximo": prox.dt.date, "Días restantes": (prox - hoy).dt.days})
    gen_c["Estado"] = np.select([gen_c["Días restantes"] < 0, gen_c["Días restantes"] <= 15],
                                ["🔴 Vencido", "🟡 Próximo (15 días)"], default="🟢 Al día")
    st.dataframe(gen_c[["Generador", "Rol", "Horas de operación", "Próximo", "Días restantes", "Estado"]],
                 hide_index=True, width="stretch")
    venc = int((gen_c["Días restantes"] < 0).sum())
    if venc:
        st.warning(f"{venc} generador(es) con mantenimiento vencido. Con redundancia N+1 el servicio se sostiene, "
                   "pero dos vencidos a la vez reducen el margen de seguridad.")

    st.subheader("Inventario de insumos")
    inv0 = pd.DataFrame({
        "Insumo": ["Diésel para generadores", "Filtros de aire", "Baterías de UPS (reposición)", "Glicol y refrigerante",
                   "Aceite lubricante", "Fibra y patch cords", "Vestuario y calzado ESD"],
        "Unidad": ["galones", "unidades", "unidades", "litros", "litros", "unidades", "juegos"],
        "Stock": [210000, 80, 24, 3500, 1200, 600, 140],
        "Punto de reorden": [200000, 100, 12, 2000, 800, 300, 60],
        "Consumo mensual": [15000, 40, 3, 150, 60, 80, 10],
        "Costo unitario (US$)": [3.2, 55, 420, 6, 9, 12, 85]})
    inv = st.data_editor(inv0, hide_index=True, width="stretch", key="inv_editor", disabled=["Insumo", "Unidad"])
    inv["Cobertura (meses)"] = (inv["Stock"] / inv["Consumo mensual"].replace(0, np.nan)).round(1)
    inv["Valor (US$)"] = (inv["Stock"] * inv["Costo unitario (US$)"]).round(0)
    inv["Estado"] = np.where(inv["Stock"] <= inv["Punto de reorden"], "🔴 Reordenar", "🟢 OK")
    st.dataframe(inv[["Insumo", "Stock", "Punto de reorden", "Cobertura (meses)", "Valor (US$)", "Estado"]],
                 hide_index=True, width="stretch")

    st.subheader("Depreciación de activos (FI)")
    st.caption("Supuestos ilustrativos, no son cifras del proyecto: reemplácelos con el libro de activos del ERP. "
               "En un modelo de co-location, los servidores de los clientes no son activos del operador. "
               "El operador deprecia sus servidores y red de gestión y, sobre todo, la infraestructura eléctrica y de enfriamiento.")
    act0 = pd.DataFrame({
        "Clase de activo": ["Servidores y red propios (gestión y monitoreo)", "UPS y baterías", "Generadores",
                            "Enfriamiento (Delta³ y free cooling)", "Obra civil y edificio"],
        "Costo (US$ M)": [12.0, 60.0, 70.0, 90.0, 250.0],
        "Vida útil (años)": [5, 10, 15, 15, 40]})
    act = st.data_editor(act0, hide_index=True, width="stretch", key="act_editor", disabled=["Clase de activo"],
                         column_config={"Vida útil (años)": st.column_config.NumberColumn(min_value=1, step=1)})
    act["Depreciación anual (US$ M)"] = (act["Costo (US$ M)"] / act["Vida útil (años)"]).round(2)
    dep_anual = float(act["Depreciación anual (US$ M)"].sum())
    años_d = np.arange(0, 11)
    f = go.Figure()
    for _, r in act.iterrows():
        f.add_bar(x=años_d, y=np.maximum(r["Costo (US$ M)"] * (1 - años_d / r["Vida útil (años)"]), 0),
                  name=r["Clase de activo"])
    f.update_layout(barmode="stack")
    f.update_xaxes(title="Años desde la puesta en marcha", dtick=1)
    f.update_yaxes(title="Valor en libros (US$ M)")
    st.plotly_chart(estilo(f, 340, "Valor en libros por clase de activo (línea recta, sin valor residual)"),
                    width="stretch")

    c = st.columns(5)
    c[0].metric("Cumplimiento del plan preventivo", f"{100 * (1 - venc / len(gen_c)):.0f} %")
    c[1].metric("Mantenimientos vencidos", f"{venc}")
    c[2].metric("Insumos por reordenar", f"{int((inv['Estado'] == '🔴 Reordenar').sum())}")
    c[3].metric("Valor del inventario", f"US$ {inv['Valor (US$)'].sum() / 1e6:.2f} M")
    c[4].metric("Depreciación anual", f"US$ {dep_anual:.1f} M", f"{dep_anual / 12:.2f} M por mes", delta_color="off")

# ---------------- Social y SST ----------------
with t_soc:
    st.caption("Metas propuestas y valores de ejemplo. Reemplácelos con los registros del área de talento humano, "
               "RSE y SST. No hay cifras oficiales de empleo del proyecto de Tenjo.")
    st.subheader("Empleo local y responsabilidad social (RSE)")
    rse = pd.DataFrame({
        "Indicador": ["Empleo local en operación (%)", "Formación técnica a la comunidad (h/año)",
                      "Cupos técnicos con colegios y SENA", "Reuniones de socialización con JAC y colectivos (por año)",
                      "Quejas comunitarias atendidas en 15 días (%)"],
        "Meta": [50, 2000, 40, 12, 90],
        "Actual": [38, 1350, 25, 5, 76]})
    rse["Avance (%)"] = (100 * rse["Actual"] / rse["Meta"]).clip(upper=100).round(0)
    st.dataframe(rse, hide_index=True, width="stretch",
                 column_config={"Avance (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")})

    st.subheader("Seguridad y salud en el trabajo (SST)")
    meses = ["May.", "Jun.", "Jul.", "Ago.", "Sep.", "Oct."]
    at = [3, 2, 2, 1, 1, 0]
    casi = [9, 8, 6, 6, 4, 3]
    trabajadores = st.number_input("Trabajadores en planta (ejemplo)", value=160, step=10)
    c = st.columns(4)
    c[0].metric("Accidentes de trabajo, último mes", f"{at[-1]}")
    c[1].metric("Tasa mensual (AT por 100 trabajadores)", f"{100 * at[-1] / trabajadores:.2f}",
                help="Indicador mínimo de frecuencia de la Resolución 0312 de 2019. Verifique la fórmula en la norma.")
    c[2].metric("Ingresos a sala con vestuario ESD verificado", "96 %", "meta 100 %", delta_color="off")
    c[3].metric("Capacitaciones en riesgo eléctrico y térmico", "88 %", "meta 100 %", delta_color="off")
    f = go.Figure()
    f.add_bar(x=meses, y=at, name="Accidentes de trabajo", marker_color=RED)
    f.add_bar(x=meses, y=casi, name="Casi-accidentes reportados", marker_color=AMBER)
    f.update_layout(barmode="group")
    st.plotly_chart(estilo(f, 300, "Accidentes y casi-accidentes por mes (datos de ejemplo)"), width="stretch")
    st.caption("Un aumento de casi-accidentes reportados suele indicar más cultura de reporte, no más riesgo. "
               "Lea las dos series juntas.")

# ---------------- Gestión y cumplimiento ----------------
with t_ges:
    cum = pd.DataFrame({
        "Requisito": ["Licencia ambiental y permisos (ANLA y CAR)", "Uso del suelo y licencia de construcción",
                      "Conexión eléctrica y RETIE", "Protección de datos (Ley 1581)",
                      "ISO/IEC 27001 y Tier III", "Gestión de RAEE", "SG-SST (Res. 0312 de 2019)",
                      "Régimen de zona franca"],
        "Estado": ["En trámite", "Pendiente", "En trámite", "Cumplido", "En trámite", "Cumplido", "En trámite", "Cumplido"],
        "Avance (%)": [60, 25, 55, 100, 40, 100, 70, 100],
        "Riesgo": ["Alto", "Alto", "Medio", "Bajo", "Medio", "Bajo", "Medio", "Bajo"]})
    st.markdown("**Cumplimiento legal (datos simulados)**")
    st.dataframe(cum, hide_index=True, width="stretch",
                 column_config={"Avance (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")})
    ph = pd.DataFrame({"Fase": ["1. Diagnóstico y marco legal", "2. Estimación técnica", "3. Modelo financiero y ERP",
                                "4. Dashboard y UI/UX", "5. Informe y sustentación"],
                       "Avance (%)": [100, 90, 70, 60, 40]})
    f = go.Figure(go.Bar(x=ph["Avance (%)"], y=ph.Fase, orientation="h", marker_color=TEAL))
    f.update_yaxes(autorange="reversed")
    f.update_xaxes(range=[0, 100])
    st.plotly_chart(estilo(f, 300, "Avance del proyecto por fase (%)"), width="stretch")
    st.metric("Avance global", f"{ph['Avance (%)'].mean():.0f} %")

# ---------------- Arquitectura IoT ----------------
with t_iot:
    st.subheader("Del sensor al dashboard")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#F1F5F9", color="#94A3B8", fontsize=11];
      subgraph cluster_ot { label="Planta física"; color="#0F766E";
        S1 [label="Temperatura y humedad\\n(pasillos, racks y descarga)"];
        S2 [label="Medidores de energía\\n(UPS, PDU, tableros)"];
        S3 [label="Caudalímetro de agua"];
        S4 [label="Sonómetro perimetral"];
        S5 [label="Generadores y enfriamiento\\n(PLC / BMS)"]; }
      GW [label="Gateway de borde\\nModbus TCP, BACnet/IP, SNMP\\na MQTT", fillcolor="#CCE7E4"];
      BR [label="Broker MQTT\\nTLS y un tema por sala", fillcolor="#CCE7E4"];
      DB [label="Base de series de tiempo\\nPostgreSQL/TimescaleDB o InfluxDB", fillcolor="#CCE7E4"];
      ETL [label="Procesamiento y alertas\\nPython y SQL", fillcolor="#CCE7E4"];
      DASH [label="Dashboard de KPIs\\nStreamlit o Power BI", fillcolor="#FDE68A"];
      ERP [label="ERP\\nFI, PM/EAM, Inventario", fillcolor="#FDE68A"];
      PUB [label="Panel público para la comunidad\\nruido, temperatura de salida, agua", fillcolor="#FDE68A"];
      ALR [label="Alertas al jefe de turno\\ncorreo y SMS", fillcolor="#FECACA"];
      S1 -> GW; S2 -> GW; S3 -> GW; S4 -> GW; S5 -> GW;
      GW -> BR -> DB -> ETL -> DASH;
      ETL -> ALR; ETL -> PUB; ETL -> ERP [dir=both, label="API REST", fontsize=10];
    }""")
    st.subheader("Contrato de datos")
    st.caption("La interoperabilidad se garantiza con protocolos abiertos en la planta y con un esquema común hacia el dashboard: "
               "marca de tiempo ISO 8601 (America/Bogota), unidades del SI y los mismos nombres de columna. "
               "Este es el esquema del CSV que acepta la barra lateral.")
    contrato = pd.DataFrame([
        ["hora", "Marca de tiempo horaria", "Todos los sensores (reloj NTP)", "ISO 8601", "Obligatoria"],
        ["ti_MW", "Carga de TI", "Medidores de PDU y UPS", "MW", "Obligatoria"],
        ["PUE", "Energía total / energía de TI", "Calculado en el procesamiento", "adimensional", "Obligatoria"],
        ["WUE_L_kWh", "Agua consumida / energía de TI", "Caudalímetro + medidor de TI", "L/kWh", "Obligatoria"],
        ["Sala A, Sala B...", "Temperatura de entrada de aire", "Sensores de pasillo frío (cada 1 a 5 min, promedio horario)", "°C", "Al menos una sala"],
        ["exterior_C", "Temperatura exterior", "Estación meteorológica", "°C", "Opcional"],
        ["ruido_dBA", "Ruido perimetral", "Sonómetro", "dB(A)", "Opcional"],
        ["humedad_pct", "Humedad relativa en sala", "Sensores de pasillo frío", "% HR", "Opcional"],
        ["salida_C", "Temperatura del aire de descarga", "Sensor en la salida de aire", "°C", "Opcional"],
        ["caida_min", "Minutos sin servicio en la hora", "Sistema de monitoreo", "min", "Opcional (calcula la disponibilidad)"],
    ], columns=["Columna", "Variable", "Origen", "Unidad", "Estado"])
    st.dataframe(contrato, hide_index=True, width="stretch")

# ---------------- Contexto y fuentes ----------------
with t_ctx:
    st.subheader("Tamaño del proyecto frente al país")
    demanda_nac = 264.47 * 365                      # GWh/año: demanda promedio de sept. 2026 (XM), anualizada
    gwh_medido = df_all.energia_total_MWh.mean() * 8760 / 1000
    gwh_pleno = 120 * pue_op * 8760 / 1000          # 120 MW de TI a plena carga con el PUE de operación
    c = st.columns(3)
    c[0].metric("Consumo anualizado (datos del dashboard)", f"{gwh_medido:,.0f} GWh",
                f"{gwh_medido / demanda_nac:.1%} de la demanda nacional", delta_color="off")
    c[1].metric("A plena carga (120 MW de TI)", f"{gwh_pleno:,.0f} GWh",
                f"{gwh_pleno / demanda_nac:.1%} de la demanda nacional", delta_color="off")
    c[2].metric("Frente a la brecha de energía firme 2026-27", f"{gwh_pleno / 7207:.0%}",
                "brecha de 7.207 GWh/año (XM)", delta_color="off")
    st.caption("Cálculo propio. La demanda nacional se anualiza desde el promedio diario de septiembre de 2026 "
               "(264,47 GWh/día). La plena carga es un escenario máximo; las primeras fases serán menores. "
               "El informe usa 0,9 %, que compara 120 MW de TI con la demanda máxima de 13.186 MW (potencia); "
               "aquí se compara energía anual incluyendo el PUE, por eso la cifra es mayor.")

    a, b = st.columns(2)
    mw = pd.DataFrame({"Referencia": ["ODATA BG03 (TI, planeado)", "Colombia instalado (est. jul. 2026)",
                                      "Bogotá, inventario 1T 2026", "ODATA BG01 (TI)"],
                       "MW": [120, 55, 44.3, 14.6]})
    f = go.Figure(go.Bar(x=mw.MW, y=mw.Referencia, orientation="h",
                         marker_color=[TEAL, GREY, GREY, GREY], text=mw.MW, textposition="outside"))
    f.update_yaxes(autorange="reversed")
    a.plotly_chart(estilo(f, 300, "Capacidad de TI (MW). Colombia: rango 50-60 MW"), width="stretch")
    pues = pd.DataFrame({"Referencia": ["Datos del dashboard", "Meta del informe", "Hiperescala nuevo",
                                        "Promedio global 2025"],
                         "PUE": [pue_med, 1.30, 1.40, 1.54]})
    f = go.Figure(go.Bar(x=pues.Referencia, y=pues.PUE, marker_color=[TEAL, AMBER, GREY, GREY],
                         text=pues.PUE.round(2), textposition="outside"))
    f.update_yaxes(range=[1, 1.7])
    b.plotly_chart(estilo(f, 300, "PUE frente a referencias"), width="stretch")
    wues = pd.DataFrame({"Referencia": ["Datos del dashboard", "Meta Platforms", "Microsoft", "Promedio EE. UU."],
                         "WUE": [wue_med, 0.26, 0.49, 1.8]})
    f = go.Figure(go.Bar(x=wues.Referencia, y=wues.WUE, marker_color=[TEAL, GREY, GREY, GREY],
                         text=wues.WUE.round(2), textposition="outside"))
    st.plotly_chart(estilo(f, 300, "WUE (L/kWh) frente a referencias"), width="stretch")

    st.subheader("Datos de referencia y fuentes")
    fuentes = pd.DataFrame([
        ["PUE promedio global 2025", "1,54", "Uptime Institute, Global Data Center Survey 2025", "Alta"],
        ["PUE de diseño hiperescala nuevo", "1,4 o menos", "Uptime Institute (blog de investigación)", "Media"],
        ["WUE promedio de EE. UU.", "1,8 L/kWh", "Shehabi et al. (2016), Lawrence Berkeley National Laboratory", "Alta (datos de 2014)"],
        ["WUE de operadores grandes", "Meta 0,26; Microsoft 0,49 L/kWh", "Autorreportado por las empresas", "Media"],
        ["Consumo mundial de centros de datos 2024", "415 TWh (1,5 %)", "IEA, Energy and AI (2025)", "Alta"],
        ["Proyección mundial a 2030", "unos 945 TWh", "IEA, Energy and AI (2025)", "Alta (escenario)"],
        ["Capacidad instalada en Colombia", "50 a 60 MW", "Panel de julio de 2026, citado por ITSitio", "Baja a media"],
        ["Inventario de Bogotá, 1T 2026", "44,3 MW (36 ocupados)", "CBRE, citado por ITSitio", "Media"],
        ["Brecha de energía firme 2026-27", "7.207 GWh/año", "XM, Boletín Energético 359", "Alta"],
        ["Demanda promedio, sept. 2026", "264,47 GWh/día", "XM, citado por Portafolio", "Alta"],
        ["Precio contratos industriales, jun. 2026", "311,8 COP/kWh", "XM, citado por El Colombiano", "Alta"],
        ["Precio de bolsa, ago. 2026", "946,6 COP/kWh", "XM, citado por Portafolio", "Alta"],
        ["Rangos térmicos de equipos", "18-27 °C recomendado; A1 hasta 32; H1 18-22", "ASHRAE TC 9.9 (2021)", "Alta"],
        ["Humedad relativa en sala", "Banda interna 30-60 %; ASHRAE: máx. 70 % recomendado, 20-80 % permitido (A1)", "ASHRAE TC 9.9 (2021)", "Media: verificar en la norma"],
        ["Renta en zona franca", "20 % condicionado, frente a 35 % general", "Mincit; Ley 2277 de 2022", "Media: verificar el caso"],
        ["Residuos electrónicos en Colombia", "unas 250.000 t (2024)", "Ministerio de Ambiente", "Media"],
        ["ODATA BG01", "14,6 MW; hasta 10 kW por rack", "DCD; Developing Telecoms", "Media"],
        ["Financiamiento verde de ODATA", "US$ 1.020 M (total US$ 2.250 M)", "Comunicado de ODATA, septiembre de 2025", "Alta (anuncio de la empresa)"],
        ["BG03: 120 MW de TI; US$ 1.300 M (BG02 + BG03)", "primeras fases 2S 2026", "Comunicado de ODATA, octubre de 2024", "Alta (anuncio, no auditado)"],
    ], columns=["Indicador", "Valor", "Fuente", "Confianza"])
    st.dataframe(fuentes, hide_index=True, width="stretch")
    st.warning("Sin verificar en fuentes independientes: las 65 ha, el vacío del Decreto 1076 de 2015, las 4,41 ha de "
               "compensación, la línea de 220 kV «Navarra a Bacatá» y los actores comunitarios de la presentación. "
               "Confírmelos con ANLA, la CAR de Cundinamarca y la Alcaldía de Tenjo.")
