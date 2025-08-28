from fastapi import FastAPI
from typing import List, Dict
import httpx, io, re, csv
import pandas as pd
import pdfplumber
from datetime import datetime, timedelta

app = FastAPI()
TZ = "Europe/Madrid"

# Fuentes (IDs/URLs públicas)
CMA_SHEET = ("1mcfvtbneJppww2gw93ByL02w9QbzR77lfEl7RmJqTzg", "1518867162")
ONE_SHEET = ("18MMzfK60eRAKj37lBkcNmYob_qXRbbO0j917v-gw-vE", "0")
MAERSK_PAGE = "https://www.maersk.com/local-information/europe/spain/export"
MSC_PAGE   = "https://www.msc.com/es/local-information/europe/spain#CustomsnbspClearanceInformation"
HAPAG_PAGE = "https://www.hapag-lloyd.com/es/services-information/offices-localinfo/europe/spain.html#tab=ti-vessel-calls-export-spain"

# ---------- utilidades ----------
def parse_sheet_csv(sheet_id: str, gid: str) -> List[Dict]:
    """
    Lee una pestaña pública de Google Sheets en formato CSV y devuelve filas normalizadas
    con campos comunes. Nunca lanza excepción: si algo falla, retorna [] y escribe un WARN.
    """
    import csv, io, re, httpx

    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    # 1) Descarga segura (maneja HTTP≠200 y timeouts)
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as c:
            r = c.get(url)
            if r.status_code != 200:
                print(f"[WARN] sheet {sheet_id} gid {gid} -> HTTP {r.status_code}")
                return []
            content = r.content.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] parse_sheet_csv failed: {e}")
        return []

    # 2) Parseo CSV
    rows = []
    reader = csv.reader(io.StringIO(content))
    data = list(reader)
    if not data:
        return []

    headers = [ (h or "").strip().lower() for h in data[0] ]

    # Helper para localizar columnas por regex
    def find(patterns):
        for i, h in enumerate(headers):
            for p in patterns:
                if re.search(p, h, flags=re.I):
                    return i
        return -1

    idx_vessel = find([r"^vessel(\s|$)|vessel name"])
    idx_voyage = find([r"^voy(age)?$|^voy "])
    idx_service= find([r"^service(\s|$)|service code"])
    idx_term   = find([r"^terminal$"])
    idx_etd    = find([r"^etd(\s|$)|etd local|departure"])
    idx_eta    = find([r"^eta(\s|$)|eta local|arrival"])
    idx_doc    = find([r"doc.*cut.?off|instruction"])
    idx_desp   = find([r"customs|despacho"])

    # 3) Construcción robusta de filas (comprueba longitudes)
    out = []
    for r in data[1:]:
        def val(idx):
            return (r[idx].strip() if (idx >= 0 and idx < len(r) and r[idx] is not None) else "")
        out.append({
            "service":           val(idx_service),
            "vessel":            val(idx_vessel),
            "voyage":            val(idx_voyage),
            "terminal":          val(idx_term),
            "ETD_local":         val(idx_etd),
            "ETA_local":         val(idx_eta),
            "DOC_cutoff":        val(idx_doc),
            "DESPACHO_cutoff":   val(idx_desp),
        })
    return out

def friday_shift_minus_hours(dt: datetime, hours: int) -> datetime:
    dow = dt.weekday()  # 0=lunes, 6=domingo
    base = dt
    if dow == 5:  # sábado
        base = dt - timedelta(days=1)
    elif dow == 6:  # domingo
        base = dt - timedelta(days=2)
    return base - timedelta(hours=hours)

def classify(now: datetime, deadline: datetime|None, buffer_h: int) -> str:
    if not deadline: return ""
    diff = (deadline - now).total_seconds()/3600
    if diff < 0: return "OVERDUE"
    if diff <= buffer_h: return "AT_RISK"
    return "OK"

def to_dt(s: str) -> datetime|None:
    if not s: return None
    for fmt in ("%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except Exception:
            continue
    return None

# ---------- conectores ----------
def pull_cma() -> List[Dict]:
    rows = parse_sheet_csv(*CMA_SHEET)
    out = []
    for r in rows:
        out.append({
            "carrier":"CMA",
            **r,
            "source":"CMA",
            "source_link": f"https://docs.google.com/spreadsheets/d/{CMA_SHEET[0]}/edit#gid={CMA_SHEET[1]}",
        })
    return out

def pull_one() -> List[Dict]:
    rows = parse_sheet_csv(*ONE_SHEET)
    out = []
    for r in rows:
        out.append({
            "carrier":"ONE",
            **r,
            "source":"ONE",
            "source_link": f"https://docs.google.com/spreadsheets/d/{ONE_SHEET[0]}/edit#gid={ONE_SHEET[1]}",
        })
    return out

def pull_maersk() -> List[Dict]:
    with httpx.Client(follow_redirects=True, timeout=30.0) as c:
        html = c.get(MAERSK_PAGE).text
    m = re.search(r'https:[^"\']+\\.xlsx', html)
    if not m: return []
    xurl = m.group(0)
    with httpx.Client(follow_redirects=True, timeout=60.0) as c:
        xbytes = c.get(xurl).content
    df = pd.read_excel(io.BytesIO(xbytes))
    # heurística de columnas
    cols = {k.lower():k for k in df.columns}
    def get(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    vcol = get("vessel","vessel name"); ycol=get("voyage","voy")
    scol = get("service","service code"); tcol=get("terminal")
    etd  = get("etd","etd local"); eta=get("eta","eta local")
    doc  = [c for c in cols if re.search(r"doc.*off|instruction", c)] 
    desp = [c for c in cols if re.search(r"customs|despacho", c)]
    out = []
    for _,r in df.iterrows():
        out.append({
            "carrier":"MAERSK",
            "service": str(r.get(scol,"") or ""),
            "vessel":  str(r.get(vcol,"") or ""),
            "voyage":  str(r.get(ycol,"") or ""),
            "terminal":str(r.get(tcol,"") or ""),
            "ETD_local": str(r.get(etd,"") or ""),
            "ETA_local": str(r.get(eta,"") or ""),
            "DOC_cutoff": str(r.get(doc[0],"") if doc else ""),
            "DESPACHO_cutoff": str(r.get(desp[0],"") if desp else ""),
            "source":"MAERSK",
            "source_link": MAERSK_PAGE
        })
    return out

# MSC y Hapag requieren Playwright/pdfplumber.
# Para simplificar, dejamos Hapag por PDF (ETA-24h) y un stub de MSC que
# probablemente necesite ajustar selectores (lo aviso claramente).
from playwright.async_api import async_playwright
import asyncio

MSC_PODS = [  # lista exacta que nos diste
 'LUANDA','JEDDAH','BUENOS AIRES','YEREVAN','SYDNEY','CHITTAGONG','SANTA CRUZ','PARANAGUA','DAR ES SALAAM','MONTREAL','SANTIAGO',
 'CAT LAI','DOUALA','SAN ANTONIO','CARTAGENA','BUSAN','SAN JOSE PORT','ABIDJAN','MARIEL','GUAYAQUIL','HOUSTON','MIAMI','BOSTON PORT',
 'EL SALVADOR','JEBEL ALI','CEBU','POTI','PUERTO CORTES','DADRI','NHAVA SHEBA','MUNDRA PORT','JAKARTA PORT','BASRAH','ASHDOD','MOMBASA',
 'MONROVIA','SINGAPUR','CASABLANCA','PORT LOUIS','VERACRUZ','KOLKATA','CORINTO','TINCAN','ONNE PORT NIGERIA','HAIFA','CRISTOBAL','MANZANILLO',
 'ASUNCIÓN','CALLAO PORT','SAN JUAN','DOHA','HAMAD','CAUCEDO','HAINA','DURBAN','SURINAM','MERSIN','TAICHUNG','KEELUNG','RADES','MONTEVIDEO',
 'PUERTO CABELLO','CHONGQING','QUINGDAO'
]

async def pull_msc_async() -> List[Dict]:
    rows = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(MSC_PAGE, wait_until="domcontentloaded")
        # IMPORTANTE: Los selectores pueden cambiar. Esto es un esqueleto y puede requerir ajuste.
        # Devolvemos vacío si no encontramos el widget (lo veremos en los logs).
        # TODO: ajustar selectores reales de la herramienta de MSC.
        await browser.close()
    return rows

def pull_hapag() -> List[Dict]:
    with httpx.Client(follow_redirects=True, timeout=30.0) as c:
        html = c.get(HAPAG_PAGE).text
        m = re.search(r'href="(https:[^"]*sailingvesselsBarcelona[^"]*\\.pdf)"', html, flags=re.I)
        if not m: return []
        pdf_url = m.group(1)
        pdf_bytes = c.get(pdf_url).content
    out = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tbl in tables or []:
                if not tbl or len(tbl)<2: continue
                headers = [ (h or "").strip().lower() for h in tbl[0] ]
                def find(name,*alts):
                    cans = (name,)+alts
                    for i,h in enumerate(headers):
                        if h in cans: return i
                    return -1
                i_vessel = find("vessel","vessel name")
                i_voy    = find("voyage","voy")
                i_srv    = find("service")
                i_term   = find("terminal")
                i_etd    = find("etd","departure","etd local")
                i_eta    = find("eta","arrival","eta local")
                for r in tbl[1:]:
                    vessel = (r[i_vessel] if i_vessel>=0 else "") or ""
                    if not vessel: continue
                    d = {
                        "carrier":"HAPAG",
                        "service": (r[i_srv] if i_srv>=0 else "") or "",
                        "vessel": vessel,
                        "voyage": (r[i_voy] if i_voy>=0 else "") or "",
                        "terminal": (r[i_term] if i_term>=0 else "") or "",
                        "ETD_local": (r[i_etd] if i_etd>=0 else "") or "",
                        "ETA_local": (r[i_eta] if i_eta>=0 else "") or "",
                        "DOC_cutoff": "",  # no lo publican
                        "DESPACHO_cutoff": "",  # lo calcularemos en el cliente si está vacío
                        "source":"HAPAG",
                        "source_link": HAPAG_PAGE
                    }
                    out.append(d)
    return out

# ---------- endpoint unificado ----------
def _safe(fn, name: str):
    try:
        return fn()
    except Exception as e:
        print(f"[WARN] {name} failed: {e}")
        return []

@app.get("/unified")
async def unified(pol: str = "Barcelona"):
    now = datetime.utcnow()
    rows: List[Dict] = []
    rows += _safe(pull_cma, "CMA")
    rows += _safe(pull_one, "ONE")
    rows += _safe(pull_maersk, "MAERSK")
    rows += _safe(pull_hapag, "HAPAG")

    # MSC (Playwright) – mejor esfuerzo, nunca rompe la respuesta
    try:
        rows += await pull_msc_async()
    except Exception as e:
        print(f"[WARN] MSC failed: {e}")

    out = []
    for r in rows:
        doc  = to_dt(r.get("DOC_cutoff",""))
        desp = to_dt(r.get("DESPACHO_cutoff","")) if r.get("DESPACHO_cutoff","") else None
        eta  = to_dt(r.get("ETA_local",""))

        # Hapag: si no hay despacho explícito → ETA−24h con regla de viernes
        if (not desp) and r.get("carrier","").upper().startswith("HAPAG") and eta:
            desp = friday_shift_minus_hours(eta, 24)

        st_doc  = classify(now, doc, 36)
        st_desp = classify(now, desp, 24)

        out.append({
            **r,
            "DOC_cutoff":       doc.strftime("%Y-%m-%d %H:%M")  if doc  else "",
            "DESPACHO_cutoff":  desp.strftime("%Y-%m-%d %H:%M") if desp else "",
            "status_DOC":       st_doc,
            "status_DESPACHO":  st_desp
        })
    return out
    
@app.get("/health")
def health(): return {"ok": True}
