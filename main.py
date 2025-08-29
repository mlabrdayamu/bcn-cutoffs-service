from fastapi import FastAPI
from typing import List, Dict
import httpx, io, re, csv
import pandas as pd
import pdfplumber
from datetime import datetime, timedelta
import asyncio

ENABLE_MSC = False  # MSC desactivado hasta afinar selectores

app = FastAPI()
TZ = "Europe/Madrid"

# Fuentes (IDs/URLs públicas)
CMA_SHEET = ("1mcfvtbneJppww2gw93ByL02w9QbzR77lfEl7RmJqTzg", "1518867162")
ONE_SHEET = ("18MMzfK60eRAKj37lBkcNmYob_qXRbbO0j917v-gw-vE", "0")
MAERSK_PAGE = "https://www.maersk.com/local-information/europe/spain/export"
MSC_PAGE   = "https://www.msc.com/es/local-information/europe/spain#CustomsnbspClearanceInformation"
HAPAG_PAGE = "https://www.hapag-lloyd.com/es/services-information/offices-localinfo/europe/spain.html#tab=ti-vessel-calls-export-spain"

# ---------- utilidades ----------

def is_esbcn(s: str) -> bool:
    """True si el código empieza por ESBCN (ignora espacios)."""
    if not s: return False
    u = re.sub(r"\s+", "", str(s).upper())
    return u.startswith("ESBCN")

def friday_shift_minus_hours(dt: datetime, hours: int) -> datetime:
    # Si el cálculo cae fin de semana, mueve a viernes antes de restar horas
    dow = dt.weekday()  # 0=lun..6=dom
    base = dt
    if dow == 5:   # sábado -> viernes
        base = dt - timedelta(days=1)
    elif dow == 6: # domingo -> viernes
        base = dt - timedelta(days=2)
    return base - timedelta(hours=hours)

def classify(now: datetime, deadline: datetime|None, buffer_h: int) -> str:
    if not deadline: return ""
    diff = (deadline - now).total_seconds()/3600
    if diff < 0: return "OVERDUE"
    if diff <= buffer_h: return "AT_RISK"
    return "OK"

def to_dt(s: str, ref: datetime | None = None) -> datetime | None:
    """Extrae TODAS las fechas del texto y devuelve la MÁS TARDE.
       Soporta '28/08/2025 10H', '29/08 12:00', '27/08/25 09H', etc.
       Si no hay año, usa el de ref o el actual. Años 2 dígitos -> 20xx."""
    if not s: return None
    txt = str(s).strip()
    if txt.upper() in {"-", "OMIT", "CLOSED", "TBN", "NA"}: return None

    # Normaliza '10H' -> '10:00'
    txt = re.sub(r"\b(\d{1,2})\s*H\b", r"\1:00", txt, flags=re.I)

    # Captura TODAS las ocurrencias fecha (+hora opcional)
    tokens = [m.group(0) for m in re.finditer(
        r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?(?:\s+\d{1,2}(?::\d{2})?)?\b", txt)]
    if not tokens: return None

    fmts = [
        "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M",
        "%d/%m/%y %H:%M", "%d-%m-%y %H:%M",
        "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y",
        "%d/%m/%y", "%d-%m-%y",
        "%d/%m %H:%M", "%d-%m %H:%M",
    ]

    candidates: List[datetime] = []
    for raw in tokens:
        for fmt in fmts:
            try:
                dt = datetime.strptime(raw, fmt)
                # Sin año -> aplica ref o año actual
                if "%Y" not in fmt and "%y" not in fmt:
                    base_year = (ref.year if ref else datetime.utcnow().year)
                    dt = dt.replace(year=base_year)
                # Año 2 dígitos -> 2000+yy
                if "%y" in fmt and dt.year < 100:
                    dt = dt.replace(year=2000 + dt.year)
                candidates.append(dt)
                break
            except Exception:
                continue
    if not candidates: return None
    return max(candidates)  # usa la más tardía del texto

# ---------- parsers Sheets ----------

def parse_sheet_csv(sheet_id: str, gid: str) -> List[Dict]:
    """Parser genérico (detecta cabecera, filtra vacíos)."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as c:
            r = c.get(url); r.raise_for_status()
            content = r.content.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] parse_sheet_csv failed: {e}")
        return []
    reader = csv.reader(io.StringIO(content))
    rows_raw = [row for row in reader if any((cell or "").strip() for cell in row)]
    if not rows_raw: return []

    header_idx, header = None, []
    pats = [r"vessel", r"\bvoy", r"service", r"\betd\b", r"\beta\b",
            r"doc.*off|instruction|\bsi\b|shipping.*instr", r"customs|aduan|despach|\bdue\b", r"port of loading|pol"]
    for i,row in enumerate(rows_raw[:50]):
        lower = [(cell or "").strip().lower() for cell in row]
        score = sum(any(re.search(p,h) for h in lower) for p in pats)
        if score >= 2: header_idx, header = i, lower; break
    if header_idx is None: return []
    data = rows_raw[header_idx+1:]

    def find(*patterns):
        for i,h in enumerate(header):
            for p in patterns:
                if re.search(p,h,flags=re.I): return i
        return -1

    i_vessel = find(r"^vessel(\s|$)|vessel name")
    i_voy    = find(r"^voy(age)?$|^voy ")
    i_srv    = find(r"^service(\s|$)|service code")
    i_term   = find(r"^terminal|terminal berth")
    i_etd    = find(r"^etd(\s|$)|etd.*barcelona|departure|vessel etd")
    i_eta    = find(r"^eta(\s|$)|eta.*barcelona|arrival|vessel eta")
    i_doc    = find(r"doc.*cut.?off|instruction|\bsi\b|shipping.*instr")
    i_desp   = find(r"customs|aduan|despach|\bdue\b")
    i_pol    = find(r"port of loading|^pol$|origin")

    def val(row, idx): return (row[idx].strip() if idx>=0 and idx<len(row) and row[idx] else "")

    out: List[Dict] = []
    for r in data:
        row = {
            "service":          val(r, i_srv),
            "vessel":           val(r, i_vessel),
            "voyage":           val(r, i_voy),
            "terminal":         val(r, i_term),
            "ETD_local":        val(r, i_etd),
            "ETA_local":        val(r, i_eta),
            "DOC_cutoff":       val(r, i_doc),
            "DESPACHO_cutoff":  val(r, i_desp),
            "pol":              val(r, i_pol),
        }
        if any(row[k] for k in ["vessel","voyage","ETD_local","ETA_local","DOC_cutoff","DESPACHO_cutoff","pol"]):
            out.append(row)
    return out

def parse_cma_csv(sheet_id: str, gid: str) -> List[Dict]:
    """CMA: mapea cabeceras exactas y filtra SOLO ESBCN (POL o Terminal Berth)."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as c:
            r = c.get(url); r.raise_for_status()
            data = list(csv.reader(io.StringIO(r.content.decode("utf-8", errors="ignore"))))
    except Exception as e:
        print(f"[WARN] CMA CSV failed: {e}")
        return []
    if not data: return []

    # Detecta cabecera (por 'Vessel ETA' y 'Shipping Instructions')
    header_idx, header = None, []
    for i, row in enumerate(data[:50]):
        low = [(x or "").strip().lower() for x in row]
        if ("vessel eta" in " ".join(low)) and ("shipping instructions" in " ".join(low)):
            header_idx, header = i, low; break
    if header_idx is None:
        header_idx, header = 0, [(x or "").strip().lower() for x in data[0]]
    rows = data[header_idx+1:]

    def col(*pats):
        for i,h in enumerate(header):
            for p in pats:
                if re.search(p,h,flags=re.I): return i
        return -1

    i_srv    = col(r"^service\b")
    i_voy    = col(r"^voyage\b|^voy\b")
    i_vessel = col(r"^vessel\b")
    i_pol    = col(r"^port of loading\b")
    i_eta    = col(r"^vessel eta\b")
    i_etd    = col(r"^vessel etd\b")
    i_doc    = col(r"^shipping instructions cut off")   # DOC
    i_desp   = col(r"^customs clearance cut off")       # DESPACHO
    i_term   = col(r"^terminal berth\b")

    out: List[Dict] = []
    for r in rows:
        v = lambda i: (r[i].strip() if i>=0 and i<len(r) and r[i] else "")
        pol  = v(i_pol)
        term = v(i_term)
        # Sólo ESBCN (POL o Terminal)
        if not (pol.upper().startswith("ESBCN") or term.upper().startswith("ESBCN")):
            continue

        out.append({
            "service":          v(i_srv),
            "vessel":           v(i_vessel),
            "voyage":           v(i_voy),
            "terminal":         term,
            "ETD_local":        v(i_etd),
            "ETA_local":        v(i_eta),
            "DOC_cutoff":       v(i_doc),
            "DESPACHO_cutoff":  v(i_desp),
            "DOC_text":         v(i_doc),   # <-- texto original completo
            "DESPACHO_text":    v(i_desp),  # <-- texto original completo
            "pol":              pol or term,
        })
    return [x for x in out if any(x[k] for k in ["vessel","voyage","ETD_local","ETA_local","DOC_cutoff","DESPACHO_cutoff"])]

def parse_one_csv(sheet_id: str, gid: str) -> List[Dict]:
    """ONE: 'SI BL' (DOC), 'CUSTOMS' (DESPACHO). Hoja BCN -> forzar pol ESBCN."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as c:
            r = c.get(url); r.raise_for_status()
            data = list(csv.reader(io.StringIO(r.content.decode("utf-8", errors="ignore"))))
    except Exception as e:
        print(f"[WARN] ONE CSV failed: {e}")
        return []
    if not data: return []

    header_idx, header = None, []
    for i,row in enumerate(data[:50]):
        low = [(x or "").strip().lower() for x in row]
        if ("vessel name" in " ".join(low)) and ("si bl" in " ".join(low)) and ("customs" in " ".join(low)):
            header_idx, header = i, low; break
    if header_idx is None:
        header_idx, header = 0, [(x or "").strip().lower() for x in data[0]]
    rows = data[header_idx+1:]

    def col(*p):
        for i,h in enumerate(header):
            for pat in p:
                if re.search(pat,h,flags=re.I): return i
        return -1

    i_vvd    = col(r"^vvd$|^dep.*voy|^voy")
    i_vessel = col(r"vessel name")
    i_eta    = col(r"^eta$")
    i_etd    = col(r"^etd$")
    i_doc    = col(r"^si\s*bl|shipping.*instr")
    i_desp   = col(r"^customs$")
    i_term   = col(r"terminal")
    i_lane   = col(r"^code$|^lane$")

    out: List[Dict] = []
    for r in rows:
        v = lambda i: (r[i].strip() if i>=0 and i<len(r) and r[i] else "")
        out.append({
            "service":          v(i_lane),
            "vessel":           v(i_vessel),
            "voyage":           v(i_vvd),
            "terminal":         v(i_term),
            "ETD_local":        v(i_etd),
            "ETA_local":        v(i_eta),
            "DOC_cutoff":       v(i_doc),
            "DESPACHO_cutoff":  v(i_desp),
            "DOC_text":         v(i_doc),   # <-- texto original completo
            "DESPACHO_text":    v(i_desp),  # <-- texto original completo
            "pol":              "ESBCN",
        })
    return [x for x in out if any(x[k] for k in ["vessel","voyage","ETD_local","ETA_local","DOC_cutoff","DESPACHO_cutoff"])]

# ---------- conectores ----------

def pull_cma() -> List[Dict]:
    rows = parse_cma_csv(*CMA_SHEET)
    return [{
        "carrier":"CMA", **r,
        "source":"CMA",
        "source_link": f"https://docs.google.com/spreadsheets/d/{CMA_SHEET[0]}/edit#gid={CMA_SHEET[1]}",
    } for r in rows]

def pull_one() -> List[Dict]:
    rows = parse_one_csv(*ONE_SHEET)
    return [{
        "carrier":"ONE", **r,
        "source":"ONE",
        "source_link": f"https://docs.google.com/spreadsheets/d/{ONE_SHEET[0]}/edit#gid={ONE_SHEET[1]}",
    } for r in rows]

def httpx_get_with_retry(url: str, timeout: float = 60.0, attempts: int = 3) -> httpx.Response:
    last = None
    for _ in range(attempts):
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout) as c:
                return c.get(url)
        except Exception as e:
            last = e
    raise last if last else RuntimeError("request failed")

def pull_maersk() -> List[Dict]:
    try:
        html = httpx_get_with_retry(MAERSK_PAGE, timeout=60.0).text
    except Exception as e:
        print(f"[WARN] MAERSK failed: {e}")
        return []
    m = re.search(r'https://[^"\']+\.xlsx', html)
    if not m:
        print("[WARN] MAERSK xlsx link not found")
        return []
    xurl = m.group(0)
    try:
        xbytes = httpx_get_with_retry(xurl, timeout=90.0).content
    except Exception as e:
        print(f"[WARN] MAERSK failed: {e}")
        return []
    df = pd.read_excel(io.BytesIO(xbytes))

    cols = {str(k).strip().lower(): k for k in df.columns}
    def get(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    vcol = get("vessel","vessel name")
    ycol = get("voyage","voy")
    scol = get("service","service code")
    tcol = get("terminal","terminal berth")
    etd  = get("etd","etd local","dep","departure")
    eta  = get("eta","eta local","arr","arrival")
    polc = get("pol","port of loading","origin port","origin","load port")

    doc_cols  = [cols[k] for k in cols if re.search(r"doc.*off|instruction|\bsi\b", k)]
    desp_cols = [cols[k] for k in cols if re.search(r"customs|aduan|despach", k)]

    out = []
    for _,r in df.iterrows():
        pol_val  = str(r.get(polc,"") or "")
        term_val = str(r.get(tcol,"") or "")
        if not (is_esbcn(pol_val) or is_esbcn(term_val)):
            continue  # SOLO ESBCN
        out.append({
            "carrier":"MAERSK",
            "service": str(r.get(scol,"") or ""),
            "vessel":  str(r.get(vcol,"") or ""),
            "voyage":  str(r.get(ycol,"") or ""),
            "terminal":term_val,
            "ETD_local": str(r.get(etd,"") or ""),
            "ETA_local": str(r.get(eta,"") or ""),
            "DOC_cutoff": str(r.get(doc_cols[0],"") if doc_cols else ""),
            "DESPACHO_cutoff": str(r.get(desp_cols[0],"") if desp_cols else ""),
            "DOC_text": str(r.get(doc_cols[0],"") if doc_cols else ""),
            "DESPACHO_text": str(r.get(desp_cols[0],"") if desp_cols else ""),
            "pol": pol_val,
            "source":"MAERSK",
            "source_link": MAERSK_PAGE
        })
    return [x for x in out if any(x[k] for k in ["vessel","voyage","ETD_local","ETA_local","DOC_cutoff","DESPACHO_cutoff"])]

async def pull_msc_async() -> List[Dict]:
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        print(f"[WARN] Playwright import failed: {e}")
        return []
    rows: List[Dict] = []
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(MSC_PAGE, wait_until="domcontentloaded")
            # TODO: scraping real; por ahora vacío
            await browser.close()
    except Exception as e:
        print(f"[WARN] MSC scraping failed: {e}")
    return rows

def pull_hapag() -> List[Dict]:
    try:
        html = httpx_get_with_retry(HAPAG_PAGE, timeout=30.0).text
    except Exception as e:
        print(f"[WARN] HAPAG page failed: {e}")
        return []
    # Buscar PDF que contenga 'barcelona'
    m = re.search(r'href="([^"]+\.pdf[^"]*)"', html, flags=re.I)
    pdf_url = None
    if m and "barcelona" in m.group(1).lower():
        pdf_url = m.group(1)
    if not pdf_url:
        for u in re.findall(r'href="([^"]+\.pdf[^"]*)"', html, flags=re.I):
            if "barcelona" in u.lower():
                pdf_url = u; break
    if not pdf_url:
        print("[WARN] HAPAG pdf link not found")
        return []
    try:
        pdf_bytes = httpx_get_with_retry(pdf_url, timeout=60.0).content
    except Exception as e:
        print(f"[WARN] HAPAG pdf download failed: {e}")
        return []

    out = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tbl in tables or []:
                if not tbl or len(tbl) < 2: continue
                headers = [(h or "").strip().lower() for h in tbl[0]]
                def find(name,*alts):
                    cans = (name,)+alts
                    for i,h in enumerate(headers):
                        if h in cans: return i
                    return -1
                i_vessel = find("vessel","vessel name")
                i_voy    = find("voyage","voy")
                i_srv    = find("service")
                i_term   = find("terminal","terminal berth")
                i_etd    = find("etd","departure","etd local")
                i_eta    = find("eta","arrival","eta local")
                for r in tbl[1:]:
                    vessel = (r[i_vessel] if i_vessel>=0 else "") or ""
                    if not vessel: continue
                    out.append({
                        "carrier":"HAPAG",
                        "service": (r[i_srv] if i_srv>=0 else "") or "",
                        "vessel": vessel,
                        "voyage": (r[i_voy] if i_voy>=0 else "") or "",
                        "terminal": (r[i_term] if i_term>=0 else "") or "",
                        "ETD_local": (r[i_etd] if i_etd>=0 else "") or "",
                        "ETA_local": (r[i_eta] if i_eta>=0 else "") or "",
                        "DOC_cutoff": "",
                        "DESPACHO_cutoff": "",
                        "DOC_text": "",
                        "DESPACHO_text": "",
                        "pol": "ESBCN",
                        "source":"HAPAG",
                        "source_link": HAPAG_PAGE
                    })
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

    if ENABLE_MSC:
        try:
            rows += await pull_msc_async()
        except Exception as e:
            print(f"[WARN] MSC failed: {e}")

    # Filtro FINAL: sólo filas con pol o terminal que empiece por ESBCN
    rows = [r for r in rows if is_esbcn(r.get("pol","")) or is_esbcn(r.get("terminal",""))]

    # Normalización + estados
    out = []
    for r in rows:
        eta  = to_dt(r.get("ETA_local",""))
        etd  = to_dt(r.get("ETD_local",""), ref=eta or datetime.utcnow())
        doc  = to_dt(r.get("DOC_cutoff",""),       ref=eta or etd)
        desp = to_dt(r.get("DESPACHO_cutoff",""),  ref=eta or etd)

        # Hapag: si no viene despacho explícito => ETA-24h con regla de viernes
        if (not desp) and r.get("carrier","").upper().startswith("HAPAG") and eta:
            desp = friday_shift_minus_hours(eta, 24)

        st_doc  = classify(now, doc, 36)   # buffer DOC 36h
        st_desp = classify(now, desp, 24)  # buffer DESPACHO 24h

        out.append({
            **r,
            "DOC_cutoff":      doc.strftime("%Y-%m-%d %H:%M")  if doc  else "",
            "DESPACHO_cutoff": desp.strftime("%Y-%m-%d %H:%M") if desp else "",
            "status_DOC":      st_doc,
            "status_DESPACHO": st_desp
        })
    return out

@app.get("/health")
def health():
    return {"ok": True}
