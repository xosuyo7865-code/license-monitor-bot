
import os, re, time, json, hashlib, datetime, math, threading
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path

import requests, feedparser
from fastapi import FastAPI
from pydantic import BaseModel

# ===================== ENV =====================
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
USER_AGENT          = os.getenv("USER_AGENT", "Contact <you@example.com>")
SUMMARY_MODEL       = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
TARGET_LANG         = os.getenv("TARGET_LANG", "ko")
TICKER_REFRESH_DAYS = int(os.getenv("TICKER_REFRESH_DAYS", "10"))

HEADERS = {"User-Agent": USER_AGENT}

# ===================== Paths =====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
TICKER_CIK_PATH = DATA_DIR / "company_tickers.json"
RSS_SEEN_PATH   = DATA_DIR / "rss_seen.json"

# ===================== RSS (exactly 4) =====================
RSS_FEEDS = [
    "https://www.prnewswire.com/rss/news-releases-list.rss",
    "https://www.globenewswire.com/RssFeed/country/United%20States/feedTitle/GlobeNewswire%20-%20News%20from%20United%20States",
    "https://www.globenewswire.com/RssFeed/industry/4573-Biotechnology/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Biotechnology",
    "https://www.globenewswire.com/RssFeed/industry/4577-Pharmaceuticals/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Pharmaceuticals",
]

# ===================== SEC cache (auto) =====================
def ensure_ticker_cache():
    """
    Download or refresh SEC company_tickers.json into data/ if missing or stale.
    Uses USER_AGENT header; refresh every TICKER_REFRESH_DAYS.
    """
    path = TICKER_CIK_PATH
    try:
        if path.exists():
            mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.datetime.now() - mtime < datetime.timedelta(days=TICKER_REFRESH_DAYS):
                print(f"[SEC] cache OK: {path} (fresh)")
                return
        print("[SEC] fetching fresh company_tickers.json ...")
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text, encoding="utf-8")
        print(f"[SEC] cache saved: {path}")
    except Exception as e:
        if path.exists():
            print(f"[SEC] refresh failed, using existing cache: {e}")
        else:
            raise

def _load_ticker_rows() -> List[dict]:
    ensure_ticker_cache()
    data = json.loads(TICKER_CIK_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = list(data.values())
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    out = []
    for r in rows:
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:
                continue
        if isinstance(r, dict):
            out.append(r)
    return out

def load_ticker_cik_map() -> Dict[str, str]:
    mp = {}
    for row in _load_ticker_rows():
        t = (row.get("ticker") or "").upper().strip()
        cik = row.get("cik_str") or row.get("cik")
        if not t or cik is None:
            continue
        try:
            mp[t] = f"{int(cik):010d}"
        except Exception:
            continue
    return mp

def normalize_name(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_name_index() -> Dict[str, Tuple[str, str]]:
    idx = {}
    for row in _load_ticker_rows():
        t = (row.get("ticker") or "").upper().strip()
        cik = row.get("cik_str") or row.get("cik")
        title = (row.get("title") or "").strip()
        if not t or not title or cik is None:
            continue
        idx[normalize_name(title)] = (t, f"{int(cik):010d}")
    return idx

# ===================== LICENSE-OUT matcher (regex + embeddings) =====================
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
SIM_THRESHOLD    = float(os.getenv("SIM_THRESHOLD", "0.84"))
TITLE_BONUS      = float(os.getenv("TITLE_BONUS", "0.03"))
MAX_ARTICLE_LEN  = int(os.getenv("MAX_ARTICLE_LEN", "6000"))

_SENT_SPLIT = re.compile(r'(?<=[\.!?])\s+')
LICENSE_GUARD = re.compile(r"\blicen[sc]e|licensing|licensed\b", re.I)
NEGATION_END = [re.compile(p, re.I) for p in [
    r"\bterminated\b|\btermination\b|\bending\b|\bended\b|\bcancel(?:led|lation)?\b",
    r"\bexpired\b|\bexpiry\b|\bno\s+longer\b|\bnot\s+proceeding\b",
]]
IN_EXCLUDE = [re.compile(p, re.I) for p in [
    r"\blicens(?:e|ing|ed)\s+in\b",
    r"\bin[- ]licen(?:s|c)ing\b|\bin[- ]license\b",
    r"\b(acquire[ds]?|obtains?|secured?|receive[sd]?)\s+(an\s+)?(exclusive|non[- ]exclusive|co[- ]exclusive|global|worldwide)?\s*license\s+from\b",
    r"\b(exclusive|non[- ]exclusive|co[- ]exclusive|global|worldwide)?\s*license\s+from\b",
    r"\blicensed\s+from\b",
    r"\boption\s+to\s+license\s+from\b",
    r"\brights?\s+(?:to\s+)?(use|practice)\s+.*\s+from\b",
]]
OUT_POS = [re.compile(p, re.I) for p in [
    r"\blicens(?:e|ing|ed)\s+out\b",
    r"\b(grant(?:ed|s)?|grants)\s+(an\s+)?(exclusive|non[- ]exclusive|co[- ]exclusive|sole|global|worldwide|territorial|regional)?\s*license\s+to\b",
    r"\bhas\s+granted\s+.*\blicense\s+to\b",
    r"\bgranted\s+.*\bexclusive\s+rights?\s+to\s+(develop|commerciali[sz]e|manufacture)\b",
    r"\bentered\s+into\s+(an?\s+)?(exclusive|co[- ]exclusive|global|worldwide|territorial|regional)?\s*(license|licensing)\s+agreement\s+with\b.*\b(grant(?:ing|ed)?|license[d]?)\s+.*\s+to\b",
    r"\b(license|licensing)\s+agreement\s+(covering|for)\s+(global|worldwide|ex[- ]?US|ex[- ]?U\.S\.|territorial|regional|country|market)\s+rights\b.*\bto\b",
    r"\b(grant(?:ed|s)?|provid(?:ed|es)?)\s+(an\s+)?option\s+(?:agreement\s+)?to\s+license\b",
    r"\boption\s+to\s+obtain\s+(an\s+)?(exclusive|co[- ]exclusive|global|worldwide)\s+license\s+to\b",
    r"\b(ex\s*[- ]?(US|U\.S\.|China|Japan|EU|Europe|Korea|KOR|ROW|Rest\s+of\s+World|LATAM|APAC|EMEA|MENA|GCC|ASEAN|CIS|EU5|DACH))\s+rights?\s+(have\s+been\s+)?(licensed|granted)\s+to\b",
    r"\b(territorial|regional|country[- ]specific|field[- ]of[- ]use|FOU)\s+rights?\s+(licensed|granted)\s+to\b",
    r"\bexclusive\s+rights?\s+in\s+(Greater\s+China|Mainland\s+China|Japan|EU|US|Korea|Asia[- ]Pacific|APAC|MENA|EMEA|LATAM|EU5|DACH)\b.*\bto\b",
    r"\bexclusive\s+distribution\s+(and\s+)?license\s+agreement\b",
    r"\bgrant(?:ed|s)?\s+.*\bright\s+to\s+sublicense\b",
    r"\b(right|rights)\s+to\s+sublicense\b",
    r"\blicens(?:e|ing|ed)\s+(the\s+)?(asset|candidate|program|platform|technology|intellectual\s+property|IP|compound|biologic|antibody|gene\s+therapy|cell\s+therapy|small\s+molecule|ADC|siRNA|vaccine)\s+to\b",
]]
LICENSE_OUT_QUERIES = [
    "exclusive license granted to",
    "granted exclusive rights to develop and commercialize",
    "entered into a global licensing agreement granting rights to",
    "out-licensing deal to partner",
    "license agreement for ex-US rights to partner",
    "licensed the asset to partner",
    "global rights licensed to partner",
    "exclusive rights in Greater China licensed to partner",
    "option granted to partner to obtain an exclusive license",
    "territorial licensing agreement granting rights to partner",
    "regional licensing rights granted to partner",
    "field-of-use license granted to partner",
    "right to sublicense granted to partner",
    "non-exclusive license granted to partner",
    "co-exclusive license granted to partner",
    "exclusive manufacturing license granted to partner",
    "exclusive distribution and license agreement with rights granted to partner",
]

_embed_cache = {}
_client = None

def _openai_client():
    global _client
    if _client is None:
        if not OPENAI_API_KEY or OpenAI is None:
            raise RuntimeError("OPENAI_API_KEY not set or openai package missing for embeddings")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def _vec(text: str):
    key = f"emb::{hashlib.sha256((text or '').encode('utf-8')).hexdigest()[:16]}"
    if key in _embed_cache:
        return _embed_cache[key]
    v = _openai_client().embeddings.create(model=EMBEDDING_MODEL, input=[text or ""]).data[0].embedding
    _embed_cache[key] = v
    return v

def _cos(u, v):
    num = sum(a*b for a,b in zip(u,v))
    den = math.sqrt(sum(a*a for a in u))*math.sqrt(sum(b*b for b in v))
    return num/den if den else 0.0

def _normalize(s: str) -> str:
    return (s or "")[:MAX_ARTICLE_LEN]

def _is_lo_sentence(s: str) -> bool:
    if not LICENSE_GUARD.search(s):
        return False
    if any(p.search(s) for p in NEGATION_END):
        return False
    if any(p.search(s) for p in IN_EXCLUDE):
        return False
    return any(p.search(s) for p in OUT_POS)

def is_license_out_doc(title: str, body: str):
    text = f"{title or ''} {body or ''}"
    if any(p.search(text) for p in IN_EXCLUDE):
        return False, {"reason": "in-licensing phrase found"}
    for sent in re.split(r'(?<=[\.!?])\s+', text):
        if _is_lo_sentence(sent):
            return True, {"exact": True, "semantic": False}
    if not LICENSE_GUARD.search(text):
        return False, {"reason": "no license token"}
    t = _normalize(title)
    b = _normalize(body)
    try:
        qvecs = [_vec(q) for q in LICENSE_OUT_QUERIES]
        tvec = _vec(t) if t else None
        bvec = _vec(b) if b else None
        def _max_sim(vec):
            if not vec: return 0.0
            return max(_cos(vec, q) for q in qvecs)
        sim_title = _max_sim(tvec)
        sim_body = _max_sim(bvec)
    except Exception as e:
        return False, {"reason": f"embedding unavailable: {e}"}
    if sim_title + TITLE_BONUS >= SIM_THRESHOLD or sim_body >= SIM_THRESHOLD:
        return True, {"exact": False, "semantic": True, "sim_title": sim_title, "sim_body": sim_body}
    return False, {"reason": "below threshold", "sim_title": sim_title, "sim_body": sim_body}

# ===================== Discord + LLM =====================
def push_discord(title: str, description: str, url: str = ""):
    if not DISCORD_WEBHOOK_URL:
        print("[discord] webhook not set; skipping"); return
    payload = {"username": "News Scanner", "embeds": [{"title": title[:256], "description": description[:4000]}]}
    if url: payload["embeds"][0]["url"] = url
    try: requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e: print(f"[discord] post error: {e}")

def summarize_and_translate(text: str, lang: str = TARGET_LANG) -> str:
    if not OPENAI_API_KEY: return text[:1000]
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"Summarize the following article in {lang}. Keep it concise (5-7 bullets or 4-6 sentences):\\n\\n{text}"
        resp = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[summary_failed] {str(e)[:200]}\\n\\n{text[:1000]}"

# ===================== RSS helpers =====================
def load_rss_seen() -> Set[str]:
    try: return set(json.loads(RSS_SEEN_PATH.read_text(encoding="utf-8")))
    except Exception: return set()

def save_rss_seen(seen: Set[str]):
    try: RSS_SEEN_PATH.write_text(json.dumps(list(seen)), encoding="utf-8")
    except Exception: pass

def rss_entries() -> List[dict]:
    out = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                key = e.get("id") or e.get("link") or e.get("title")
                out.append({
                    "key": key or "",
                    "title": e.get("title") or "",
                    "summary": e.get("summary") or e.get("description") or "",
                    "link": e.get("link") or "",
                })
        except Exception as ex:
            print(f"[rss] fetch error: {url} -> {ex}")
    return out

# ===================== Company detection =====================
TICKER_RE = re.compile(r"""
    (?<![A-Z0-9$])      # not preceded by ticker-like char
    \$?                 # optional leading dollar sign
    ([A-Z]{2,5})         # core ticker: 2-5 uppercase letters
    (?![A-Z0-9])         # not followed by ticker-like char
""", re.VERBOSE)

def pick_companies(text: str, ticker_map: Dict[str,str], name_index: Dict[str,Tuple[str,str]]) -> List[Tuple[str,str]]:
    found: Dict[str, str] = {}
    # 1) Ticker-style matches ($TSLA, TSLA)
    for m in TICKER_RE.finditer((text or "").upper()):
        t = m.group(1)
        if t in ticker_map:
            found[t] = ticker_map[t]
    # 2) Company name substring matches (normalized)
    norm = normalize_name(text or "")
    for name, (t, cik) in name_index.items():
        if len(name) >= 4 and name in norm:
            found[t] = cik
    return [(t, cik) for t, cik in found.items()]

# ===================== Scan cycle & Web =====================
# ===================== Scan cycle & Web =====================
class CycleStats(BaseModel):
    last_started: Optional[str] = None
    last_finished: Optional[str] = None
    last_hits: int = 0
    last_entries: int = 0
    total_hits: int = 0
    total_cycles: int = 0

_stats = CycleStats()

def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def run_cycle():
    global _stats
    _stats.last_started = iso_now()

    ticker_map = load_ticker_cik_map()
    name_index = build_name_index()

    seen = load_rss_seen()
    entries = rss_entries()
    hits = 0
    for e in entries:
        key = e["key"]
        if not key or key in seen:
            continue
        title = e["title"]; body = e["summary"]
        ok, dbg = is_license_out_doc(title, body)
        if not ok:
            continue

        companies = pick_companies(f"{title}\\n{body}", ticker_map, name_index)
        if not companies:
            continue

        hits += 1
        seen.add(key); save_rss_seen(seen)

        text = f"{title}\\n\\n{body}"
        summary = summarize_and_translate(text, TARGET_LANG)
        ticker_line = ", ".join([f"{t} (CIK {c})" for t,c in companies])
        desc = summary + (f"\\n\\n{ticker_line}" if ticker_line else "")
        push_discord(f"[RSS] {title}", desc, e.get("link",""))

    _stats.last_finished = iso_now()
    _stats.last_hits = hits
    _stats.last_entries = len(entries)
    _stats.total_hits += hits
    _stats.total_cycles += 1
    print(f"[cycle] entries={len(entries)}, hits={hits}, seen={len(seen)}")

app = FastAPI(title="RSS→SEC→Summary→Discord")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/status", response_model=CycleStats)
def status():
    return _stats

@app.post("/trigger")
def trigger():
    threading.Thread(target=run_cycle, daemon=True).start()
    return {"ok": True, "detail": "scan scheduled"}

def _bg_loop():
    while True:
        try:
            run_cycle()
        except Exception as e:
            print(f"[bg] error: {e}")
        finally:
            import random
            delay = random.randint(35, 50)
            print(f"[bg] sleeping {delay}s before next cycle")
            time.sleep(delay)

@app.on_event("startup")
def _startup():
    ensure_ticker_cache()
    print(f"[BOOT] Web mode. Random interval 35~50s per cycle. data={TICKER_CIK_PATH}")
    t = threading.Thread(target=_bg_loop, daemon=True)
    t.start()
