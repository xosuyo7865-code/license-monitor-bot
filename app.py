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
SIM_THRESHOLD    = float(os.getenv("SIM_THRESHOLD", "0.78"))
TITLE_BONUS      = float(os.getenv("TITLE_BONUS", "0.02"))
MAX_ARTICLE_LEN  = int(os.getenv("MAX_ARTICLE_LEN", "6000"))

_SENT_SPLIT = re.compile(r'(?<=[\.!?])\s+')

# ---------- False-positive guardrails (supply/purchase/gov only) ----------

# ---------- Intent/Rumor markers (for *_INTENT tagging) ----------
INTENT_TAG_MARKERS = re.compile(
    r"\b(plans?\s+to|intends?\s+to|aims?\s+to|seeks?\s+to|in\s+talks\s+to|negotiat(?:e|ing)\s+to|under\s+discussion|reportedly|rumored|according\s+to\s+sources)\b",
    re.I
)

def _has_intent_marker(s: str) -> bool:
    return bool(INTENT_TAG_MARKERS.search(s or ""))
YEAR_OLD = re.compile(r"\b(20\d{2}|19\d{2})\b")
BACKGROUND_MARKERS = re.compile(r"\b(previously|back\s+in|in\s+\d{4}|as\s+of\s+\d{4}|historically|earlier|formerly|prior\s+to)\b", re.I)
INTENT_MARKERS = re.compile(r"\b(intends?\s+to|plans?\s+to|aims?\s+to|seeks?\s+to|in\s+talks\s+to|negotiat(?:e|ing)\s+to|explor(?:e|ing)\s+an?\s+agreement)\b", re.I)
RUMOR_MARKERS = re.compile(r"\b(report(?:ed|ing)ly|according\s+to\s+sources|rumors?)\b", re.I)

def _recent_enough(sentence: str, current_year: int) -> bool:
    years = [int(y) for y in YEAR_OLD.findall(sentence or "")]
    return (not years) or (max(years) >= current_year - 1)

def _not_background(sentence: str) -> bool:
    return not BACKGROUND_MARKERS.search(sentence or "")

def _not_intent_or_rumor(sentence: str) -> bool:
    return not (INTENT_MARKERS.search(sentence or "") or RUMOR_MARKERS.search(sentence or ""))

def _lead_sentences(text: str, k: int = 2):
    sents = re.split(_SENT_SPLIT, text or "")
    return [s for s in sents[:k] if s and s.strip()]
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

# ===================== SUPPLY / PURCHASE / GOV-CONTRACT (regex + embeddings) =====================
GOV_ENTITY = re.compile(r"""
    \b(
        (U\.?S\.?\s+)?(Federal|State|City|County)\s+Government|Government\b|Govt\b|
        Department\s+of\s+\w+|Ministry\s+of\s+\w+|
        DoD|Pentagon|US\s+Army|US\s+Navy|US\s+Air\s+Force|
        NASA|NIH|BARDA|DARPA|HHS\b|DOE\b|DoE\b|DHS|DOJ|DOT|FDA|EMA|NHS|GSA|
        European\s+Commission|EC\b|EU\b|UK\s+MoD|HM\s+Government|Home\s+Office|Treasury|
        (Republic|Kingdom)\s+of\s+\w+|Korea\s+Government|Korean\s+Government|
        Ministry\s+of\s+Defense|Ministry\s+of\s+Defence|Ministry\s+of\s+Health|Ministry\s+of\s+Industry|
        국방부|정부|산업통상자원부|보건복지부|과학기술정보통신부|조달청
    )\b
""", re.I | re.X)

NEGATION_GENERIC = [re.compile(p, re.I) for p in [
    r"\bterminated\b|\btermination\b|\bending\b|\bended\b|\bcancel(?:led|lation)?\b",
    r"\bexpired\b|\bexpiry\b|\bnot\s+proceeding\b|\bnon[- ]binding\b|\bnonbinding\b|\bMOU\b",
    r"\bletter\s+of\s+intent\b|\bLOI\b",
]]

SUPPLY_POS = [re.compile(p, re.I) for p in [
    r"\b(supply|supplies|supplying)\s+(agreement|contract|deal)\b",
    r"\b(supplier|supply)\s+agreement\s+with\b",
    r"\bentered\s+into\s+(a[n]?\s+)?(long[- ]term|multi[- ]year)?\s*supply\s+(agreement|contract|deal)\b",
    r"\bframework\s+supply\s+agreement\b",
    r"\b(master|strategic)\s+supply\s+(agreement|contract|deal)\b",
    r"\bsupply\s+and\s+distribution\s+agreement\b",
    r"\b(awarded|wins?|win)\s+(an?\s+)?supply\s+contract\b",
    r"\baward(?:ed)?\s+of\s+contract\b",
    r"\bcontract\s+award\b",
    r"\bsupply\s+deal(\s+with)?\b",
    r"\bcall[- ]off\s+(contract|order)s?\b",
    r"\bblanket\s+purchase\s+agreement\b",
    r"\bGSA\s+Schedule\s+(contract|award)\b",
    r"\bframework\s+agreement\s+for\s+(supply|procurement|purchase)\b",
]]

PURCHASE_POS = [re.compile(p, re.I) for p in [
    r"\b(purchase|procurement)\s+(agreement|contract|order|orders|po|deal)\b",
    r"\b(awarded|wins?|win)\s+(an?\s+)?(purchase|procurement)\s+contract\b",
    r"\bentered\s+into\s+(a[n]?\s+)?(purchase|procurement)\s+(agreement|contract|deal)\b",
    r"\b(IDIQ|indefinite[- ]delivery|indefinite[- ]quantity)\s+(contract|award)\b",
    r"\bframework\s+purchase\s+agreement\b",
    r"\breceived\s+purchase\s+orders?\s+under\s+(a\s+)?master\s+supply\s+agreement\b",
    r"(?<!I)\bPOs?\b",
    r"\bpurchase\s+deal(\s+with)?\b",
    r"\bprocurement\s+deal\b",
    r"\bblanket\s+purchase\s+agreement\b",
    r"\bcall[- ]off\s+(contract|order)s?\b",
]]

SUPPLY_QUERIES = [
    "entered into a long-term supply agreement with partner",
    "signed a multi-year supply contract for components",
    "signed a framework supply agreement with a partner",
    "strategic supply contract for large-scale manufacturing",
    "supply and distribution agreement with partner",
    "received purchase orders under a master supply agreement",
    "entered into a master supply agreement with a global OEM partner",
    "signed a supplier agreement with Company B",
    "entered into a framework agreement for the supply of components",
    "contract award for supply of materials",
    "awarded a supply contract by the Ministry of Defense",
    "supply deal with multinational manufacturer",
]

PURCHASE_QUERIES = [
    "entered into a purchase agreement with another company",
    "awarded a purchase contract by a corporation",
    "signed a procurement contract with an industry partner",
    "wins IDIQ procurement contract from a corporate buyer",
    "entered into a framework purchase agreement",
    "received purchase orders under a master supply agreement",
    "purchase order under a master agreement",
    "call-off contract under a framework agreement",
    "blanket purchase agreement with a key customer",
    "procurement deal signed with a strategic partner",
]

GOV_CONTRACT_QUERIES = [
    "awarded a government procurement contract",
    "wins IDIQ contract from the U.S. Army",
    "awarded a purchase contract by the Department of Defense",
    "signed a supply contract with the Ministry of Defense",
    "entered into a procurement agreement with the Ministry of Health",
    "awarded a supply contract by the European Commission",
    "signed a strategic supply deal with the Government of Korea",
    "entered into a supply agreement with the U.S. Department of Energy",
    "government contract award to Company A for production and supply",
    "awarded a contract for manufacturing of systems",
]

def _has_any(text: str, patterns):
    return any(p.search(text) for p in patterns)

def _no_negation(text: str):
    return not any(p.search(text) for p in NEGATION_GENERIC)

def _max_sim(vec, qvecs):
    if not vec: return 0.0
    return max(_cos(vec, q) for q in qvecs)

def is_supply_or_purchase_doc(title: str, body: str):
    """
    Returns (matched: bool, info: dict)
    category in {"corp_supply","corp_purchase","gov_supply","gov_purchase"}
    Guardrails applied: recency, background/intent/rumor exclusion, stronger later-paragraph requirements.
    """
    text = f"{title or ''} {body or ''}"
    if not _no_negation(text):
        return False, {"reason": "negation found"}

    import datetime
    current_year = datetime.datetime.now().year

    # 1) Title + lead sentences (strict recency & intent guards)
    for s in [x for x in [title] + _lead_sentences(body, k=2) if x]:
        if _not_background(s) and _recent_enough(s, current_year):
            is_supply = _has_any(s, SUPPLY_POS)
            is_purchase = _has_any(s, PURCHASE_POS)
            if is_supply or is_purchase:
                if GOV_ENTITY.search(s):
                    cat = "gov_supply" if is_supply else "gov_purchase"
                else:
                    cat = "corp_supply" if is_supply else "corp_purchase"
                if _has_intent_marker(s):
                    cat = f"{cat}_intent"
                return True, {"category": cat, "exact": True, "semantic": False, "zone": "title/lead"}

    # 2) Later paragraphs: require company or value cues
    for s in re.split(_SENT_SPLIT, body or ""):
        if not s or not s.strip():
            continue
        if not (_not_background(s) and _recent_enough(s, current_year)):
            continue
        is_supply = _has_any(s, SUPPLY_POS)
        is_purchase = _has_any(s, PURCHASE_POS)
        if is_supply or is_purchase:
            has_company = bool(TICKER_RE.search(s)) or bool(re.search(r"\b(Inc\.|Corp\.|PLC|Ltd\.|LLC|S\.?A\.|Co\.)\b", s))
            has_value   = bool(re.search(r"\$\s?\d+(?:\.\d+)?\s?(?:M|B|million|billion)", s, re.I))
            if has_company or has_value:
                if GOV_ENTITY.search(s):
                    cat = "gov_supply" if is_supply else "gov_purchase"
                else:
                    cat = "corp_supply" if is_supply else "corp_purchase"
                if _has_intent_marker(s):
                    cat = f"{cat}_intent"
                return True, {"category": cat, "exact": True, "semantic": False, "zone": "body_strict"}

    # 3) Embedding fallback (guarded by background/intent)
    try:
        tvec = _vec(_normalize(title)) if title else None
        bvec = _vec(_normalize(body)) if body else None

        q_supply = [_vec(q) for q in SUPPLY_QUERIES]
        q_purchase = [_vec(q) for q in PURCHASE_QUERIES]
        q_gov = [_vec(q) for q in GOV_CONTRACT_QUERIES]

        def _max_sim(vec, qs):
            if not vec: return 0.0
            return max(_cos(vec, q) for q in qs)

        sim_title_supply = _max_sim(tvec, q_supply)
        sim_body_supply  = _max_sim(bvec, q_supply)
        sim_title_purch  = _max_sim(tvec, q_purchase)
        sim_body_purch   = _max_sim(bvec, q_purchase)
        sim_title_gov    = _max_sim(tvec, q_gov)
        sim_body_gov     = _max_sim(bvec, q_gov)

        TH = max(SIM_THRESHOLD, 0.78)
        body_guard = _not_background(body or "")

        sims = {
            "supply": max(sim_body_supply, sim_title_supply),
            "purchase": max(sim_body_purch, sim_title_purch),
            "gov": max(sim_body_gov, sim_title_gov),
        }
        best_label, best_sim = max(sims.items(), key=lambda x: x[1])

        if best_sim >= TH and body_guard:
            is_gov = (best_label == "gov") or bool(GOV_ENTITY.search(text))
            if is_gov:
                cat = "gov_supply" if sims["supply"] >= sims["purchase"] else "gov_purchase"
            else:
                cat = "corp_supply" if sims["supply"] >= sims["purchase"] else "corp_purchase"
            if _has_intent_marker(title or "") or _has_intent_marker(body or ""):
                cat = f"{cat}_intent"
            return True, {
                "category": cat,
                "exact": False, "semantic": True,
                "sim_title": max(sim_title_supply, sim_title_purch, sim_title_gov),
                "sim_body":  max(sim_body_supply,  sim_body_purch,  sim_body_gov),
                "zone": "embedding",
            }
        return False, {"reason": "below threshold",
                       "sim_title": max(sim_title_supply, sim_title_purch, sim_title_gov),
                       "sim_body":  max(sim_body_supply,  sim_body_purch,  sim_body_gov)}
    except Exception as e:
        return False, {"reason": f"embedding unavailable: {e}"}

def _openai_client():
    global _client
    if _client is None:
        if not OPENAI_API_KEY or OpenAI is None:
            raise RuntimeError("OPENAI_API_KEY not set or openai package missing for embeddings")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

_embed_cache: Dict[str, List[float]] = {}
_client = None

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
def push_discord(title: str, description: str, url: str = "", published_dt_utc: datetime.datetime | None = None):
    if not DISCORD_WEBHOOK_URL:
        print("[discord] webhook not set; skipping")
        return
    from zoneinfo import ZoneInfo
    payload = {"username": "News Scanner", "embeds": [{"title": title[:256], "description": description[:4000]}]}
    if url:
        payload["embeds"][0]["url"] = url
    # Footer time (ET + KST) if available
    if published_dt_utc is not None:
        try:
            et = published_dt_utc.astimezone(ZoneInfo("America/New_York"))
            kst = published_dt_utc.astimezone(ZoneInfo("Asia/Seoul"))
            footer_text = (
                f"Published: {et.strftime('%Y-%m-%d %H:%M')} ET "
                f"({kst.strftime('%Y-%m-%d %H:%M')} KST)"
            )
        except Exception:
            # Fallback to UTC representation
            footer_text = f"Published (UTC): {published_dt_utc.strftime('%Y-%m-%d %H:%M UTC')}"
        payload["embeds"][0]["footer"] = {"text": footer_text}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"[discord] post error: {e}")

def summarize_and_translate(text: str, lang: str = TARGET_LANG) -> str:
    if not OPENAI_API_KEY: return text[:1000]
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"Summarize the following article in {lang}. Keep it concise (5-7 bullets or 4-6 sentences):\n\n{text}"
        resp = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[summary_failed] {str(e)[:200]}\n\n{text[:1000]}"

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
                # Determine published datetime (UTC) if present
                published_dt = None
                try:
                    t = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
                    if t:
                        published_dt = datetime.datetime(*t[:6], tzinfo=datetime.timezone.utc)
                except Exception:
                    published_dt = None
                out.append({
                    "key": key or "",
                    "title": e.get("title") or "",
                    "summary": e.get("summary") or e.get("description") or "",
                    "link": e.get("link") or "",
                    "published_dt": published_dt.isoformat() if published_dt else "",
                })
        except Exception as ex:
            print(f"[rss] fetch error: {url} -> {ex}")
    return out

# ===================== HTML full-text fetch & extract =====================
from urllib.parse import urljoin

def _clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _bs4_parser() -> str:
    """Prefer html5lib for robustness; fallback to built-in html.parser if missing."""
    try:
        import html5lib  # noqa: F401
        return "html5lib"
    except Exception:
        return "html.parser"

def fetch_article_text(url: str, timeout: int = 20) -> str:
    """Fetch article HTML and extract readable text. Falls back gracefully if libs missing.
    Priority: readability-lxml > <article>/<main> paragraphs > all paragraphs.
    """
    if not url:
        return ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"[fetch] failed {url}: {e}")
        return ""

    # Try readability-lxml if available
    try:
        from readability import Document  # type: ignore
        doc = Document(html)
        content_html = doc.summary(html_partial=True) or ""
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(content_html, _bs4_parser())
            text = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all(["p","li"]))
            return _clean_text(text)[:MAX_ARTICLE_LEN]
        except Exception:
            # crude fallback: strip tags
            text = re.sub(r"<[^>]+>", " ", content_html)
            return _clean_text(text)[:MAX_ARTICLE_LEN]
    except Exception:
        pass

    # Fallback: BeautifulSoup heuristic
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, _bs4_parser())
        # Remove scripts/styles
        for t in soup(["script","style","noscript"]):
            t.decompose()
        # Prefer <article> or <main>
        cand = soup.find("article") or soup.find("main") or soup
        paras = cand.find_all("p")
        if not paras:
            paras = soup.find_all("p")
        text = "\n".join(p.get_text(" ", strip=True) for p in paras)
        return _clean_text(text)[:MAX_ARTICLE_LEN]
    except Exception as e:
        print(f"[extract] bs4 failed for {url}: {e}")
        return ""

# ===================== Company detection =====================
TICKER_RE = re.compile(r"(?<![A-Z])\$?([A-Z]{1,5})(?![A-Z])")  # fixed: use \$? (not \\$?)

def pick_companies(text: str, ticker_map: Dict[str,str], name_index: Dict[str,Tuple[str,str]]) -> List[Tuple[str,str]]:
    found = {}
    for m in TICKER_RE.finditer(text.upper()):
        t = m.group(1)
        if t in ticker_map:
            found[t] = ticker_map[t]
    norm = normalize_name(text)
    for name, (t, cik) in name_index.items():
        if len(name) >= 4 and name in norm:
            found[t] = cik
    return [(t, cik) for t, cik in found.items()]

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
        title = e["title"]
        body_summary = e["summary"]
        link = e.get("link", "")

        # === NEW: fetch full article body ===
        body_full = fetch_article_text(link) if link else ""
        body_for_detection = body_full or body_summary

        ok, dbg = is_license_out_doc(title, body_for_detection)
        sp_ok, sp_dbg = is_supply_or_purchase_doc(title, body_for_detection)
        if not ok and not sp_ok:
            continue

        companies = pick_companies(f"{title}\n{body_for_detection}", ticker_map, name_index)
        if ok and not companies:
            continue
        if sp_ok and sp_dbg.get("category","{}").startswith("corp") and not companies:
            continue

        hits += 1
        seen.add(key); save_rss_seen(seen)

        # For LLM summary, prefer full text if available
        text_for_summary = f"{title}\n\n{body_full or body_summary}"
        summary = summarize_and_translate(text_for_summary, TARGET_LANG)
        ticker_line = ", ".join([f"{t} (CIK {c})" for t,c in companies])
        desc = summary + (f"\n\n{ticker_line}" if ticker_line else "")
        tag = "LICENSE" if ok else sp_dbg.get("category","{}").upper()
        # Convert ISO back to aware datetime
        pub_iso = e.get("published_dt")
        pub_dt = datetime.datetime.fromisoformat(pub_iso) if pub_iso else None
        push_discord(f"[RSS][{tag}] {title}", desc, link, published_dt_utc=pub_dt)

    _stats.last_finished = iso_now()
    _stats.last_hits = hits
    _stats.last_entries = len(entries)
    _stats.total_hits += hits
    _stats.total_cycles += 1
    print(f"[cycle] entries={len(entries)}, hits={hits}, seen={len(seen)})")

app = FastAPI(title="RSS→SEC→Summary→Discord (FullText)")

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
    print(f"[BOOT] Web mode (FullText). Random interval 35~50s per cycle. data={TICKER_CIK_PATH}")
    t = threading.Thread(target=_bg_loop, daemon=True)
    t.start()
