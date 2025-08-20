import asyncio
import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Set, Optional, Tuple

import google.generativeai as genai
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

# fix cors problems
from fastapi.middleware.cors import CORSMiddleware

# NEW: lightweight scraping
try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# -----------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Gemini API configuration
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logger.critical(
        "Erreur critique: la variable d'environnement GEMINI_API_KEY n'est pas définie."
    )
    raise SystemExit(
        "GEMINI_API_KEY non définie. Veuillez définir cette variable d'environnement et redémarrer."
    )

# Default Gemini model
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# -----------------------------------------------------------------------------
# File paths and defaults
SOURCES_FILE: str = os.getenv("SOURCES_FILE", "sources.json")
MEMORY_FILE: str = os.getenv("MEMORY_FILE", "memory_db.json")

DEFAULT_MEMORY: Dict[str, Any] = {
    "seen_urls": [],
    "details": {},     # url -> {"title": str, "summary": str, "text": str}
    "reports": [],
}

# -----------------------------------------------------------------------------
# Pydantic models
class SourceConfig(BaseModel):
    veille_par_sujet: List[str] = []
    veille_par_url: List[str] = []

class UpdateSourcesRequest(BaseModel):
    add_subjects: Optional[List[str]] = None
    add_urls: Optional[List[str]] = None
    remove_subjects: Optional[List[str]] = None
    remove_urls: Optional[List[str]] = None
    replace: Optional[SourceConfig] = None

# -----------------------------------------------------------------------------
# Memory helpers
def safe_load_memory(path: str = MEMORY_FILE) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "seen_urls" not in data:
            logger.warning(
                f"Format de fichier mémoire invalide pour '{path}'. Réinitialisation de la mémoire."
            )
            return DEFAULT_MEMORY.copy()
        memory: Dict[str, Any] = {
            "seen_urls": data.get("seen_urls", []),
            "details": data.get("details", {}),
            "reports": data.get("reports", []),
        }
        if not isinstance(memory["seen_urls"], list):
            logger.warning(f"'seen_urls' n'est pas une liste. Reset.")
            return DEFAULT_MEMORY.copy()
        if not isinstance(memory["details"], dict):
            logger.warning(f"'details' n'est pas un objet. Reset.")
            return DEFAULT_MEMORY.copy()
        if not isinstance(memory["reports"], list):
            logger.warning(f"'reports' n'est pas une liste. Reset.")
            return DEFAULT_MEMORY.copy()
        return memory
    except FileNotFoundError:
        logger.info(f"Fichier mémoire '{path}' non trouvé. Initialisation.")
        return DEFAULT_MEMORY.copy()
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Lecture mémoire '{path}' impossible: {e}. Reset.")
        return DEFAULT_MEMORY.copy()

def atomic_save_memory(memory: Dict[str, Any], path: str = MEMORY_FILE) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    mem_copy = memory.copy()
    mem_copy["seen_urls"] = sorted(set(mem_copy.get("seen_urls", [])))
    fd, temp_path = tempfile.mkstemp(prefix=".memory_tmp_", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(mem_copy, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(temp_path, path)
        logger.info(
            f"Sauvegarde mémoire OK. URLs: {len(mem_copy['seen_urls'])}."
        )
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass

# -----------------------------------------------------------------------------
# Scraping helpers (NEW)
USER_AGENT = (
    "Mozilla/5.0 (compatible; WatcherBot/1.0; +https://example.com/bot) "
    "PythonRequests"
)

def _clean_text(text: str) -> str:
    # Normalize whitespace, remove extra blank lines
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()

def extract_main_text(html: str) -> Tuple[str, str]:
    """
    Extract (title, visible_text) from HTML using BeautifulSoup if available.
    If BeautifulSoup is missing, return a rough fallback.
    """
    if not html:
        return ("", "")
    if BeautifulSoup is None:
        # crude fallback: strip tags
        title_match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
        title = title_match.group(1).strip() if title_match else ""
        # remove tags
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return (title, _clean_text(text))

    soup = BeautifulSoup(html, "html.parser")

    # remove scripts/styles/nav/footer
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for sel in ["nav", "footer", "header", "form", "aside"]:
        for t in soup.select(sel):
            t.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        # try meta og:title
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = og["content"].strip()

    # prefer common article containers if present
    candidates = soup.select("article") or soup.select("main") or [soup.body or soup]
    # join text from candidates
    chunks: List[str] = []
    for node in candidates:
        text = node.get_text(separator="\n", strip=True)
        if text:
            chunks.append(text)
    text = "\n\n".join(chunks) if chunks else soup.get_text(separator="\n", strip=True)
    return (title, _clean_text(text))

def fetch_url_text(url: str, timeout: int = 12) -> Tuple[str, str]:
    """
    Returns (title, main_text). Empty strings if requests isn't available or fetch failed.
    """
    if requests is None:
        logger.warning("Le module 'requests' n'est pas installé. Impossible de scraper.")
        return ("", "")
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=timeout,
        )
        if resp.status_code >= 400:
            logger.warning(f"Échec HTTP {resp.status_code} pour {url}")
            return ("", "")
        html = resp.text
        return extract_main_text(html)
    except Exception as e:
        logger.warning(f"Erreur réseau pour {url}: {e}")
        return ("", "")

# -----------------------------------------------------------------------------
# Gemini helpers
def call_gemini_with_retry(
    prompt: Optional[str] = None,
    *,
    contents: Optional[List[Any]] = None,
    max_retries: int = 3,
    initial_delay: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """
    Call Gemini with retry. Provide either `prompt` (string) OR `contents` (list of parts).
    Returns plain text (str) or "" on failure.
    """
    model = genai.GenerativeModel(model_name=model_name)
    for attempt in range(max_retries):
        try:
            if contents is not None:
                response = model.generate_content(contents)
            elif prompt is not None:
                response = model.generate_content(prompt)
            else:
                raise ValueError("Either `prompt` or `contents` must be provided.")

            # standardize text extraction
            if getattr(response, "text", None):
                return response.text.strip()
            if getattr(response, "candidates", None):
                cand = response.candidates[0]
                text = getattr(cand, "text", "") or getattr(cand, "content", "")
                if text:
                    return str(text).strip()
            logger.warning(f"Réponse vide de Gemini (tentative {attempt+1}/{max_retries})")
        except Exception as e:
            logger.error(f"Erreur Gemini (tentative {attempt+1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            sleep_time = initial_delay * (2 ** attempt)
            logger.info(f"Nouvelle tentative dans {sleep_time} secondes...")
            time.sleep(sleep_time)
    return ""

def summarize_with_url_context(url: str, scraped_text: str) -> str:
    """
    Try Gemini URL Context first. If not supported or fails, fall back to summarizing the scraped text.
    """
    # 1) Try URL Context (per https://ai.google.dev/gemini-api/docs/url-context)
    # Newer SDKs accept a "url" part inside a content list, like:
    # contents = [{"role": "user", "parts": [{"text": "...prompt..."}, {"url": url}]}]
    url_prompt = (
        "Analyse et résume précisément en français le contenu de cette URL. "
        "Mets en avant les mises à jour, nouvelles informations et points clés. "
        "Structure la réponse avec des puces claires et un court paragraphe de synthèse à la fin."
    )
    try:
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": url_prompt},
                    {"url": url},  # URL Context
                ],
            }
        ]
        text = call_gemini_with_retry(contents=contents)
        if text:
            return text
    except Exception as e:
        logger.info(f"URL Context non disponible ou a échoué pour {url}: {e}")

    # 2) Fallback: summarize scraped text if available
    if scraped_text:
        fallback_prompt = (
            "Voici le contenu d'une page web. Résume-le en français, en listant d'abord les points clés, "
            "puis une synthèse courte et actionnable.\n\n"
            f"CONTENU:\n{scraped_text[:15000]}"  # keep a safe token budget
        )
        return call_gemini_with_retry(prompt=fallback_prompt) or "Aucune description disponible pour cette URL."
    # 3) Last resort
    return "Aucune description disponible pour cette URL."

# -----------------------------------------------------------------------------
# Veille logic
async def perform_watch_task() -> None:
    logger.info("Tâche de veille démarrée.")
    try:
        with open(SOURCES_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = SourceConfig(**config_data)
    except FileNotFoundError:
        logger.error(f"Fichier de configuration '{SOURCES_FILE}' introuvable. Veille annulée.")
        return
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Config '{SOURCES_FILE}' invalide: {e}. Veille annulée.")
        return

    subjects_to_watch = config.veille_par_sujet
    urls_to_watch = config.veille_par_url

    memory = safe_load_memory()
    seen_urls_set: Set[str] = set(memory.get("seen_urls", []))
    details: Dict[str, Any] = memory.get("details", {})
    new_urls: Set[str] = set()
    new_details: Dict[str, Any] = {}

    # Step 1: Find URLs from subjects (Gemini)
    for subject in subjects_to_watch:
        logger.info(f"Analyse du sujet: '{subject}'")
        prompt = (
            f"Liste jusqu'à 5 URLs pertinentes (http/https) pour ce sujet: {subject}\n"
            "Retourne uniquement les URLs séparées par des espaces ou sauts de ligne, sans texte additionnel."
        )
        response_text = call_gemini_with_retry(prompt=prompt)
        if not response_text:
            logger.info(f"Aucune réponse pour le sujet '{subject}'.")
            continue
        urls = re.findall(r"https?://\S+", response_text)
        cleaned_urls = [u.rstrip('.,);') for u in urls]
        for url in cleaned_urls:
            if url in seen_urls_set or url in new_urls:
                continue
            new_urls.add(url)
            logger.info(f"Nouvelle URL détectée pour '{subject}': {url}")

    # Step 2: Add explicit URLs from config
    for url in urls_to_watch:
        if url in seen_urls_set or url in new_urls:
            logger.info(f"URL déjà connue, ignorée: '{url}'")
            continue
        new_urls.add(url)
        logger.info(f"Nouvelle URL depuis la configuration: '{url}'")

    # Step 3: For each newly discovered URL: SCRAPE + SUMMARIZE with URL Context
    for url in new_urls:
        title, text = fetch_url_text(url)
        summary = summarize_with_url_context(url, text)

        new_details[url] = {
            "title": title or "",
            "summary": summary or "",
            "text": text or "",
        }
        logger.info(f"Résumé généré pour '{url}' (title='{title or 'N/A'}').")

    # Step 4: Build a synthetic report of all new items
    report_text = ""
    if new_details:
        lines = []
        for url, info in new_details.items():
            t = info.get("title") or ""
            s = info.get("summary") or ""
            lines.append(f"- {t or 'Sans titre'}\n{url}\n{s}\n")
        report_prompt = (
            "Rédige un rapport synthétique (en français) sur les nouveautés suivantes. "
            "Pour chaque URL, liste 3-6 points clés, puis termine par une synthèse globale et 3 recommandations actionnables.\n\n"
            + "\n".join(lines)
        )
        report = call_gemini_with_retry(prompt=report_prompt)
        report_text = report or "\n".join(lines)

    # Step 5: Persist
    if new_urls:
        memory_seen = set(memory.get("seen_urls", []))
        memory_seen.update(new_urls)
        memory["seen_urls"] = list(memory_seen)

        merged_details = memory.get("details", {})
        merged_details.update(new_details)
        memory["details"] = merged_details

        if report_text:
            import datetime
            report_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "new_urls": list(new_urls),
                "report": report_text,
            }
            reports_list = memory.get("reports", [])
            reports_list.append(report_entry)
            memory["reports"] = reports_list

        atomic_save_memory(memory)
    else:
        logger.info("Aucune nouvelle URL à sauvegarder.")
    logger.info("Tâche de veille terminée.")

# -----------------------------------------------------------------------------
# FastAPI app
app = FastAPI(
    title="Watcher API v2 (with Scraping & URL Context)",
    description="API de veille stratégique avec scraping basique et contexte URL pour Gemini.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/watch", summary="Déclenche la veille en tâche de fond")
async def trigger_watch_endpoint(background_tasks: BackgroundTasks) -> Dict[str, str]:
    background_tasks.add_task(perform_watch_task)
    logger.info("Requête de veille reçue. Tâche programmée en arrière-plan.")
    return {
        "message": "La veille a été lancée en arrière-plan. Consultez les logs pour les détails."
    }

@app.get("/memory", summary="Affiche la mémoire complète")
async def get_memory_content() -> Dict[str, Any]:
    return safe_load_memory()

@app.get("/", summary="Endpoint de santé")
async def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Watcher API is operational."}

# Sources management
@app.get("/sources", summary="Lire la configuration des sources")
async def read_sources() -> Dict[str, Any]:
    try:
        with open(SOURCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = SourceConfig(**data)
        return config.dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Le fichier '{SOURCES_FILE}' est introuvable.")
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=500, detail=f"Le fichier '{SOURCES_FILE}' est invalide: {e}")

@app.post("/sources", summary="Modifier la configuration des sources")
async def update_sources(update: UpdateSourcesRequest) -> Dict[str, Any]:
    try:
        if os.path.exists(SOURCES_FILE):
            with open(SOURCES_FILE, "r", encoding="utf-8") as f:
                current_data = json.load(f)
            current_config = SourceConfig(**current_data)
        else:
            current_config = SourceConfig()
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=500, detail=f"Le fichier '{SOURCES_FILE}' est invalide: {e}")

    if update.replace is not None:
        new_config = update.replace
    else:
        new_config = SourceConfig(
            veille_par_sujet=list(current_config.veille_par_sujet),
            veille_par_url=list(current_config.veille_par_url),
        )
        if update.add_subjects:
            for subj in update.add_subjects:
                if subj not in new_config.veille_par_sujet:
                    new_config.veille_par_sujet.append(subj)
        if update.add_urls:
            for url in update.add_urls:
                if url not in new_config.veille_par_url:
                    new_config.veille_par_url.append(url)
        if update.remove_subjects:
            new_config.veille_par_sujet = [s for s in new_config.veille_par_sujet if s not in update.remove_subjects]
        if update.remove_urls:
            new_config.veille_par_url = [u for u in new_config.veille_par_url if u not in update.remove_urls]

    try:
        with open(SOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump(new_config.dict(), f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'écriture de '{SOURCES_FILE}': {e}")
    return new_config.dict()

# Details & reports
@app.get("/details", summary="Consulter les descriptions enregistrées")
async def get_details() -> Dict[str, Any]:
    memory = safe_load_memory()
    return memory.get("details", {})

@app.get("/reports", summary="Consulter l'historique des rapports générés")
async def get_reports() -> List[Dict[str, Any]]:
    memory = safe_load_memory()
    return memory.get("reports", [])
