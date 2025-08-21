import asyncio
import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Set, Optional

import google.generativeai as genai
from google.generativeai.types import Tool
from google.generativeai.types.generation_types import GenerationConfig
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

# fix cors problems
from fastapi.middleware.cors import CORSMiddleware


# -----------------------------------------------------------------------------
# Logging configuration
#
# Configure the root logger to emit timestamped messages.  This helps when
# debugging asynchronous background tasks, as their logs may appear interleaved
# with request logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Gemini API configuration
#
# The Gemini client library reads the API key from an environment variable.  If
# the variable is not set, we log a critical error and exit.  We configure
# Gemini once at import time so that model initialisation in `call_gemini`
# happens quickly.
try:
    # genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    genai.configure(api_key="AIzaSyBXhRRf9eMJE1T6p_UtzWLfpylD1FAIyEk")
except KeyError:
    logger.critical(
        "Erreur critique: la variable d'environnement GEMINI_API_KEY n'est pas définie."
    )
    # Abort import if the API key is missing.  This prevents the FastAPI app
    # from starting in a misconfigured state.
    raise SystemExit(
        "GEMINI_API_KEY non définie. Veuillez définir cette variable d'environnement et redémarrer."
    )


# -----------------------------------------------------------------------------
# File paths and defaults
#
# By default the module reads configuration from `sources.json` and persists
# discovered URLs into `memory_db.json`.  These filenames can be overridden
# via environment variables if necessary (e.g. for testing).
SOURCES_FILE: str = os.getenv("SOURCES_FILE", "sources.json")
MEMORY_FILE: str = os.getenv("MEMORY_FILE", "memory_db.json")

# Default structure for an empty memory file.  Using a dict makes it trivial
# to extend the schema later if needed.  In addition to the list of seen URLs,
# we now store a mapping of URL -> detailed description and a list of reports.
DEFAULT_MEMORY: Dict[str, Any] = {
    "seen_urls": [],       # List of all URLs that have been processed
    "details": {},         # Mapping from URL to its detailed description
    "reports": []          # History of generated reports with timestamps
}


# -----------------------------------------------------------------------------
# Pydantic model for the sources configuration
#
# This model enforces that `veille_par_sujet` and `veille_par_url` are lists of
# strings.  If additional keys are present in the JSON, they will be ignored
# unless explicitly defined here.
class SourceConfig(BaseModel):
    veille_par_sujet: List[str] = []
    veille_par_url: List[str] = []


# Request model for updating the sources configuration.  Clients can either
# specify a complete replacement via the `replace` field or indicate lists of
# subjects/URLs to add or remove.  At least one of the fields must be
# provided.
class UpdateSourcesRequest(BaseModel):
    add_subjects: Optional[List[str]] = None
    add_urls: Optional[List[str]] = None
    remove_subjects: Optional[List[str]] = None
    remove_urls: Optional[List[str]] = None
    replace: Optional[SourceConfig] = None


# -----------------------------------------------------------------------------
# Memory management helpers
#
def safe_load_memory(path: str = MEMORY_FILE) -> Dict[str, Any]:
    """Load the persisted memory from disk.

    If the file is missing, empty, corrupt, or does not match the expected
    schema, a fresh copy of ``DEFAULT_MEMORY`` is returned and a log message
    is emitted.  The returned value always contains the keys ``seen_urls``,
    ``details``, and ``reports``.

    Args:
        path: The path to the memory JSON file.
    Returns:
        A dictionary with keys ``seen_urls`` (list of str), ``details`` (dict
        mapping str to str), and ``reports`` (list of dicts).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "seen_urls" not in data:
            logger.warning(
                f"Format de fichier mémoire invalide pour '{path}'. Réinitialisation de la mémoire."
            )
            return DEFAULT_MEMORY.copy()
        # Assemble a full memory dict, filling in missing keys with defaults.
        memory: Dict[str, Any] = {
            "seen_urls": data.get("seen_urls", []),
            "details": data.get("details", {}),
            "reports": data.get("reports", []),
        }
        # Validate types: seen_urls must be list, details must be dict, reports must be list
        if not isinstance(memory["seen_urls"], list):
            logger.warning(
                f"Le champ 'seen_urls' dans '{path}' n'est pas une liste. Réinitialisation de la mémoire."
            )
            return DEFAULT_MEMORY.copy()
        if not isinstance(memory["details"], dict):
            logger.warning(
                f"Le champ 'details' dans '{path}' n'est pas un objet. Réinitialisation de la mémoire."
            )
            return DEFAULT_MEMORY.copy()
        if not isinstance(memory["reports"], list):
            logger.warning(
                f"Le champ 'reports' dans '{path}' n'est pas une liste. Réinitialisation de la mémoire."
            )
            return DEFAULT_MEMORY.copy()
        return memory
    except FileNotFoundError:
        logger.info(f"Fichier mémoire '{path}' non trouvé. Initialisation d'une nouvelle mémoire.")
        return DEFAULT_MEMORY.copy()
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Impossible de lire le fichier mémoire '{path}': {e}. Réinitialisation de la mémoire.")
        return DEFAULT_MEMORY.copy()


def atomic_save_memory(memory: Dict[str, Any], path: str = MEMORY_FILE) -> None:
    """Atomically write the full memory structure to disk.

    The data is first written to a temporary file within the same directory
    before being moved into place.  This prevents partial writes from
    corrupting the memory file.  The ``seen_urls`` list will be sorted to
    ensure deterministic ordering.

    Args:
        memory: The memory dictionary to persist.  Must contain keys
            ``seen_urls``, ``details``, and ``reports``.
        path: The target memory file path.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    # Ensure 'seen_urls' is sorted for consistency.
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
            f"Sauvegarde de la mémoire terminée. Nombre total d'URLs: {len(mem_copy['seen_urls'])}."
        )
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Gemini invocation helper
#
def call_gemini_with_retry(
    prompt: str,
    max_retries: int = 3,
    initial_delay: int = 5,
    model_name: str = "gemini-2.5-flash"
) -> str:
    """Call the Gemini API with retry logic and return the response text.

    This helper abstracts away repeated attempts to contact the model.  It does
    not raise on failure; instead, it logs errors and returns an empty string
    after exhausting retries.

    Args:
        prompt: The textual prompt to send to the model.
        max_retries: Maximum number of attempts before giving up.
        initial_delay: Initial backoff delay (seconds) between retries.
        model_name: Name of the Gemini model to use.
    Returns:
        The model's textual response, or an empty string if all attempts fail.
    """

    tools = [
      {"url_context": {}},
      {"google_search": {}}
    ]

    model = genai.GenerativeModel(model_name=model_name)
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                        contents="Give me three day events schedule based on YOUR_URL. Also let me know what needs to taken care of considering weather and commute.",
                        generation_config=GenerationConfig(
                            tools=tools,
                        )
                    )

            # The API returns an object where `.text` holds the plain content.
            # Fallback to candidate text if `.text` is missing.
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            # Some SDK versions nest the text inside `candidates[0]`.
            if getattr(response, "candidates", None):
                candidate = response.candidates[0]
                text = getattr(candidate, "text", "") or getattr(candidate, "content", "")
                if text:
                    return str(text).strip()
            logger.warning(
                f"Réponse vide de Gemini (tentative {attempt + 1}/{max_retries}) pour le prompt: {prompt}"
            )
        except Exception as e:
            logger.error(
                f"Erreur lors de l'appel à Gemini (tentative {attempt + 1}/{max_retries}) pour le prompt '{prompt}': {e}"
            )
        # If there are remaining attempts, sleep before the next try.
        if attempt < max_retries - 1:
            sleep_time = initial_delay * (2 ** attempt)
            logger.info(f"Nouvelle tentative dans {sleep_time} secondes...")
            time.sleep(sleep_time)
    logger.error(
        f"Toutes les tentatives ont échoué pour le prompt '{prompt}'. Retour d'une chaîne vide."
    )
    return ""



# -----------------------------------------------------------------------------
# Veille logic executed in a background task
#
async def perform_watch_task() -> None:
    """Execute the monitoring logic in the background.

    This coroutine reads the source configuration, consults Gemini for new URLs
    based on subjects, deduplicates results against the memory file, and then
    persists any new discoveries.  Exceptions are caught and logged so that
    background tasks never propagate errors to the ASGI layer.
    """
    logger.info("Tâche de veille démarrée.")
    try:
        # Load and validate the sources configuration.
        with open(SOURCES_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = SourceConfig(**config_data)
    except FileNotFoundError:
        logger.error(
            f"Fichier de configuration '{SOURCES_FILE}' introuvable. Veille annulée."
        )
        return
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(
            f"Le fichier de configuration '{SOURCES_FILE}' est invalide: {e}. Veille annulée."
        )
        return

    subjects_to_watch = config.veille_par_sujet
    urls_to_watch = config.veille_par_url
    # Load the full memory structure.  This provides seen_urls, details and reports.
    memory = safe_load_memory()
    seen_urls_set: Set[str] = set(memory.get("seen_urls", []))
    details: Dict[str, str] = memory.get("details", {})
    new_urls: Set[str] = set()
    new_details: Dict[str, str] = {}

    # Step 1: Monitor by subjects.  For each subject, ask Gemini to suggest a
    # handful of relevant URLs.  The prompt instructs the model to output only
    # URLs separated by whitespace to simplify parsing.  A simple regex is
    # employed to extract the links.
    for subject in subjects_to_watch:
        logger.info(f"Analyse du sujet: '{subject}'")
        prompt = (
            f"Liste jusqu'à 5 URLs pertinentes qui traitent du sujet suivant: {subject}. "
            "Retourne uniquement les URLs (commençant par http ou https) séparées par des espaces ou des sauts de ligne. "
            "Ne fournis aucune explication ou autre texte."
        )
        response_text = call_gemini_with_retry(prompt)
        if not response_text:
            logger.info(f"Aucune réponse obtenue pour le sujet '{subject}'.")
            continue
        urls = re.findall(r"https?://\S+", response_text)
        cleaned_urls = [u.rstrip('.,);') for u in urls]
        for url in cleaned_urls:
            if url in seen_urls_set or url in new_urls:
                continue
            new_urls.add(url)
            logger.info(f"Nouvelle URL détectée pour le sujet '{subject}': {url}")

    # Step 2: Monitor by explicit URLs.  Any URL not already in memory is
    # considered new.
    for url in urls_to_watch:
        if url in seen_urls_set or url in new_urls:
            logger.info(f"URL déjà connue, ignorée: '{url}'")
            continue
        new_urls.add(url)
        logger.info(f"Nouvelle URL ajoutée depuis la configuration: '{url}'")

    # Step 3: For each newly discovered URL, generate a detailed description.
    for url in new_urls:
        # Build a prompt asking the model to describe the content of the URL.
        description_prompt = (
            f"Rédige un résumé détaillé et précis en français du contenu trouvé sur cette page : {url}. "
            "Mets l'accent sur les mises à jour, les nouvelles ou les détails importants que cette page apporte. "
            "Si tu ne peux pas accéder directement à la page, base-toi sur tes connaissances générales pour deviner la nature du contenu."
        )
        description = call_gemini_with_retry(description_prompt)
        if not description:
            description = "Aucune description disponible pour cette URL."
        new_details[url] = description
        logger.info(f"Description générée pour '{url}'.")

    # Step 4: Generate a synthetic report summarising all new details.
    report_text = ""
    if new_details:
        # Compose a prompt summarising the new details.
        summary_lines = [f"{url}: {desc}" for url, desc in new_details.items()]
        report_prompt = (
            "Rédige un rapport synthétique en français qui résume les nouvelles informations suivantes en mettant en avant les éléments clés et leur pertinence :\n"
            + "\n".join(summary_lines)
        )
        report = call_gemini_with_retry(report_prompt)
        if report:
            report_text = report
        else:
            # Fallback to concatenating the descriptions if the model fails.
            report_text = "\n".join(summary_lines)

    # Step 5: Persist the updated memory if there are any new URLs or details.
    if new_urls:
        # Update seen_urls, details, and reports in memory.
        memory_seen = set(memory.get("seen_urls", []))
        memory_seen.update(new_urls)
        memory["seen_urls"] = list(memory_seen)
        # Merge existing details with new ones
        merged_details = memory.get("details", {})
        merged_details.update(new_details)
        memory["details"] = merged_details
        # Append report entry if we generated a report
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
        # Save memory to disk
        atomic_save_memory(memory)
    else:
        logger.info("Aucune nouvelle URL à sauvegarder.")

    logger.info("Tâche de veille terminée.")


# -----------------------------------------------------------------------------
# FastAPI application definition
#
app = FastAPI(
    title="Watcher API",
    description=(
        "Une API pour effectuer une veille stratégique sur des sujets et des URLs."
    ),
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/watch", summary="Déclenche la veille en tâche de fond")
async def trigger_watch_endpoint(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Trigger the background monitoring task and return immediately.

    This endpoint schedules the `perform_watch_task` coroutine to run in the
    background and immediately returns a confirmation message.  If the
    configuration file is missing or invalid, the error will be logged by the
    task itself and not affect the response.
    """
    # Use FastAPI's BackgroundTasks to schedule the coroutine.  Starlette will
    # handle invocation of async callables automatically, so we pass the
    # coroutine function directly.  Any exceptions will be logged within
    # `perform_watch_task` and will not affect the request lifecycle.
    background_tasks.add_task(perform_watch_task)
    logger.info("Requête de veille reçue. La tâche a été programmée en arrière-plan.")
    return {
        "message": "La veille a été lancée en arrière-plan. Consultez les logs pour les détails."
    }


@app.get("/memory", summary="Affiche la mémoire complète")
async def get_memory_content() -> Dict[str, Any]:
    """Return the full memory structure.

    This includes the list of seen URLs, the stored descriptions for each URL
    and the history of generated reports.
    """
    memory = safe_load_memory()
    return memory


@app.get("/", summary="Endpoint de santé")
async def root() -> Dict[str, str]:
    """Simple health‑check endpoint."""
    return {"status": "ok", "message": "Watcher API is operational."}


# -----------------------------------------------------------------------------
# Sources management endpoints
#
@app.get("/sources", summary="Lire la configuration des sources")
async def read_sources() -> Dict[str, Any]:
    """Return the current sources configuration.

    Reads the contents of the sources file and returns it.  If the file is
    missing or corrupt, an HTTP error is raised.
    """
    try:
        with open(SOURCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate using SourceConfig to ensure correct structure
        config = SourceConfig(**data)
        return config.dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Le fichier '{SOURCES_FILE}' est introuvable.")
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=500, detail=f"Le fichier '{SOURCES_FILE}' est invalide: {e}")


@app.post("/sources", summary="Modifier la configuration des sources")
async def update_sources(update: UpdateSourcesRequest) -> Dict[str, Any]:
    """Update the sources configuration.

    Clients can specify lists of subjects and/or URLs to add or remove, or
    provide a complete replacement configuration via the `replace` field.
    Unknown keys in the payload will be ignored.
    """
    # Load existing configuration or initialize a fresh one if the file is missing.
    try:
        if os.path.exists(SOURCES_FILE):
            with open(SOURCES_FILE, "r", encoding="utf-8") as f:
                current_data = json.load(f)
            current_config = SourceConfig(**current_data)
        else:
            current_config = SourceConfig()
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=500, detail=f"Le fichier '{SOURCES_FILE}' est invalide: {e}")

    # If replace is provided, overwrite entirely with the new configuration.
    if update.replace is not None:
        new_config = update.replace
    else:
        # Start from the current config and apply additions/removals.
        new_config = SourceConfig(
            veille_par_sujet=list(current_config.veille_par_sujet),
            veille_par_url=list(current_config.veille_par_url),
        )
        # Add new subjects
        if update.add_subjects:
            for subj in update.add_subjects:
                if subj not in new_config.veille_par_sujet:
                    new_config.veille_par_sujet.append(subj)
        # Add new URLs
        if update.add_urls:
            for url in update.add_urls:
                if url not in new_config.veille_par_url:
                    new_config.veille_par_url.append(url)
        # Remove subjects
        if update.remove_subjects:
            new_config.veille_par_sujet = [s for s in new_config.veille_par_sujet if s not in update.remove_subjects]
        # Remove URLs
        if update.remove_urls:
            new_config.veille_par_url = [u for u in new_config.veille_par_url if u not in update.remove_urls]

    # Persist the updated configuration
    try:
        with open(SOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump(new_config.dict(), f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'écriture de '{SOURCES_FILE}': {e}")
    return new_config.dict()


# -----------------------------------------------------------------------------
# Details and reports endpoints
#
@app.get("/details", summary="Consulter les descriptions enregistrées")
async def get_details() -> Dict[str, str]:
    """Return the mapping of URLs to their detailed descriptions."""
    memory = safe_load_memory()
    return memory.get("details", {})


@app.get("/reports", summary="Consulter l'historique des rapports générés")
async def get_reports() -> List[Dict[str, Any]]:
    """Return the list of generated reports."""
    memory = safe_load_memory()
    return memory.get("reports", [])



