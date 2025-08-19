"""
Watcher API Version 2 (Rewritten)

This module exposes a FastAPI application that performs basic web monitoring ("veille")
by leveraging Google's Gemini generative models.  It reads a configuration file
(`sources.json`) containing two lists:

* `veille_par_sujet`: A list of high-level subjects to watch.  For each subject,
  the application asks the Gemini model to suggest a handful of relevant URLs.
  New URLs not previously seen are persisted.
* `veille_par_url`: A list of explicit URLs to watch.  Any URL not already
  present in the in-memory database is added to the watch list.

The module maintains a simple JSON "database" (`memory_db.json`) storing
previously discovered URLs.  Reads and writes to this file are performed
robustly: invalid or missing files trigger an automatic reset, and writes are
done atomically to avoid corruption in concurrent environments.

Usage:

1. Set the environment variable `GEMINI_API_KEY` with a valid Gemini API key.
2. Place a `sources.json` file in the working directory with the structure:

   {
     "veille_par_sujet": ["Sujet 1", "Sujet 2", ...],
     "veille_par_url": ["https://example.com/page", ...]
   }

3. Run this module with a WSGI/ASGI server (e.g. `uvicorn main:app`).

Endpoints:

* POST `/watch`: Triggers a background task that processes the sources and
  updates the memory file.  Returns immediately.
* GET `/memory`: Returns the current contents of the memory file.
* GET `/`: Health‑check endpoint.

Note: This rewrite deliberately avoids using Gemini function calling.  Instead
it formulates a plain‑text query to the model asking for a list of URLs.  The
response is parsed with a simple regular expression to extract HTTP/HTTPS
links.  This approach bypasses the "Unknown field for FunctionDeclaration"
errors encountered when using invalid tool declarations.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Set

import google.generativeai as genai
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, ValidationError


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
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
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
# to extend the schema later if needed.
DEFAULT_MEMORY: Dict[str, Any] = {"seen_urls": []}


# -----------------------------------------------------------------------------
# Pydantic model for the sources configuration
#
# This model enforces that `veille_par_sujet` and `veille_par_url` are lists of
# strings.  If additional keys are present in the JSON, they will be ignored
# unless explicitly defined here.
class SourceConfig(BaseModel):
    veille_par_sujet: List[str] = []
    veille_par_url: List[str] = []


# -----------------------------------------------------------------------------
# Memory management helpers
#
def safe_load_memory(path: str = MEMORY_FILE) -> Set[str]:
    """Load the set of previously seen URLs from the memory file.

    If the file is missing, empty, corrupt, or does not match the expected
    schema, an empty set is returned and a log message is emitted.  This
    behaviour avoids raising exceptions from background tasks and allows the
    watcher to recover gracefully from invalid state.

    Args:
        path: The path to the memory JSON file.
    Returns:
        A set of URLs previously stored.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "seen_urls" not in data:
            logger.warning(
                f"Format de fichier mémoire invalide pour '{path}'. Réinitialisation de la mémoire."
            )
            return set()
        urls = data.get("seen_urls", [])
        if not isinstance(urls, list):
            logger.warning(
                f"Le champ 'seen_urls' dans '{path}' n'est pas une liste. Réinitialisation de la mémoire."
            )
            return set()
        return set(urls)
    except FileNotFoundError:
        logger.info(f"Fichier mémoire '{path}' non trouvé. Initialisation d'une nouvelle mémoire.")
        return set()
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Impossible de lire le fichier mémoire '{path}': {e}. Réinitialisation de la mémoire.")
        return set()


def atomic_save_memory(urls: Set[str], path: str = MEMORY_FILE) -> None:
    """Atomically write the set of seen URLs to the memory file.

    Data is first written to a temporary file in the same directory and then
    moved into place.  This pattern guards against partial writes if the
    process crashes or is killed during the write.

    Args:
        urls: The set of URLs to persist.
        path: The target memory file path.
    """
    # Prepare the directory for the temporary file.  If the directory is the
    # current working directory (""), `os.makedirs` with empty string is a no-op.
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    # Sort the URLs to ensure deterministic output for diff‑based tools.
    data_to_write = {"seen_urls": sorted(urls)}
    # Create a temporary file in the target directory.  Using `text=True` makes
    # the file handle operate in text mode.
    fd, temp_path = tempfile.mkstemp(prefix=".memory_tmp_", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(data_to_write, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(temp_path, path)
        logger.info(f"Sauvegarde de la mémoire terminée. Nombre total d'URLs: {len(urls)}.")
    finally:
        # In case of exceptions during write/replace, clean up the temp file.
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
    model_name: str = "gemini-1.5-flash",
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
    model = genai.GenerativeModel(model_name=model_name)
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
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
    seen_urls = safe_load_memory()
    newly_found_urls: Set[str] = set()

    # Step 1: Monitor by subjects.  For each subject, ask Gemini to suggest a
    # handful of relevant URLs.  The prompt instructs the model to output only
    # URLs separated by whitespace to simplify parsing.  A simple regex is
    # employed to extract the links.
    for subject in subjects_to_watch:
        logger.info(f"Analyse du sujet: '{subject}'")
        prompt = (
            f"Liste jusqu'à cinq URLs pertinentes qui traitent du sujet suivant: {subject}. "
            "Retourne uniquement les URLs (commençant par http ou https) séparées par des espaces ou des sauts de ligne. "
            "Ne fournis aucune explication ou autre texte."
        )
        response_text = call_gemini_with_retry(prompt)
        if not response_text:
            logger.info(f"Aucune réponse obtenue pour le sujet '{subject}'.")
            continue
        # Extraire toutes les URLs HTTP/HTTPS à l'aide d'une regex simple.  On
        # coupe les éventuelles ponctuations finales.
        urls = re.findall(r"https?://\S+", response_text)
        cleaned_urls = [u.rstrip('.,);') for u in urls]
        new_urls = [url for url in cleaned_urls if url not in seen_urls]
        if new_urls:
            newly_found_urls.update(new_urls)
            logger.info(
                f"{len(new_urls)} nouvelles URL(s) trouvée(s) pour le sujet '{subject}'."
            )
        else:
            logger.info(f"Aucune nouvelle URL pour le sujet '{subject}'.")

    # Step 2: Monitor by explicit URLs.  Any URL not already in memory is
    # considered new.  At this stage we do not fetch or summarise the content;
    # only deduplicate and store.
    for url in urls_to_watch:
        if url in seen_urls:
            logger.info(f"URL déjà connue, ignorée: '{url}'")
            continue
        newly_found_urls.add(url)
        logger.info(f"Nouvelle URL ajoutée depuis la configuration: '{url}'")

    # Step 3: Persist the union of old and new URLs if any new ones were found.
    if newly_found_urls:
        updated_urls = seen_urls.union(newly_found_urls)
        atomic_save_memory(updated_urls)
    else:
        logger.info("Aucune nouvelle URL à sauvegarder.")

    logger.info("Tâche de veille terminée.")


# -----------------------------------------------------------------------------
# FastAPI application definition
#
app = FastAPI(
    title="Watcher API v2 (Rewritten)",
    description=(
        "Une API qui utilise Gemini pour effectuer une veille stratégique sur des sujets et des URLs."
    ),
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


@app.get("/memory", summary="Affiche le contenu actuel de la mémoire")
async def get_memory_content() -> Dict[str, List[str]]:
    """Return the list of all seen URLs for debugging purposes."""
    seen = safe_load_memory()
    return {"seen_urls": sorted(seen)}


@app.get("/", summary="Endpoint de santé")
async def root() -> Dict[str, str]:
    """Simple health‑check endpoint."""
    return {"status": "ok", "message": "Watcher API is operational."}
