# main.py
import os
import json
import time
import logging
from typing import List, Set, Dict, Any

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, ValidationError

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration de l'API Gemini ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logger.critical("Erreur: La variable d'environnement GEMINI_API_KEY n'est pas définie.")
    raise Exception("GEMINI_API_KEY non définie. Veuillez la configurer.")

app = FastAPI(
    title="Watcher API v2 (Piloté par IA) - Robuste",
    description="Une API qui utilise Gemini pour effectuer une veille stratégique avec gestion des erreurs et tâches de fond."
)

# Fichiers pour la configuration et la mémoire
SOURCES_FILE = "sources.json"
MEMORY_FILE = "memory_db.json"

# --- Modèles de Données pour la Validation ---
class SourceConfig(BaseModel):
    veille_par_sujet: List[str] = []
    veille_par_url: List[str] = []

# --- Fonctions de Gestion de la Mémoire (avec gestion d'erreurs) ---
def load_memory() -> Set[str]:
    """Charge les URLs déjà vues depuis le fichier mémoire."""
    if not os.path.exists(MEMORY_FILE):
        logger.info(f"Fichier mémoire '{MEMORY_FILE}' non trouvé. Création d'une nouvelle mémoire vide.")
        return set()
    try:
        with open(MEMORY_FILE, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict) or "seen_urls" not in data or not isinstance(data["seen_urls"], list):
                logger.warning(f"Format de fichier mémoire '{MEMORY_FILE}' invalide. Réinitialisation de la mémoire.")
                return set()
            return set(data["seen_urls"])
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Erreur lors du chargement du fichier mémoire '{MEMORY_FILE}': {e}. Réinitialisation de la mémoire.")
        return set()

def save_to_memory(urls_to_add: Set[str]):
    """Ajoute de nouvelles URLs à la mémoire."""
    seen_urls = load_memory() # Recharger pour éviter les race conditions si le fichier est modifié ailleurs
    seen_urls.update(urls_to_add)
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump({"seen_urls": sorted(list(seen_urls))}, f, indent=2)
        logger.info(f"Mémoire mise à jour avec {len(urls_to_add)} nouvelles URLs. Total: {len(seen_urls)} URLs.")
    except IOError as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier mémoire '{MEMORY_FILE}': {e}")

# --- Fonction d'Appel à Gemini avec Retries ---
def call_gemini_with_retry(prompt: str, tools: List[str], max_retries: int = 3, initial_delay: int = 5) -> Any:
    """
    Appelle l'API Gemini avec une logique de retry exponentiel.
    """
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash', # Ou 'gemini-1.5-pro' si nécessaire
        tools=tools
    )
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            # Vérifier si la réponse contient du texte et des candidats valides
            if response and response.candidates and response.text:
                return response
            else:
                logger.warning(f"Réponse vide ou invalide de Gemini pour le prompt: '{prompt}'. Tentative {attempt + 1}/{max_retries}.")
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à Gemini (tentative {attempt + 1}/{max_retries}) pour le prompt '{prompt}': {e}")
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt) # Délai exponentiel
                logger.info(f"Nouvelle tentative dans {sleep_time} secondes...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Toutes les tentatives ont échoué pour le prompt '{prompt}'.")
    return None

# --- Logique de Veille (exécutée en tâche de fond) ---
async def perform_watch_task():
    """
    Exécute la logique de veille complète.
    """
    logger.info("Début de la tâche de veille en arrière-plan.")
    
    try:
        with open(SOURCES_FILE, 'r') as f:
            sources_data = json.load(f)
        config = SourceConfig(**sources_data) # Valider la structure du fichier sources.json
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Erreur lors du chargement ou de la validation du fichier de configuration '{SOURCES_FILE}': {e}")
        return {"status": "error", "message": f"Erreur de configuration: {e}"}

    subjects_to_watch = config.veille_par_sujet
    urls_to_watch = config.veille_par_url # Pour l'implémentation future de url_context
    
    all_new_findings = []
    newly_found_urls = set()
    
    seen_urls = load_memory() # Charger la mémoire une seule fois au début de la tâche

    # --- Veille par Sujet (Grounding with Google Search) ---
    for subject in subjects_to_watch:
        logger.info(f"Traitement du sujet de veille : '{subject}'")
        gemini_response = call_gemini_with_retry(subject, tools=['google_search'])
        
        if not gemini_response:
            logger.warning(f"Aucune réponse valide de Gemini pour le sujet '{subject}'.")
            continue

        response_text = gemini_response.text
        
        sources_from_gemini = []
        try:
            # Assurez-vous que grounding_metadata existe et est accessible
            if gemini_response.candidates and gemini_response.candidates[0].grounding_metadata:
                metadata = gemini_response.candidates[0].grounding_metadata
                sources_from_gemini = [chunk.web.uri for chunk in metadata.grounding_chunks if chunk.web and chunk.web.uri]
        except AttributeError as e:
            logger.warning(f"Impossible d'extraire les métadonnées de grounding pour le sujet '{subject}': {e}")

        new_sources_for_this_subject = [url for url in sources_from_gemini if url not in seen_urls]
        
        if new_sources_for_this_subject:
            all_new_findings.append({
                "type": "sujet",
                "sujet": subject,
                "synthese": response_text,
                "nouvelles_sources": new_sources_for_this_subject
            })
            newly_found_urls.update(new_sources_for_this_subject)
            logger.info(f"Nouveautés détectées pour le sujet '{subject}': {len(new_sources_for_this_subject)} nouvelles sources.")
        else:
            logger.info(f"Aucune nouvelle source pour le sujet '{subject}'.")

    # --- Veille par URL (URL Context) - À implémenter ---
    # Cette section serait similaire à la veille par sujet, mais appellerait Gemini avec l'outil 'url_context'
    # et un prompt spécifique pour résumer le contenu de l'URL.
    for url in urls_to_watch:
        logger.info(f"Traitement de l'URL de veille : '{url}'")
        # Exemple de prompt pour URL context
        prompt_url = f"Résume les points clés et les nouveautés de cette page : {url}"
        gemini_response_url = call_gemini_with_retry(prompt_url, tools=['url_context'])
        
        if not gemini_response_url:
            logger.warning(f"Aucune réponse valide de Gemini pour l'URL '{url}'.")
            continue
        
        response_text_url = gemini_response_url.text
        
        # Pour l'URL context, les sources sont l'URL elle-même si elle est nouvelle
        if url not in seen_urls:
            all_new_findings.append({
                "type": "url",
                "url": url,
                "synthese": response_text_url,
                "nouvelles_sources": [url] # La source est l'URL elle-même
            })
            newly_found_urls.add(url)
            logger.info(f"Nouveauté détectée pour l'URL '{url}'.")
        else:
            logger.info(f"URL '{url}' déjà vue.")


    # Sauvegarder toutes les nouvelles URLs trouvées en une seule fois
    if newly_found_urls:
        save_to_memory(newly_found_urls)
    
    logger.info("Tâche de veille terminée.")
    return {"status": "success", "message": "Tâche de veille terminée.", "new_findings_count": len(all_new_findings)}

# --- Endpoint de l'API ---
@app.post("/watch", summary="Déclenche la veille complète en tâche de fond")
async def trigger_watch_endpoint(background_tasks: BackgroundTasks):
    """
    Déclenche la tâche de veille complète en arrière-plan.
    Retourne une réponse immédiate au client.
    """
    background_tasks.add_task(perform_watch_task)
    logger.info("Requête de veille reçue. La tâche est lancée en arrière-plan.")
    return {"message": "La tâche de veille a été lancée en arrière-plan. Vérifiez les logs pour le statut."}

# --- Endpoint pour consulter la mémoire (pour le débogage) ---
@app.get("/memory", summary="Affiche le contenu actuel de la mémoire")
async def get_memory_content():
    """
    Retourne la liste des URLs actuellement stockées dans la mémoire.
    Utile pour le débogage.
    """
    return {"seen_urls": sorted(list(load_memory()))}

