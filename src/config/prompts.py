# -*- coding: utf-8 -*-
"""
Config Loader - Centralized prompt configuration management

Charge et gère la configuration des prompts depuis prompts.yaml.
Utilise le pattern Singleton pour éviter les chargements multiples.
"""

import yaml
import os
from pathlib import Path
from typing import Optional
from loguru import logger


class PromptsConfig:
    """
    Charge et gère la configuration des prompts depuis prompts.yaml
    
    Responsabilités :
    - Charger le fichier YAML une seule fois
    - Fournir accès aux prompts (system, rag, no_results)
    - Fournir accès à la configuration RAG
    - Gérer les erreurs de chargement avec fallback
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialise le chargeur de configuration.
        
        Args:
            config_path: Chemin vers prompts.yaml (par défaut: config/prompts.yaml)
        """
        if config_path is None:
            # Chemin par défaut relatif à la racine du projet
            config_path = os.environ.get(
                "PROMPTS_PATH",
                os.path.join(Path(__file__).resolve().parents[2], 'configs', 'prompts.yaml'),
            )
        
        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"PromptsConfig initialized from {self.config_path}")
    
    def _load_config(self) -> dict:
        """
        Charge le fichier YAML de configuration.
        
        Returns:
            dict: Configuration chargée depuis YAML ou fallback par défaut
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not config:
                logger.warning(f"Configuration vide dans {self.config_path}, utilisation du fallback")
                return self._get_default_config()
            
            logger.info(f"✅ Prompts chargés depuis : {self.config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"❌ Fichier non trouvé : {self.config_path}")
            logger.warning("Utilisation de la configuration par défaut")
            return self._get_default_config()
            
        except yaml.YAMLError as e:
            logger.error(f"❌ Erreur de parsing YAML : {e}")
            logger.warning("Utilisation de la configuration par défaut")
            return self._get_default_config()
            
        except Exception as e:
            logger.error(f"❌ Erreur inattendue : {e}")
            logger.warning("Utilisation de la configuration par défaut")
            return self._get_default_config()
    
    def get_system_prompt(self) -> str:
        """
        Retourne le system_prompt pour les appels LLM.
        
        Returns:
            str: System prompt complet
        """
        return self.config.get('system_prompt', '')
    
    def get_rag_prompt(self) -> str:
        """
        Retourne le template de rag_prompt.
        
        Returns:
            str: Template rag_prompt avec placeholders {context} et {query}
        """
        return self.config.get('rag_prompt', '')
    
    def get_no_results_prompt(self) -> str:
        """
        Retourne le prompt à afficher quand aucun résultat n'est trouvé.
        
        Returns:
            str: Template no_results_prompt avec placeholder {query}
        """
        return self.config.get('no_results_prompt', '')
    
    def get_search_config(self) -> dict:
        """
        Retourne la configuration RAG (max_results, scores, weights, etc.).
        
        Returns:
            dict: Configuration RAG
        """
        return self.config.get('search_config', {})
    
    def format_rag_prompt(self, context: str, query: str) -> str:
        """
        Formate le rag_prompt avec le contexte et la requête.
        
        Args:
            context: Contexte fourni par la recherche RAG
            query: Requête utilisateur
            
        Returns:
            str: rag_prompt formaté
        """
        template = self.get_rag_prompt()
        try:
            return template.format(context=context, query=query)
        except KeyError as e:
            logger.error(f"Erreur de formatage rag_prompt : clé manquante {e}")
            return f"Contexte:\n{context}\n\nQuestion:\n{query}"
    
    def format_no_results_prompt(self, query: str) -> str:
        """
        Formate le no_results_prompt avec la requête utilisateur.
        
        Args:
            query: Requête utilisateur
            
        Returns:
            str: no_results_prompt formaté
        """
        template = self.get_no_results_prompt()
        try:
            return template.format(query=query)
        except KeyError as e:
            logger.error(f"Erreur de formatage no_results_prompt : clé manquante {e}")
            return f"Aucun résultat trouvé pour : \"{query}\""
    
    @staticmethod
    def _get_default_config() -> dict:
        """
        Configuration par défaut si le YAML échoue.
        
        Returns:
            dict: Configuration minimale de fallback
        """
        return {
            'system_prompt': (
                'Tu es Luciole, un assistant documentaire. '
                'Tu t\'appuies en priorité sur les documents fournis. '
                'Ne jamais inventer de données factuelles. '
                'Toujours citer tes sources.'
            ),
            'rag_prompt': (
                'Contexte documentaire :\n\n{context}\n\n'
                'Question de l\'utilisateur : {query}\n\n'
                'Réponds à la question en te basant UNIQUEMENT sur le contexte fourni. '
                'Si l\'information n\'est pas présente, indique-le clairement. '
                'Cite le document et la page indiqués dans l\'étiquette de source '
                'de chaque extrait utilisé.'
            ),
            'no_results_prompt': (
                'Aucun document pertinent n\'a été trouvé pour cette requête : \"{query}\"\n\n'
                'Suggestions :\n'
                '- Reformulez votre question avec des termes différents\n'
                '- Vérifiez que les documents concernés ont été indexés\n'
                '- Essayez une recherche plus générale'
            ),
            'search_config': {
                'max_results': 10,
                'min_score': 0.3,
                'bm25_weight': 0.6,
                'knn_weight': 0.4
            }
        }


# Singleton global pour éviter les chargements multiples
_prompts_instance: Optional[PromptsConfig] = None


def load_prompts(config_path: str = None) -> PromptsConfig:
    """
    Charge la configuration des prompts (Singleton pattern).
    
    La première fois qu'elle est appelée, elle crée une instance et la cache.
    Les appels suivants retournent la même instance.
    
    Args:
        config_path: Chemin vers prompts.yaml (optionnel, pour override)
        
    Returns:
        PromptsConfig: Instance singleton
        
    Example:
        ```python
        prompts = load_prompts()
        system_prompt = prompts.get_system_prompt()
        ```
    """
    global _prompts_instance
    
    if _prompts_instance is None:
        _prompts_instance = PromptsConfig(config_path)
        logger.info("✅ PromptsConfig singleton initialisé")
    
    return _prompts_instance


def reset_prompts_instance():
    """
    Réinitialise l'instance singleton (utile pour les tests).
    """
    global _prompts_instance
    _prompts_instance = None
    logger.info("PromptsConfig singleton réinitialisé")
