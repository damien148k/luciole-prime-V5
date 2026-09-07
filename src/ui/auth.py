# -*- coding: utf-8 -*-
"""Authentification des interfaces web — DESACTIVEE.

Reprise telle quelle de la V3 : toutes les fonctions laissent passer.
Le deploiement suppose un acces physique ou un reseau de confiance.

ATTENTION : les services chat et feedback publient leur port sur toutes
les interfaces (voir docker-compose.yml). Sans authentification, toute
machine du reseau local qui atteint ce port peut lire les reponses, les
avis deposes et la configuration de l'instance, et la modifier.

Pour retablir un controle d'acces, remplacer les quatre fonctions
ci-dessous par une verification reelle : le reste du code ne fait aucune
hypothese sur leur implementation.
"""

from typing import Optional

AUTH_COOKIE_NAME = "luciole_admin"
AUTH_COOKIE_MAX_AGE = 86400


def verify_credentials(username: str, password: str) -> bool:
    return True


def make_session_token(username: str) -> str:
    return f"{username}:0:noauth"


def validate_session_token(token: str) -> Optional[str]:
    return "admin"


def get_login_html(error: str = "") -> str:
    return '<html><body><script>window.location="/"</script></body></html>'
