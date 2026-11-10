"""Les deux seules familles d'erreurs attendues des connecteurs.

- ConfigError    : l'environnement est faux (variable absente, URL invalide...). Rien n'a été contacté.
- ConnectorError : la configuration est lisible mais le service ne répond pas comme attendu
                   (injoignable, identifiants refusés, schéma absent...).
"""

from __future__ import annotations


class ConfigError(ValueError):
    """Configuration absente ou invalide."""


class ConnectorError(RuntimeError):
    """Service injoignable ou qui refuse la connexion."""

    def __init__(self, service: str, target: str, detail: str, hint: str | None = None) -> None:
        self.service = service
        self.target = target
        self.detail = detail
        self.hint = hint
        message = f'{service} ({target}) : {detail}'
        if hint:
            message += f' -> {hint}'
        super().__init__(message)
