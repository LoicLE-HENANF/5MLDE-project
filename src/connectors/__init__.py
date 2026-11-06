"""Connecteurs vers les briques de la stack (MLflow, Prefect, PostgreSQL, Grafana).

Règles communes :
- configuration uniquement par variables d'environnement (connectors.config), aucun argument CLI :
  le même code tourne dans un flow Prefect, un worker, l'API ou un test ;
- aucune I/O dans __init__ : construire un connecteur ne contacte jamais le service ;
- les clients bas niveau sont injectables (tests sans réseau) ;
- check() lève ConnectorError (cible + conseil) si le service ne répond pas comme attendu ;
  une configuration fausse lève ConfigError avant tout appel réseau.
"""

from connectors.errors import ConfigError, ConnectorError

__all__ = ['ConfigError', 'ConnectorError']
