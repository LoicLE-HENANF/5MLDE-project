# Choix du sujet — Signaux de trading crypto (horizon horaire)

> Statut : **brouillon**. ✅ = décidé, 🟡 = proposition à valider.

## 1. Sujet

Produire chaque heure, pour un panier de cryptomonnaies, deux prédictions servies par une API :

1. **Rendement** à court terme (signal de direction et d'intensité) ;
2. **Volatilité** réalisée à venir (niveau de risque).

Inspiration : [Alpha Arena (nof1.ai)](https://www.nof1.ai/), qui compare des modèles sur du trading réel.
Ici, **aucun ordre réel n'est passé** : les décisions sont simulées (*paper trading*) et évaluées a posteriori.

Conformément à l'énoncé, la performance prédictive n'est pas l'objectif. Les métriques servent à
**décider la mise en service, la surveillance et le retour arrière** des modèles.

### Pourquoi pas la seule précision de direction

Sur un marché liquide, prédire « hausse / baisse » donne environ 50 % de bonnes réponses quel que soit le modèle.
Cette précision ne distingue donc pas un bon modèle d'un mauvais. On évalue plutôt :
- la **qualité du classement** des mouvements (IC) ;
- le **résultat économique net de frais**, comparé à des références ;
- pour la volatilité, l'**erreur face à une prévision naïve**, sur une cible bien plus prévisible.

## 2. Décisions

| # | Décision | Statut |
|---|---|---|
| D1 | Marché : **crypto, spot**, paires en USDT | ✅ |
| D2 | Pas de temps : **bougies 1 h** | ✅ |
| D3 | Deux cibles : **rendement + volatilité** | ✅ |
| D4 | **Deux modèles séparés**, chacun entraîné sur tous les symboles (cf. §4) | 🟡 |
| D5 | Univers : `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT` | ✅ |
| D6 | Historique : depuis 2022-01 (≈ 4 ans, ≈ 35 000 bougies par symbole) | ✅ (à revoir à l'ingestion) |
| D10 | Stratégies simulées : **long/short transversal** (référence pour la promotion) + **long/cash** (réaliste en spot), modulées par la volatilité prédite (cf. §4 bis) | ✅ |
| D11 | Application : **Streamlit** | ✅ (à revoir à l'implémentation) |
| D7 | **Application de pilotage** : prédictions, marche/arrêt, configuration, modèles (cf. §6) | ✅ principe, 🟡 techno |
| D9 | Horizon de la cible de rendement : **4 h** | ✅ |
| D8 | Simulation uniquement, frais de 0,1 % par ordre (taker spot Binance) | 🟡 |

## 3. Données

### Source

**Binance Public Data**, via deux canaux. Les deux sont gratuits et ne demandent pas de clé d'API.

| Usage | Canal | Détail |
|---|---|---|
| Historique (ingestion initiale, réentraînements) | `https://data.binance.vision` | Archives mensuelles/journalières `data/spot/monthly/klines/<SYMBOLE>/1h/<SYMBOLE>-1h-<AAAA-MM>.zip`, chacune avec un fichier `.CHECKSUM` (SHA-256) |
| Nouvelles bougies (fonctionnement horaire) | API REST `GET /api/v3/klines` | Dernières bougies clôturées |

Format d'une bougie : `open_time, open, high, low, close, volume, close_time, quote_volume, count,
taker_buy_base_volume, taker_buy_quote_volume, ignore` (CSV **sans en-tête**).

### Provenance et conditions d'utilisation (à documenter dans le README)

- Les données sont publiées par Binance. Leur usage est soumis aux conditions d'utilisation de Binance,
  qui seront citées et datées à la récupération.
- **Le dépôt ne contient aucune donnée** : un script les télécharge, et les sommes SHA-256 assurent la reproductibilité.
- L'accès à l'API Binance est restreint dans certains pays. Le projet est exécuté depuis la France.

### Rejeu et temps réel

Deux modes, sélectionnés par la configuration :
- **rejeu** : l'historique est découpé en lots horaires et joué en accéléré (ce mode est déclaré comme tel, comme le permet l'énoncé) ;
- **temps réel** : chaque heure, un flow récupère la bougie qui vient de se clôturer.

### Contrôles qualité (bloquants avant entraînement)

- Bougies manquantes ou dupliquées, horodatages non alignés sur l'heure.
- Cohérence OHLC (`low ≤ open, close ≤ high`), prix strictement positifs, volumes ≥ 0.
- **Unité des horodatages** : à vérifier sur les fichiers. Les archives spot récentes semblent être passées de la
  milliseconde à la microseconde, un piège typique à détecter.
- Checksum de l'archive différent de celui publié.
- Valeurs aberrantes (rendement horaire extrême) : signalées, ou bloquantes au-delà d'un seuil.

## 4. Combien de modèles ?

**Proposition : 2 modèles**, chacun entraîné sur l'ensemble des 5 symboles.

| | Modèle `rendement` | Modèle `volatilite` |
|---|---|---|
| Cible | Rendement logarithmique sur les **4 h** suivantes (✅) | Volatilité réalisée sur les **24 h** suivantes (racine de la somme des carrés des rendements horaires) |
| Type | Régression (le signe donne la direction) | Régression |
| Métrique principale | IC (Spearman) moyen et sa stabilité (ICIR) | QLIKE ou MAE sur la log-volatilité |
| Référence à battre | Signaux aléatoires, buy & hold, momentum naïf | Volatilité des dernières 24 h (persistance) |
| Rythme naturel de réentraînement | Plus fréquent (signal instable) | Plus rare (cible stable) |

Le modèle de rendement indique où se positionner, et le modèle de volatilité combien engager ou quand
s'abstenir. La stratégie simulée combine les deux.

### Pourquoi deux modèles et pas un seul qui prédit tout

- **Cycles de vie indépendants** : chaque modèle a ses versions, son alias de production et ses
  règles de mise en service. Si le modèle de rendement se dégrade, on revient à sa version précédente sans toucher à la volatilité.
- **Cibles, pertes et métriques différentes** : un modèle multi-sorties obligerait à un compromis entre les deux cibles et
  rendrait la règle de promotion ambiguë (meilleur sur l'une, pire sur l'autre).
- C'est le schéma déjà utilisé au TP05 (deux tâches), réutilisable pour l'API, la traçabilité et Grafana.

### Pourquoi un seul modèle multi-symboles et pas un modèle par crypto

- **5 fois plus de données** par modèle, avec des features normalisées (rendements, ratios) comparables d'un symbole à l'autre.
- **IC transversal possible** : à chaque heure, on compare le classement des 5 symboles prédit et réalisé.
- On passe de 10 modèles à 2 à versionner et à superviser.
- Le symbole peut rester une feature si les comportements diffèrent trop.

### Algorithmes 🟡

Des modèles simples avec scikit-learn : **Ridge** comme référence linéaire et **HistGradientBoostingRegressor**
comme candidat. La comparaison entre les deux illustre la règle de promotion.

### Spécification initiale des features (V1)

Une ligne d'entraînement ou d'inférence représente un couple **(symbole, heure t)**. Les features
sont calculées après la clôture de la bougie t, à partir de cette bougie et des précédentes. Les prix
bruts ne sont pas utilisés : les rendements et ratios rendent les entrées comparables entre actifs.

| Famille | Features V1 | Définition |
|---|---|---|
| Momentum | `return_1h`, `return_4h`, `return_12h`, `return_24h`, `return_72h`, `return_168h` | Rendement logarithmique `log(close[t] / close[t-h])` |
| Volatilité réalisée | `realized_vol_6h`, `realized_vol_24h`, `realized_vol_72h`, `realized_vol_168h` | Racine de la somme des carrés des rendements horaires disponibles jusqu'à t, sur h heures |
| Forme de la bougie | `candle_range`, `candle_body`, `close_location` | Respectivement `log(high/low)`, `log(close/open)` et `(2*close-high-low)/(high-low)` ; `close_location` vaut 0 si `high == low` |
| Activité | `quote_volume_ratio_24h`, `taker_buy_ratio`, `trade_count_ratio_24h` | Volume quote et nombre de transactions rapportés à leur médiane des 24 heures précédentes ; part du volume quote achetée par des takers |
| Contexte transversal | `market_return_{1h,4h,24h}`, `relative_return_{1h,4h,24h}` | Médiane des rendements des cinq symboles à l'horizon h ; rendement du symbole moins cette médiane |
| Temps | `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos` | Encodage cyclique de l'heure et du jour de semaine UTC de la bougie t |
| Identité | `symbol` | Symbole parmi les cinq de l'univers ; encodage explicite en colonnes indicatrices pour les modèles scikit-learn |

Les rapports d'activité sont calculés avec la médiane des bougies **strictement antérieures à t** ; si
elle est nulle, le ratio vaut 0. Le ratio acheteur est `taker_buy_quote_volume / quote_volume` ;
il vaut 0 quand `quote_volume` est nul. Les features transversales sont calculées à heure identique,
sur les symboles disponibles ; la ligne est marquée non exploitable si le symbole lui-même est absent.

La cible rendement à 4 h est distincte des features : `log(close[t+4h] / close[t])`. Aucune donnée
postérieure à la clôture de t ne peut entrer dans les features. Les fenêtres d'historique insuffisantes
produisent des valeurs manquantes à écarter pour l'entraînement et l'inférence, sans les remplacer
par des valeurs calculées à partir du futur. En production, la bougie t n'est utilisée que si elle est
clôturée ; la bougie ouverte est ignorée.

Cette V1 est volontairement limitée aux informations déjà présentes dans les bougies. Elle exclut
les indicateurs techniques redondants, les données externes et toute sélection de variables fondée
sur la période de test finale. La même liste ordonnée de features devra être utilisée à l'entraînement
et à l'inférence.

## 4 bis. Stratégies ≠ modèles

Une **stratégie** est une règle de décision appliquée aux prédictions. Ce n'est pas un modèle.
Toutes les stratégies lisent les mêmes prédictions et ne diffèrent que par la règle de position :

| Stratégie | Règle (chaque heure, sur les 5 symboles) | Rôle |
|---|---|---|
| Long/short transversal | Achat du symbole au rendement prédit le plus haut, short du plus bas, montants égaux | Mesure la qualité du classement, indépendamment de la tendance du marché : **critère de promotion** |
| Long/cash | Achat si le rendement prédit dépasse les frais aller-retour, sinon on reste en dollars | Stratégie réaliste en spot, affichée dans l'application |
| Modulation par la volatilité | Taille de position inversement proportionnelle à la volatilité prédite ; abstention au-delà d'un seuil | Commune aux deux stratégies |

Les stratégies sont du **code de backtest paramétré par la configuration**. On peut en ajouter une sans réentraîner aucun modèle.
Entraîner un modèle par stratégie donnerait des modèles identiques, puisque la cible, les features et les données seraient les mêmes.

### À quoi servent les stratégies

Les stratégies n'influencent jamais l'entraînement : elles **lisent** les prédictions et les transforment en positions simulées.
Elles **notent** bien les modèles, mais pas seules : le choix d'un modèle se fait en trois niveaux (cf. §5).
Choisir uniquement sur le backtest reviendrait à sur-apprendre la période de test : avec assez de variantes
(features, hyperparamètres, seuils), l'une finit toujours par avoir un bon PnL simulé par hasard. Le PnL dépend aussi
des frais, du seuil d'abstention et de la taille des positions, et pas seulement de la qualité du modèle.

Elles ont trois usages :

1. **Évaluer avant la mise en service (backtest).** Après l'entraînement, les prédictions de la nouvelle version
   sont rejouées sur la période de test (données non vues), avec chaque stratégie. Les résultats sont enregistrés dans MLflow
   (Sharpe, max drawdown, rendement net de frais). Le **long/short** sert de **critère de promotion** (niveau 2) : la version passe
   en production seulement si elle n'est pas moins bonne que la version en production et fait mieux que les références (cf. §5).
2. **Présenter dans l'application (paper trading).** En fonctionnement, chaque heure, les stratégies prennent des
   positions *simulées* à partir des vraies prédictions de l'API. L'application affiche :
   - les positions du moment (ex. « long SOL, short XRP ») ;
   - la courbe de gain/perte cumulé de chaque stratégie depuis la mise en service. C'est le côté « Alpha Arena » de la démonstration.
3. **Surveiller en production.** Quand le réalisé arrive 4 h plus tard (feedback), on calcule le gain réel de chaque position
   simulée. Une perte persistante de la stratégie long/short signale une dégradation du modèle : alerte, voire retour
   automatique à la version précédente, comme la règle sur l'IC glissant.

En résumé : on **entraîne** le modèle, on l'**évalue** avec les stratégies sur l'historique, puis on le **surveille** avec les stratégies en
conditions réelles, et **l'application montre** le tout.

**Modèle supplémentaire possible (optionnel)** : un classifieur `opportunite` qui estime la probabilité que le rendement
à 4 h dépasse les frais aller-retour. Il répond directement à la question de la stratégie long/cash,
et il apporte des métriques de calibration (Brier, courbe de fiabilité). On l'ajoutera seulement si le temps le permet, une fois les deux modèles principaux
intégrés de bout en bout.

## 5. Évaluation et règles de mise en service

### Validation

**Walk-forward** : on entraîne sur le passé et on teste sur la période suivante, en glissant dans le temps. Un intervalle sans données
(*embargo*) d'au moins l'horizon de la cible sépare l'entraînement du test. **Jamais de split aléatoire.**
Les features sont calculées uniquement avec l'information disponible à l'instant t, pour éviter le *look-ahead*.

- **Cibles qui se chevauchent** : le rendement à 4 h de deux bougies voisines partage 3 h. Les exemples dont la cible
  déborde sur la période de validation sont retirés de l'entraînement (*purge*), en plus de l'embargo.
- **Période de test finale** (ex. les 3 derniers mois) : jamais utilisée pendant la mise au point (features, hyperparamètres,
  seuils), évaluée **une seule fois** par version candidate. C'est sur elle que tournent le backtest et la décision de promotion.
- **Mêmes features à l'entraînement et à l'inférence** : un seul module de calcul, embarqué dans le modèle avec le code.
- **Bougie en cours ignorée** : seule une bougie clôturée sert de feature ; la bougie ouverte renvoyée par l'API REST est écartée.

### Trois niveaux d'évaluation

| Niveau | Quand | Données | Métriques | Sert à |
|---|---|---|---|---|
| 1. Statistique | Pendant l'entraînement | Folds walk-forward | IC, ICIR (rendement) ; QLIKE (volatilité) | Choisir features et hyperparamètres |
| 2. Stratégie | Après l'entraînement, une fois | Période de test finale | Sharpe net, drawdown, rotation, comparés au champion et aux références | Décider la promotion |
| 3. Production | En continu | Nouvelles bougies réelles | IC glissant, PnL du paper trading | Surveiller, déclencher le retour arrière |

Le niveau 1 sélectionne, le niveau 2 valide, le niveau 3 confirme. Une version qui ne passe que le niveau 2 n'est pas promue.

### Métriques suivies (MLflow)

| Famille | Métriques |
|---|---|
| Signal | IC moyen, ICIR, précision de direction sur les 10 % de prédictions les plus confiantes |
| Économique (backtest net de frais) | Rendement cumulé, Sharpe, max drawdown, profit factor, turnover, nombre de trades |
| Références | Écart avec buy & hold et avec des signaux aléatoires (p-valeur empirique sur N tirages) |
| Volatilité | QLIKE et MAE de la log-volatilité, gain relatif face à la persistance |
| Robustesse | Métriques par fold et par régime de marché (haussier, baissier, latéral) |

### Règles 🟡 (seuils à fixer après une première mesure)

- **Promotion** d'une nouvelle version du modèle `rendement`, si toutes les conditions sont remplies :
  - niveau 1 : IC moyen > 0 sur la majorité des folds et supérieur à celui de la version en production ;
  - niveau 2 : Sharpe net de frais du long/short supérieur à celui du buy & hold et pas inférieur à celui du champion ;
  - niveau 2 : max drawdown pas pire de plus de X points.
- **Promotion** d'une nouvelle version du modèle `volatilite` : QLIKE meilleur que la persistance **et** que la version en production.
- **Challenger** : une version candidate peut d'abord recevoir l'alias `challenger`. Elle prédit en parallèle du `champion`
  (prédictions tracées, sans positions réelles) pendant N jours ; elle est promue si son niveau 3 confirme le niveau 2.
- **Retour arrière automatique** : si l'IC glissant mesuré en production reste sous 0 pendant N jours, **ou** si le gain/perte
  glissant de la stratégie long/short simulée reste négatif sur la même fenêtre, l'alias de production revient à la version précédente et une alerte est envoyée.
- **Retour arrière manuel** : une seule commande (ou un bouton de l'application) remet l'alias `champion` sur la version
  précédente et l'enregistre dans `tracing.model_events`.
- **Réentraînement** : planifié, et aussi déclenché par la **dérive** des features (PSI au-delà d'un seuil) ou par une baisse du niveau 3.

## 6. Retours, supervision et pilotage

- **Feedbacks naturels** : une prédiction faite à t est confrontée au rendement et à la volatilité réalisés à
  t + horizon, par un flow Prefect. Le système produit donc lui-même ses vérités terrain, sans saisie manuelle.
- **Traces** : chaque prédiction est enregistrée avec la version du modèle, la durée de traitement, les erreurs et le feedback associé.
- **Inférence idempotente** : une seule prédiction par (symbole, bougie, version du modèle), garantie par une contrainte
  d'unicité ; relancer un flow ne crée pas de doublon.
- **Application de pilotage** (D7) : interface utilisateur du système, qui sert aussi de support de démonstration.

  | Page | Contenu |
  |---|---|
  | Prédictions | Dernières prédictions par symbole (rendement 4 h, volatilité 24 h), signal de la stratégie, réalisé vs prédit |
  | Pilotage | **Marche / arrêt** du système, activation des alertes (push), paramètres modifiables (symboles actifs, seuils), historique des changements |
  | Modèles | Versions du registre MLflow, alias de production, métriques, **promotion et retour arrière** |
  | Supervision | Liens ou panneaux intégrés vers Grafana, état des services et des derniers runs Prefect |

  Principes :
  - L'application est un **client de l'API** : toute action passe par des endpoints validés et tracés
    (qui, quand, ancienne et nouvelle valeur). Elle n'écrit jamais directement en base.
  - **État et configuration en base** (`actif` / `inactif`, paramètres) : chaque flow les lit au démarrage et s'arrête
    proprement si le système est inactif. Les schedules Prefect restent aussi pausables depuis l'UI Prefect.
  - Grafana reste en lecture seule : il affiche l'état du système, l'application le modifie.
  - Techno 🟡 : **Streamlit** (100 % Python, rapide à écrire) plutôt qu'un front web séparé.

## 7. Questions ouvertes

- [x] Horizon de la cible de rendement : **4 h**.
- [x] Univers : les **5 symboles** proposés.
- [x] Historique depuis 2022 (à confirmer à l'ingestion) ; taille des lots de rejeu à fixer.
- [x] Stratégies : long/short transversal + long/cash, modulées par la volatilité. Le seuil d'abstention reste à fixer.
- [x] Interrupteur : **application de pilotage** + état en base via l'API.
- [x] Application : Streamlit (à revoir à l'implémentation).
- [ ] 3e modèle `opportunite` (classifieur) : optionnel, à décider après l'intégration des deux premiers.
- [ ] Durée de la période de test finale (3 mois proposés) et durée de la phase `challenger`.
- [ ] Seuil de dérive (PSI) déclenchant un réentraînement.

**Ordre de travail** : un circuit complet simple (ingestion → entraînement → inférence → retours → Grafana) avec le modèle
`rendement` d'abord, puis le modèle `volatilite`, puis les raffinements. Un circuit complet vaut mieux que deux modèles sans production.
