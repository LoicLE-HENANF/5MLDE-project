# Choix du sujet — Prévisions crypto et chaîne MLOps

> Document de cadrage et de justification, mis en cohérence avec l'énoncé et l'implémentation du dépôt.
> « En place » décrit une fonctionnalité présente dans le code ; « Limite » et « Suite possible » signalent
> explicitement ce qui n'est pas encore implémenté.

## 1. Sujet

Industrialiser une chaîne de prévision horaire pour un panier de cryptomonnaies : ingestion, contrôles qualité,
entraînement, versionnement, service, feedback et supervision. Deux prédictions sont servies par l'API :

1. **Rendement** à court terme (signal de direction et d'intensité) ;
2. **Volatilité** réalisée à venir (niveau de risque).

Les ordres ne sont jamais transmis à une plateforme : les positions sont simulées (*paper trading*) et évaluées
a posteriori. Cela illustre les décisions d'exploitation sans exposer de fonds réels.

Conformément à l'énoncé, la performance prédictive n'est pas un objectif à atteindre. Les métriques servent à
illustrer la sélection, la mise en service, la surveillance et le retour arrière des modèles.

### Pourquoi ces métriques

La précision de direction transforme un rendement continu en une classe et ignore son amplitude. Pour le
rendement, l'IC de Spearman mesure plutôt si le modèle classe correctement les actifs les uns par rapport aux
autres à une même heure ; il correspond au rôle du signal long/short. Le backtest net de frais vérifie ensuite
si ce classement donne une stratégie simulée cohérente. Pour la volatilité, QLIKE mesure l'erreur d'une
prévision de risque face à des références simples. Ces métriques sont des critères de décision opérationnels,
pas une promesse de rentabilité.

## 2. Décisions

| # | Décision retenue | État dans le projet |
|---|---|---|
| D1 | Marché **crypto spot**, paires cotées en USDT | En place |
| D2 | Bougies horaires (`1h`) | En place |
| D3 | Deux cibles : rendement et volatilité réalisée | En place |
| D4 | Deux modèles multi-symboles, avec cycles de vie et métriques distincts | En place |
| D5 | Univers : `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT` | En place |
| D6 | Historique à partir de janvier 2022 ; la période disponible dépend du dernier chargement | En place |
| D7 | Application de pilotage pour le service, les prédictions et le registre | En place, sans configuration métier libre |
| D8 | Simulation uniquement ; frais de 0,1 % appliqués au turnover du portefeuille | En place |
| D9 | Horizon de la cible rendement : 4 h | En place |
| D10 | Long/short et long/cash pour rendement ; ciblage de volatilité pour le portefeuille risque | En place |
| D11 | Interface Streamlit cliente de l'API | En place |

## 3. Données

### Source

Les bougies proviennent de Binance Spot, via deux canaux. Le client implémenté n'utilise pas de clé API.

| Usage | Canal | Détail |
|---|---|---|
| Historique (chargement initial et mois absents) | `https://data.binance.vision` | Archives mensuelles `data/spot/monthly/klines/<SYMBOLE>/1h/<SYMBOLE>-1h-<AAAA-MM>.zip`, vérifiées avec `.CHECKSUM` (SHA-256) |
| Bougies récentes | API REST Binance Spot `GET /api/v3/klines` | Reprise après la dernière bougie stockée |

Format d'une bougie : `open_time, open, high, low, close, volume, close_time, quote_volume, count,
taker_buy_base_volume, taker_buy_quote_volume, ignore` (CSV **sans en-tête**).

### Provenance et conditions d'utilisation

- Le dépôt ne contient pas les fichiers de données brutes : les archives Parquet locales sont sous `data/`,
  chemin ignoré par Git. Les téléchargements sont vérifiés par checksum et l'identité des jeux de données
  (période, symboles, nombre de lignes et SHA-256) est enregistrée dans MLflow.
- Les sources et le lien vers les conditions d'utilisation Binance sont documentés dans le [README](README.md).
  La date d'accès indiquée y est à distinguer d'une analyse juridique des conditions.

### Rejeu et temps réel

Le flow `ingest-candles` charge les mois d'historique absents depuis les archives puis récupère les bougies
récentes par l'API REST. Il est planifié toutes les heures. La validation et les backtests rejouent les données
historiques sur les folds temporels et le test final.

Pour montrer le cycle de vie complet sans attendre des mois, `python -m replay` rejoue 12 mois d'exploitation
(nov. 2025 - oct. 2026) avec le même code que le live : entraînement mensuel, promotion, prévisions horaires,
retours, santé et retour arrière. Il écrit dans un registre `<modèle>-replay` et des traces `mode = 'replay'`,
séparés du live. Seul écart volontaire : un niveau 2 en échec n'y bloque pas la promotion, pour que les
promotions et retours arrière soient visibles ; la raison est tracée.

### Contrôles qualité

- Le schéma, les doublons, les heures non alignées, la cohérence OHLC et un volume acheteur supérieur au volume total
  sont bloquants. Les prix sont strictement positifs et les volumes non négatifs selon le schéma.
- Les horodatages sources en millisecondes ou microsecondes sont normalisés en UTC.
- Un checksum d'archive différent, une variation absolue de log-rendement horaire supérieure à 50 % ou plus de 1 %
  d'heures manquantes rejettent le lot.
- Les bougies manquantes (jusqu'à 1 %) et les variations supérieures à 20 % et inférieures ou égales à 50 % sont
  signalées sans bloquer le lot.

#### Bougies interrompues (décision du 2026-11-18)

Le premier chargement complet de l'historique a rejeté le mois de mars 2023 pour les cinq symboles. Une seule
bougie était en cause : celle du **2023-03-24 à 12:00 UTC**, dont le `close_time` tombe à 12:39:41 au lieu de
12:59:59.999. Son volume et son nombre de transactions sont nuls. Elle correspond à l'arrêt du trading spot sur
Binance ce jour-là : la bougie a été interrompue en cours d'heure. La bougie de 13:00 est absente de l'archive
(743 lignes au lieu de 744), le trading n'ayant pas repris pendant cette heure.

Trois options ont été envisagées :
1. garder le rejet et laisser mars 2023 manquant (un mois de trou, flow d'ingestion en échec à chaque relance) ;
2. écarter uniquement les bougies interrompues et les tracer comme anomalie non bloquante ;
3. traiter cette date comme une exception codée en dur.

**Choix : option 2.** Une bougie interrompue ne décrit pas une heure complète. La conserver fausserait les
rendements, les volumes et la volatilité. La rejeter avec tout le mois ferait perdre 742 heures valides pour une
seule heure invalide. Une exception codée en dur ne couvrirait pas le prochain incident du même type.

Règle appliquée (`marketdata.sources.normalize_dropping_interrupted`, utilisée par l'ingestion) :
- `close_time` dans l'heure mais avant sa fin (`open_time ≤ close_time < open_time + 1 h − 1 ms`) : bougie
  **interrompue**, retirée du lot et tracée dans `tracing.ingestion_batches` (règle `bougie interrompue écartée`,
  non bloquante, avec les `open_time` concernés) ;
- `close_time` avant l'ouverture ou après la fin de l'heure : toujours une **erreur de format bloquante** ;
- l'heure retirée devient une bougie manquante, couverte par la règle des trous (bloquante seulement au-delà de
  1 % du lot). Les features sont calculées sur une grille horaire complète : les lignes touchées par le trou ont des
  valeurs manquantes et sont exclues des jeux d'entraînement.

`marketdata.sources.normalize` reste strict (il lève sur toute bougie interrompue) pour les usages qui ne passent
pas par l'ingestion tracée.

Résultat du rechargement : mars 2023 accepté pour les cinq symboles avec 742 bougies (12:00 écartée, 13:00
absente), les deux heures signalées comme anomalies non bloquantes.

## 4. Deux modèles, deux rôles

Les deux modèles sont des régressions scikit-learn entraînées sur le panel des cinq symboles. Ils partagent
les features, mais leurs cibles, métriques, références et règles de promotion sont différentes. Ils sont
entraînés chaque semaine par des flows distincts ; ce rythme est un choix opérationnel du projet, pas une
affirmation qu'il est optimal pour les marchés.

| | `rendement` | `volatilite` |
|---|---|---|
| Cible | Rendement logarithmique des **4 h** suivantes | Volatilité réalisée des **24 h** suivantes : racine de la somme des carrés des rendements horaires |
| Candidats | Ridge et HistGradientBoosting | Ridge et HistGradientBoosting |
| Métrique de sélection walk-forward | Moyenne des IC de Spearman | QLIKE moyen |
| Références | Rendement nul, momentum 1 h et momentum 4 h | Persistance 24 h, semaine précédente ramenée à 24 h, moyenne géométrique par symbole |
| Stratégies de niveau 2 | Long/short et long/cash, net de frais | Portefeuille équipondéré dont l'exposition est ciblée par la volatilité prédite |
| Déploiement d'entraînement | Chaque lundi à 03:00 UTC | Chaque lundi à 03:30 UTC |

Le modèle `rendement` classe les symboles selon leur rendement attendu. Le modèle `volatilite` estime le
risque futur et pilote séparément un portefeuille équipondéré à exposition réduite ou inchangée. Le code
actuel n'assemble pas les deux prévisions dans une stratégie commune.

### Pourquoi deux modèles et pas un seul qui prédit tout

- **Cycles de vie indépendants** : chaque modèle a ses propres versions, alias et règles de promotion.
  Une dégradation du rendement ne modifie pas le registre de la volatilité, et inversement.
- **Cibles, pertes et métriques différentes** : un modèle multi-sorties obligerait à un compromis entre les deux cibles et
  rendrait la règle de promotion ambiguë (meilleur sur l'une, pire sur l'autre).
- C'est le schéma déjà utilisé au TP05 (deux tâches), réutilisable pour l'API, la traçabilité et Grafana.

### Pourquoi un seul modèle multi-symboles et pas un modèle par crypto

- **5 fois plus de données** par modèle, avec des features normalisées (rendements, ratios) comparables d'un symbole à l'autre.
- **IC transversal possible** : à chaque heure, on compare le classement des 5 symboles prédit et réalisé.
- On passe de 10 modèles à 2 à versionner et à superviser.
- Le symbole peut rester une feature si les comportements diffèrent trop.

### Algorithmes

Les deux candidats sont **Ridge** (régression linéaire, avec standardisation) et
**HistGradientBoostingRegressor**. Ils sont évalués sur les mêmes folds ; le meilleur est choisi selon la
métrique propre à la cible. Les références naïves sont évaluées en parallèle, mais ne sont pas des candidats
à l'enregistrement comme nouveau modèle.

### Features utilisées

Une ligne d'entraînement ou d'inférence représente un couple **(symbole, heure t)**. Les features
sont calculées après la clôture de la bougie t, à partir de cette bougie et des précédentes. Les prix
bruts ne sont pas utilisés : les rendements et ratios rendent les entrées comparables entre actifs.

| Famille | Features V1 | Définition |
|---|---|---|
| Momentum | `return_1h`, `return_4h`, `return_12h`, `return_24h`, `return_72h`, `return_168h` | Rendement logarithmique `log(close[t] / close[t-h])` |
| Volatilité réalisée | `realized_vol_6h`, `realized_vol_24h`, `realized_vol_72h`, `realized_vol_168h` | Racine de la somme des carrés des rendements horaires des h dernières heures |
| Forme de la bougie | `candle_range`, `candle_body`, `close_location` | Respectivement `log(high/low)`, `log(close/open)` et `(2*close-high-low)/(high-low)` ; `close_location` vaut 0 si `high == low` |
| Activité | `quote_volume_ratio_24h`, `taker_buy_ratio`, `trade_count_ratio_24h` | Volume quote et nombre de transactions rapportés à leur médiane des 24 heures précédentes ; part du volume quote achetée par des takers |
| Contexte transversal | `market_return_{1h,4h,24h}`, `relative_return_{1h,4h,24h}` | Médiane des rendements des cinq symboles à l'horizon h ; rendement du symbole moins cette médiane |
| Temps | `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos` | Encodage cyclique de l'heure et du jour de semaine UTC de la bougie t |
| Identité | `symbol_BTCUSDT`, `symbol_ETHUSDT`, `symbol_SOLUSDT`, `symbol_BNBUSDT`, `symbol_XRPUSDT` | Symbole encodé en colonnes indicatrices |

Le modèle reçoit 26 features numériques et 5 colonnes indicatrices de symbole. La colonne `symbol` et
l'horodatage restent des identifiants ; ils ne sont pas transmis comme valeurs brutes au régresseur.

Les rapports d'activité sont calculés avec la médiane des bougies **strictement antérieures à t** ; si
elle est nulle, le ratio vaut 0. Le ratio acheteur est `taker_buy_quote_volume / quote_volume` ;
il vaut 0 quand `quote_volume` est nul. Les features transversales sont calculées à heure identique,
sur les symboles disponibles ; la ligne est marquée non exploitable si le symbole lui-même est absent.

La cible rendement à 4 h est distincte des features : pour une ligne `(symbole, t)`, c'est
`log(close[symbole, t+4h] / close[symbole, t])`. La bougie cible est recherchée pour le même symbole
à l'horodatage exact `t+4h` ; on ne décale pas la fenêtre vers la prochaine bougie disponible. Si la
bougie de départ ou celle à `t+4h` est absente, ou si la bougie cible n'est pas encore clôturée, la cible
est manquante (`NaN`). Ces lignes sont exclues de l'entraînement, mais une cible manquante n'empêche
pas de calculer ni de servir les features de la bougie t.

La cible utilise une donnée future uniquement comme étiquette d'entraînement ou de feedback ; elle
n'entre jamais dans les features. Aucune donnée postérieure à la clôture de t ne peut entrer dans les
features. Les fenêtres d'historique insuffisantes produisent des features manquantes, sans les remplacer
par des valeurs calculées à partir du futur. En production, la bougie t n'est utilisée que si elle est
clôturée ; la bougie ouverte est ignorée.

La cible volatilité à 24 h est distincte des features : pour une ligne `(symbole, t)`, elle vaut
`sqrt(r[1] ** 2 + r[2] ** 2 + ... + r[24] ** 2)`, où
`r[k] = log(close[symbole, t+k] / close[symbole, t+k-1])`.
Elle utilise les 24 rendements horaires exacts jusqu'à `t+24h`, pour le même symbole. La bougie de
départ et chacune des 24 bougies suivantes doivent être présentes et clôturées à `as_of` (leur
`close_time < as_of`) ; si une bougie manque ou n'est pas encore clôturée, la cible est `NaN`. On ne
décale pas la fenêtre pour contourner une lacune. La fonction `target_volatility` conserve les
identifiants `(symbol, open_time)`, ajoute `target_volatility`, et accepte un horizon entier positif
en heures, égal à 24 par défaut. `as_of` est facultatif et vaut par défaut l'instant courant en UTC.
Une cible manquante n'empêche pas de calculer ni de servir les features de la bougie t.

Cette V1 est volontairement limitée aux informations déjà présentes dans les bougies. Elle exclut
les indicateurs techniques redondants, les données externes et toute sélection de variables fondée
sur la période de test finale. La même liste ordonnée de features devra être utilisée à l'entraînement
et à l'inférence.

## 4 bis. Stratégies simulées

Une stratégie transforme les prédictions en poids de portefeuille ; elle ne constitue pas un troisième
modèle. Les backtests sont du *paper trading* et les frais sont calculés sur le turnover (y compris
l'allocation initiale depuis le cash), au taux par défaut de 0,1 %.

| Stratégie | Règle implémentée | Usage |
|---|---|---|
| Long/short rendement | Les rangs transversaux des prédictions sont centrés ; poids nets nuls et exposition brute normalisée à 1. | Évaluer le classement indépendamment du mouvement général du marché. |
| Long/cash rendement | Chaque symbole reçoit une part égale du capital si le rendement prédit dépasse le seuil de 0,1 %, sinon la position reste en cash. | Comparer un signal long seul, réaliste pour un univers spot. |
| Tranches rendement | Chaque nouveau portefeuille est moyenné avec les trois dernières allocations horaires : quatre tranches pour tenir la cible de 4 h. | Réduire les réallocations et le turnover par rapport à une rotation complète chaque heure. |
| Ciblage volatilité | Un portefeuille acheteur équipondéré est réduit selon `cible / volatilité prédite`, avec levier maximal égal à 1. | Comparer le contrôle de l'exposition au risque à un portefeuille brut et au ciblage par persistance. |

Les poids sont décidés à la clôture de t et appliqués au rendement réalisé entre les clôtures de t et t+1 h.
Les tranches font donc correspondre la tenue du portefeuille à l'horizon de prévision de 4 h sans attribuer
à une décision l'information des bougies futures.

Le ciblage de volatilité est évalué séparément : il ne module pas les poids des stratégies long/short ou
long/cash du modèle rendement. Les backtests rendement comparent aussi le portefeuille équipondéré brut et
les stratégies construites avec les momentum de 1 h et 4 h.

Un rendement brut positif peut devenir négatif après frais : le classement doit compenser le coût du turnover,
alors que le signal porte sur un rendement de seulement quatre heures. Les tranches réduisent ce coût, mais
ne l'annulent pas. Ce résultat est informatif pour le projet ; le backtest ne constitue pas une promesse de
rentabilité.

Les stratégies servent à évaluer les candidats sur le test final, puis à mesurer le PnL *paper* à partir
des retours observés. Les règles sont définies avant le test final et comparées à des références ; elles ne
sont pas choisies en optimisant les résultats sur ce test.

Une action de promotion manuelle existe pour l'exploitation du registre et exige une raison tracée. Elle
n'est pas utilisée pour désigner arbitrairement un champion de soutenance ni pour contourner un échec des
critères automatiques. Si aucun candidat ne satisfait ces critères, il faut présenter ce résultat et ses
limites, plutôt que sélectionner après coup une version défavorable.

Un classifieur optionnel `opportunite` (probabilité que le rendement 4 h dépasse un seuil) reste une piste
future ; il n'est pas entraîné dans la version actuelle.

## 5. Évaluation et règles de mise en service

### Validation

**Règle V1 des folds walk-forward** — tous les horodatages sont en UTC et aucune séparation aléatoire n'est utilisée :

- **Fenêtres** : entraînement expansif depuis le début de l'historique (2022-01), avec au moins 12 mois
  d'historique avant le premier fold. Chaque validation couvre un mois calendaire complet ; le fold suivant
  avance d'un mois et ajoute le mois précédent à l'entraînement. Le premier mois de validation est donc
  2023-01 si les données requises sont disponibles.
- **Panel multi-symboles** : toutes les lignes ayant le même `open_time` sont affectées au même côté du fold.
- **Horizon et séparation** : `H = 4 h` pour la cible rendement et `H = 24 h` pour la cible volatilité.
  Pour une validation commençant à `V`, les exemples d'entraînement dont la fenêtre cible atteint ou dépasse
  `V` (`t + H >= V`) sont purgés. On réserve également avant `V` un embargo d'au moins `H` sans exemples
  d'entraînement ; en pratique, la séparation configurée vaut donc 4 h pour rendement et 24 h pour volatilité.
- **Test final** : les trois derniers mois calendaires disponibles sont réservés au test final. Aucun fold de
  validation ne les chevauche ; les exemples de validation dont la cible atteindrait le début du test final
  sont aussi exclus. Le test final n'intervient ni dans la sélection des modèles par walk-forward ni dans
  le choix de features ou d'hyperparamètres. Il sert à évaluer le candidat retenu pour la décision de promotion.

**Règle V1 de `holdout_split`** — le découpage final reprend les mêmes conventions temporelles :

- Le début du holdout est le premier jour UTC du mois correspondant aux trois derniers mois calendaires
  disponibles (mois de la dernière observation inclus). Le holdout contient les observations de cette date
  jusqu'à la dernière observation disponible.
- L'entraînement utilise les observations antérieures au holdout, après purge des lignes dont la cible atteint
  ou dépasse son début (`t + H >= holdout_start`). Aucun exemple d'entraînement n'est retenu dans
  l'intervalle `[holdout_start - embargo, holdout_start)`, avec un embargo d'au moins `H`.
  `H` vaut 4 h pour le rendement et 24 h pour la volatilité.
- Le découpage conserve ensemble toutes les lignes d'un même `open_time`. Si une colonne cible est fournie,
  les lignes dont la cible est indisponible (`NaN`) sont exclues des deux ensembles ; cela retire notamment
  les cibles incomplètes en fin d'historique.
- Le splitter renvoie des indices positionnels. Il échoue explicitement si l'un des deux ensembles est vide.

Les features sont calculées uniquement avec l'information disponible à l'instant t, pour éviter le *look-ahead*.
- **Mêmes features à l'entraînement et à l'inférence** : un seul module de calcul est partagé ; la liste ordonnée de colonnes
  attendue par le régresseur est identique dans les deux parcours.
- **Bougie en cours ignorée** : seule une bougie clôturée sert de feature ; la bougie ouverte renvoyée par l'API REST est écartée.

### Trois niveaux d'évaluation

**Règle V1 des métriques du modèle `rendement`** :

- Pour chaque `open_time`, l'IC est la corrélation de Spearman transversale entre les prédictions et les
  rendements réalisés, calculée sur les symboles ayant les deux valeurs disponibles. Il faut au moins trois
  symboles et une variation dans les deux séries ; sinon l'IC de cette heure est `NaN`.
- `information_coefficient` renvoie une série d'IC indexée par `open_time`. Pour un fold, la moyenne des IC
  ignore les valeurs `NaN`.
- L'ICIR d'un fold vaut la moyenne des IC divisée par leur écart-type d'échantillon (`ddof=1`), sans
  annualisation. Il vaut `NaN` s'il reste moins de deux IC définis ou si leur écart-type est nul.

| Niveau | Quand | Données | Règle effective | Sert à |
|---|---|---|---|---|
| 1. Walk-forward | Entraînement | Folds temporels | Sélection du candidat selon IC moyen (rendement) ou QLIKE moyen (volatilité), comparé entre Ridge et HistGradientBoosting | Choisir le candidat |
| 2. Holdout | Après la sélection | Trois derniers mois, exclus des folds | Comparaison aux références et aux stratégies naïves ; pour rendement, IC et Sharpe long/short ; pour volatilité, QLIKE et Sharpe du ciblage | Décider l'enregistrement comme `champion`, `challenger` ou rejet |
| 3. Production | Quotidiennement | Feedbacks arrivés à échéance sur 7 jours | IC et PnL *paper* (rendement) ; QLIKE face à la persistance (volatilité) | Surveiller et, après 3 jours consécutifs dégradés, tenter le retour arrière |

Le niveau 1 choisit le candidat, le niveau 2 applique la règle d'enregistrement et de promotion. Le niveau 3
est un contrôle de production distinct : il déclenche le retour arrière automatique après trois jours
consécutifs dégradés, mais n'est pas un critère d'attente avant la première promotion.

### Métriques suivies (MLflow)

| Famille | Métriques journalisées |
|---|---|
| Walk-forward | IC moyen et ICIR par prédicteur pour `rendement` ; QLIKE et MAE de log-volatilité pour `volatilite` |
| Test final | Métriques des références, prédictions retenues et scores booléens indiquant si les seuils de comparaison sont satisfaits |
| Stratégies | Rendement total, Sharpe, drawdown maximal, turnover, frais et nombre d'heures selon les stratégies et références |
| Production | Santé quotidienne sur la fenêtre glissante de 7 jours, métrique de référence et PnL simulé |

### Règles de promotion et de retour arrière en place

- **Candidat rendement** : Ridge ou HistGradientBoosting est choisi selon l'IC moyen walk-forward. Au holdout,
  il doit battre la meilleure référence naïve en IC, et son Sharpe long/short doit dépasser ceux du portefeuille
  brut et des stratégies momentum 1 h et 4 h. Si un champion existe, le candidat doit aussi avoir un Sharpe
  long/short au moins égal au sien sur le même holdout.
- **Candidat volatilité** : le modèle est choisi selon le QLIKE moyen walk-forward. Au holdout, son QLIKE doit
  être inférieur aux trois références et le Sharpe de sa stratégie de ciblage doit être au moins égal à celui
  du ciblage fondé sur la persistance. La règle commune du registre empêche de remplacer un champion par un
  candidat dont le QLIKE n'est pas meilleur sur le holdout.
- **Issue** : un candidat qui échoue au niveau 1 n'est pas enregistré. S'il passe le niveau 1 mais échoue au
  niveau 2, il est enregistré comme `challenger` ; s'il passe les deux niveaux mais ne bat pas le champion,
  il reste également `challenger`. Sinon, il devient `champion` et l'ancien champion devient `precedent`.
- **Challenger** : il prédit en parallèle sans produire de positions servies. La durée d'observation n'est pas
  fixée et aucune promotion automatique au terme d'une période de challenger n'est implémentée ; une promotion
  manuelle est possible, avec une raison obligatoire et une trace.
- **Santé quotidienne** : fenêtre de 7 jours. `rendement` est dégradé si son IC moyen est négatif ou si son PnL
  *paper* est négatif ; `volatilite` est dégradée si son QLIKE dépasse celui de la persistance.
- **Retour arrière automatique** : après trois jours consécutifs de santé dégradée, le flow tente de remettre
  l'alias `champion` sur `precedent`. Une alerte externe n'est pas envoyée par cette règle.
- **Réentraînement** : planifié chaque semaine. La dérive PSI n'est pas implémentée et ne déclenche pas encore
  d'entraînement automatique.

## 6. Retours, supervision et pilotage

- **Feedbacks** : les flows confrontent les prévisions aux rendements et volatilités réalisés dès que leur horizon
  est clôturé. Les résultats alimentent les métriques et la santé ; aucune vérité terrain n'est saisie à la main.
- **Traçabilité** : les prédictions, versions utilisées, résultats des flows, feedbacks et événements du registre
  sont conservés en base et dans MLflow selon leur rôle.
- **Inférence idempotente** : une seule prédiction par (symbole, bougie, version du modèle), garantie par une contrainte
  d'unicité ; relancer un flow ne crée pas de doublon.
- **Application Streamlit** : interface de démonstration et d'exploitation, cliente de l'API.

  | Page | Contenu |
  |---|---|
  | Projet | Présentation statique de la chaîne, accessible même si l'API est arrêtée |
  | Service | Santé des dépendances, **marche / arrêt** des flows et historique des bascules |
  | Prévisions | Dernière prévision par symbole, référence naïve et poids de position pour le modèle et l'alias choisis |
  | Modèles | Alias `champion`, `challenger` et `precedent`, santé quotidienne, événements, **promotion manuelle** et retour arrière |

  Principes :
  - Toute action passe par l'API et les changements (opérateur, motif, date) sont tracés ; Streamlit n'écrit
    pas directement en base.
  - L'interrupteur persistant arrête les flows planifiés au démarrage ; ils terminent alors avec le statut
    `cancelled`. Les déploiements peuvent aussi être gérés depuis Prefect.
  - L'application ne propose pas de modification libre des paramètres métier ni d'envoi d'alertes.
    Grafana est séparé et en lecture seule ; il expose quatre dashboards et six règles d'alerte dans son interface
    (échec de flow, lot rejeté, données périmées, prévisions absentes, modèle dégradé, système arrêté).
    Aucun canal externe de notification n'est configuré.

### Architecture et responsabilités

| Composant | Rôle dans le projet |
|---|---|
| Binance archives et API Spot | Fournir l'historique et les bougies récentes |
| Prefect | Orchestrer ingestion, entraînements, prévisions et feedbacks selon des déploiements UTC |
| Parquet local | Conserver les archives historiques téléchargées, sans les versionner dans Git |
| PostgreSQL | Stocker les bougies récentes, les prédictions, les feedbacks, l'état opérationnel et les traces |
| MLflow | Tracer les entraînements et gérer le registre ; son backend utilise PostgreSQL, pas SQLite |
| FastAPI | Exposer les prévisions et les opérations du système à l'application |
| Streamlit | Présenter les prévisions et permettre le pilotage via l'API |
| Grafana | Lire les données de supervision et les présenter dans quatre dashboards |

Les services sont assemblés avec Docker Compose. Le worker Prefect exécute les flows dans cet ordre horaire :

| Déploiement | Horaire UTC | Rôle |
|---|---|---|
| Ingestion | `2 * * * *` | Récupérer les bougies closes |
| Prévision volatilité | `7 * * * *` | Prévoir la volatilité 24 h |
| Prévision rendement | `8 * * * *` | Prévoir le rendement 4 h |
| Feedback volatilité | `20 * * * *` | Enregistrer les retours et la santé |
| Feedback rendement | `21 * * * *` | Enregistrer les retours et la santé |
| Entraînement rendement | `0 3 * * 1` | Réentraîner et appliquer les règles de registre |
| Entraînement volatilité | `30 3 * * 1` | Réentraîner et appliquer les règles de registre |

## 6 ter. Dimensionnement de l'API

*Décision du 2026-11-18* (protocole et chiffres complets : README, section « Dimensionnement »).

- **Charge cible** : 200 bots de trading qui relisent la dernière prévision et arrivent en rafale au début de
  chaque heure, plus 20 analystes et 2 opérateurs. SLO : p95 < 200 ms, moins de 1 % d'erreurs.
- **Règle** : plus petite configuration qui tient le pic et deux fois la cible sur une rampe jusqu'à rupture.
- **Choix : 1 processus uvicorn, 1 cœur** (défaut de `.env.example`) : p95 de 36 ms au pic, 500 bots tenus
  (2,5× la cible). 0,5 cœur casse dès 300 bots ; plus de processus coûte de la mémoire (+170 Mio chacun) pour
  peu ou pas de capacité en plus.
- **À réévaluer** si la cible change, ou sur une machine dédiée : la mesure partage les cœurs avec le client
  Locust, et le plafond observé (~300 req/s) semble venir du client ou du pool PostgreSQL plutôt que de l'API.

## 7. Synthèse OPS

Cette section regroupe les choix d'exploitation, leurs preuves et leurs limites. Le détail des commandes est dans
le [README](README.md).

### Décisions techniques

| Domaine | Choix | Raison |
|---|---|---|
| Assemblage | Docker Compose, 7 services avec healthcheck, `docker compose up -d --build --wait` | Un seul PC, démarrage reproductible en une commande |
| Base | PostgreSQL 18 unique, bases `mlops`, `prefect`, `mlflow` ; rôles `api_writer` (INSERT/SELECT) et `grafana_reader` (lecture seule) | Pas de SQLite ; moindre privilège, traces en ajout seul |
| Schéma | Scripts `infra/postgres/init/` et service `postgres-migrate` qui applique les scripts manquants (`public.schema_migrations`) | Un volume existant reçoit les nouvelles tables sans réinitialisation |
| Orchestration | Prefect 3, 7 déploiements planifiés en UTC, servis par le worker | Ordre horaire ingestion → prévisions → retours ; entraînements le lundi |
| Registre | MLflow (backend PostgreSQL), alias `champion` / `challenger` / `precedent`, promotion à deux niveaux, retour arrière après 3 jours dégradés | Cycle de vie traçable et réversible |
| Provenance | Code versionné par tag Git (hatch-vcs), image construite depuis un tag propre, archive du code et dépendances jointes à chaque run | Un modèle publié est relié à un code identifiable |
| Pilotage | Interrupteur persistant (`/system/state`) : les flows planifiés terminent en `cancelled` avec la raison | Arrêt d'urgence sans toucher à Docker |
| Supervision | Grafana provisionné (4 dashboards, 6 règles d'alerte), variable `Mode` live / replay | Configuration versionnée, aucune saisie manuelle |
| Rejeu | `python -m replay`, registre `<modèle>-replay`, traces `mode = 'replay'` | Montrer 12 mois de cycle de vie sans toucher au live |

### Preuves d'intégration

- **Tests** : 439 tests unitaires (`uv run pytest`, sans stack) et 70 tests d'intégration
  (`uv run pytest -m integration`, contre la stack réelle) : santé des briques, run MLflow → registre → alias → chargement, écritures PostgreSQL
  lues par Grafana, ingestion Binance de bout en bout (archive vérifiée par checksum, upsert idempotent),
  dashboards et alertes provisionnés, chaque requête SQL des dashboards exécutée par Grafana, lien entre run
  Prefect, run MLflow et traces.
- **Exploitation réelle** (18/11 13:17 → 19/11 14:20, heure de Paris) : 119 runs planifiés ; 420 lots
  d'ingestion acceptés et 5 rejetés (le seul run en échec, contrôle qualité bloquant) ; 125 prévisions
  `volatilite` et 90 `rendement`, 75 retours enregistrés. Les 4 runs `cancelled` sont attendus : 2 sans
  champion avant la première promotion manuelle, 1 test de l'interrupteur, 1 sans champion `rendement`.
  La santé quotidienne live commence 48 h après les premières prévisions (`HEALTH_DELAY`).
- **Rejeu** : 12 mois simulés (nov. 2025 → oct. 2026) avec le même code, visibles dans Grafana en mode `replay`.

### Résultats des tests de charge

Campagne Locust `2026-11-18T2120Z`, 6 configurations × 3 scénarios (normal, pic de début d'heure, rampe jusqu'à
rupture), SLO p95 < 200 ms et < 1 % d'erreurs. Toutes les erreurs restent sous 0,02 %. Avec 0,5 cœur, toutes les
configurations cassent dès 300 bots. Avec 1 cœur : 500 bots (1 processus), 600 bots (2 ou 4 processus). Tableau
complet : README, section « Dimensionnement » ; dashboard Grafana « Charge ».

### Dimensionnement recommandé

**API : 1 processus uvicorn, 1 cœur, ~200 Mio** (défaut de `.env.example`). Pic tenu avec p95 36 ms, rupture à
500 bots, soit 2,5 fois la cible de 200 bots. Doubler les processus n'ajoute que 100 bots pour deux fois plus de
mémoire. PostgreSQL reste sous 10 % de CPU.

### Limites connues

- Tout tourne sur un seul PC : pas de haute disponibilité, la collecte s'arrête si le PC ou Docker s'arrête ;
  client de charge et serveur partagent les cœurs.
- Plafond de ~300 req/s identique à 2 et 4 processus : limite probable côté client Locust ou pool de connexions
  (`API_DB_POOL_MAX=5`), non tranchée ; une seule campagne, paliers de 20 à 60 s.
- Routes de supervision (`/health`, `/models/{nom}`) lentes car elles interrogent MLflow.
- Pas de sauvegarde planifiée de la base ni des volumes (export manuel) ; pas de canal d'alerte externe ni de
  dérive PSI en production (voir section 8).

## 8. Limites et suites possibles

Les éléments ci-dessous ne sont pas présentés comme des fonctions livrées :

- **Dérive PSI de bout en bout** : pas de calcul PSI en production ni de déclenchement de réentraînement sur dérive.
- **Durée du challenger** : le challenger prédit en parallèle, mais aucune durée cible ni promotion automatique à
  l'issue d'une période d'observation n'est définie.
- **Classifieur `opportunite`** : troisième modèle optionnel, hors du périmètre implémenté.
- **Notifications externes** : six règles d'alerte Grafana sont visibles dans l'interface, mais aucun canal externe
  (courriel, Teams, webhook, etc.) n'est configuré.
- **Charge** : les résultats sont une campagne sur une machine où le client Locust partage les ressources avec la
  stack. Ils devront être reconfirmés sur l'infrastructure cible et après évolution de la charge.

Ces suites ne remettent pas en cause le circuit livré : ingestion, contrôles qualité, entraînement walk-forward,
registre MLflow, inférence API, feedback, supervision et pilotage sont déjà reliés.

## 9. Points à démontrer et expliquer

Pour la présentation, le projet peut être parcouru dans cet ordre :

1. Montrer un cycle nominal d'ingestion puis un lot volontairement invalide dans un environnement de démonstration :
   le lot doit être rejeté, tracé et visible dans Grafana. Expliquer aussi pourquoi la bougie du 2023-03-24 à
   12:00 UTC est écartée sans rejeter tout le mois.
2. Les deux entraînements comparent des candidats à des références sur des folds temporels, avec purge/embargo
   et un test final séparé ; les métriques n'impliquent pas une promesse de rentabilité.
3. Le registre conserve les versions et décisions ; le champion est servi, le challenger prédit en parallèle,
   et les promotions/retours arrière sont tracés.
4. Les retours alimentent la santé sur 7 jours ; trois jours dégradés consécutifs déclenchent une tentative de rollback.
5. L'API et l'application montrent les prévisions et permettent le pilotage ; Grafana expose traitements, modèles,
   service et charge.
6. La campagne Locust documentée dans le README illustre la capacité mesurée et ses limites, sans la présenter
   comme une garantie de performance en production.
