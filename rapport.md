# Classification des documents du procès des groupes américains du tabac
### _Mélodie Boillet - Text Analysis - 22/10/2018_

## Introduction 

Afin de faciliter l'exploitation de documents par les avocats dans le cadre de ce procès, nous devons mettre en place une classification automatique des types de documents. Pour cela, nous disposons d'un échantillon aléatoire de documents qui a été collecté et classifié par des opérateurs selon les catégories suivantes :
  * _Advertisement_,
  * _Email_,
  * _Form_,
  * _Letter_,
  * _Memo_, 
  * _News_, 
  * _Note_,
  * _Report_,
  * _Resume_ et
  * _Scientific_.

Nous allons donc mettre en place une méthodologie de projet de Machine Learning afin de prédire automatiquement les types des documents. 

Tout d'abord, nous analyserons les données dont nous disposons. Nous exposerons ensuite une solution afin de répondre à la problématique et présenterons une implémentation de cette solution. Enfin, nous analyserons les performances de cette solution et détaillerons les possibles pistes d'amélioration.

## 1. Analyse des données. 

La première étape à réaliser ici est d'étudier les données que nous avons à notre disposition. Pour cela, nous avons choisi de visualiser le nombre de textes par catégorie ainsi que le nombre de lettres moyen par type de texte.

### 1.1 Nombre de textes par catégorie. 
![Categories](./images_rapport/categories.png)

Nous voyons ici que les catégories _Email_, _Form_, _Letter_ et _Memo_ sont sur-représentées. De plus, la catégorie _Resume_ est légèrement sous-représentée. Nous ne sommes donc pas dans les conditions optimales pour la suite. Nous aurions préféré avoir une distribution plus ou moins uniforme des catégories. Nous n'en tiendrons pas compte pour la suite mais pourrons revenir sur les données et les uniformiser  plus tard.

### 1.2 Nombre de lettres moyen des textes par catégorie. 

![Letters](/images_rapport/letters.png)

Nous pouvons voir ici que deux catégories se distinguent nettement des autres. Il s'agit des catégories _News_ et _Note_ qui possèdent respectivement des moyennes proches de 3500 et 250 lettres. On peut donc supposer que la catégorie _News_ sera très peu confondue, au regard de cette caractéristique, avec les catégories de petites moyennes : _Advertisement_, _Email_, _Form_ et _Note_. Il est en de même avec la catégorie _Note_ et les catégories de moyennes élevées : _News_, _Report_, _Resume_ et _Scientific_.

## 2. Analyse du problème et présentation de la solution.

La seconde étape consiste à analyser le problème et trouver une solution adaptée afin de le résoudre.

### 2.1 Analyse du problème.

Nous disposons d'environ 3500 textes appartenant à différentes catégories. Ces textes ont été labellisés par des opérateurs, nous considérerons que l'erreur de labellisation est négligeable ici. Nous allons donc nous placer dans un cadre d'apprentissage supervisé afin de catégoriser automatiquement de nouveaux textes.

Pour commencer, nous allons découper l'ensemble des textes en trois ensembles _Train_, _Dev_ et _Test_ selon les proportions suivantes :

_Train_ | _Dev_ | _Test_
------------ | ------------- | -------------
60% | 20% | 20%

### 2.2 Solution envisagée.

Dans un premier temps, nous avons choisi d'utiliser un algorithme simple afin d'avoir des premiers résultats rapidement. Nous avons donc choisi d'implémenter un algorithme de classification naïve Bayésienne. Celui-ci a été choisi car il est adapté à l'apprentissage supervisé, est simple à mettre en oeuvre et donne de bons résultats rapidement.

De plus, nous avons d'abord choisi de représenter nos données comme des sacs de mots. Cette représentation pourra être améliorée par la suite.

### 2.3 Pseudo-code de l'algorithme.

L'idée globale de l'algorithme est présentée ci-dessous : 

```
Début:
  donnees = charger_donnees()
  train, dev, test = separer_donnees()
  train, dev, test = sac_de_mots(train, dev, test)
  train_prediction, dev_prediction, test_prediction = entrainer_tester_classifieur(train, dev, test)
Fin.
```

L'implémentation de cet algorithme se trouve dans le fichier ![TAIR_projet.py](/TAIR_projet.py). Une version de ce code est également disponible dans le document ![TAIR_projet.ipynb](/TAIR_projet.ipynb).

## 3. Analyse des performances.

### 3.1 Performances.
Après application de cet algorithme, nous obtenons les résultats suivants :

![BoW](/images_rapport/bow.PNG)

Nous pouvons voir que les résultats sont satisfaisants avec la simple représentation en sac de mots. En effet, nous obtenons presque 72% de textes correctement classifiés pour l'ensemble de tests. Cependant, nous aimerions augmenter cette valeur.

Afin d'améliorer ces résultats, nous avons testé de représenter les données en TF-IDF. Cette représentation permet de mettre moins de poids sur les mots qui apparaissent souvent dans de nombreux documents et qui ne sont pas déterminants.
Nous avons donc le nouvel algorithme suivant :
```
Début:
  donnees = charger_donnees()
  train, dev, test = separer_donnees()
  train, dev, test = sac_de_mots(train, dev, test)
  train_tf, dev_tf, test_tf = tfidf(train, dev, test)
  train_prediction, dev_prediction, test_prediction = entrainer_tester_classifieur(train_tf, dev_tf, test_tf)
Fin.
```

Avec cette nouvelle représentation, nous obtenons les résultats suivants :

![TF-IDF](/images_rapport/tfidf.PNG)

Les résultats obtenus sont moins satisfaisants que les premiers. En effet, nous perdons environ 5% de précision avec cette nouvelle représentation. Plusieurs raisons pourrait expliquer cela. Tout d'abord, il manque dans de nombreux textes des espaces entre certains mots, cela a un effet important sur notre représentation car les fréquences d'occurences de ces mots sont biaisées. De plus, nous possédons des documents qui n'apportent presque aucune information utile. C'est notamment le cas du document ci-dessous : 
```
as/@6/1998 12:45 +49-9131-856592 PSYUR. UNIV, Tea

oO .
Dy Thaveel J Er lense

 

(sr se sO 2!

SpZSLrsSOSe
```

Il n'y a aucun texte anglais dans ce document qui puisse être utile à notre modèle. De nombreux textes classés comme _Note_ ne contiennent pas de texte mais seulement des références, dates ou numéros. Ainsi, nous garderons notre premier algorithm avec les sac de mots.

### 3.2 Analyse des erreurs

La dernière étape à réaliser ici est d'analyser nos résultats et les erreurs. Cela devra permettre d'améliorer, plus tard, les résultats en passant par une possible correction des données ou un autre algorithme de classification.

Pour faire cette analyse, nous avons décider de calculer les erreurs de classification en fonction des catégories des documents. Nous obtenons l'analyse suivante pour l'ensemble de tests :

![Scores](/images_rapport/scores_test.PNG)

## Conclusion + pistes d'amélioration 
