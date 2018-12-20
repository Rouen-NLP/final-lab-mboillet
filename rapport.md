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

Nous voyons ici que les catégories _Email_, _Form_, _Letter_ et _Memo_ sont sur-représentées. De plus, la catégorie _Resume_ est légèrement sous-représentée. Cela ne nous place pas dans les conditions optimales pour la suite. Nous aurions préféré avoir une distribution plus ou moins uniforme des catégories.

__Afin de contrer ce problème, nous n'allons pas prendre en compte tous les textes des catégories sur-représentées afin de nous approcher d'une distribution uniforme. Ainsi, nous obtenons la répartition suivante.__

### 1.2 Nombre de lettres moyen des textes par catégorie. 

![Letters](/images_rapport/letters.png)

Nous pouvons voir ici que deux catégories se distinguent nettement des autres. Il s'agit des catégories _News_ et _Note_ qui possèdent respectivement des moyennes proches de 3500 et 250 lettres. On peut donc supposer que la catégorie _News_ sera très peu confondue avec les catégories de petites moyennes : _Advertisement_, _Email_, _Form_ et _Note_. Il est en de même avec la catégorie _Note_ et les catégories de moyennes élevées : _News_, _Report_, _Resume_ et _Scientific_.

## 2. Analyse du problème et présentation de la solution.

La seconde étape consiste à analyser le problème et trouver une solution adaptée afin de le résoudre.

### 2.1 Analyse du problème.

Nous disposons d'environ 3500 textes appartenants à différentes catégories. Ces labellisé par des opérateurs, nous considérerons que l'erreur de labellisation est négligeable ici. Nous allons donc nous placer dans un cadre d'apprentissage supervisé afin de catégoriser automatique de nouveaux textes.

Pour commencer, nous allons découper l'ensemble des textes en trois ensembles _Train_, _Dev_ et _Test_ selon les proportions suivantes :

_Train_ | _Dev_ | _Test_
------------ | ------------- | -------------
60% | 20% | 20%

### 2.2 Solution envisagée.

Dans un premier temps, nous avons choisi d'utiliser un algorithme simple afin d'avoir des premiers résultats rapidement. Nous avons donc choisi d'implémenter un algorithme de classification naïve Bayésienne. Celui-ci a été choisi car il est adapté à l'apprentissage supervisé, est simple à mettre en oeuvre et donne de bons résultats rapidement.

De plus, nous avons d'abord choisi de représenter nos données comme des sacs de mots. Cette représentation pourra être améliorée par la suite.

### 2.3 Pseudo-code de l'algorithme.

```
Début:
  donnees = charger_donnees()
  train, dev, test = separer_donnees()
  train, dev, test = sac_de_mots(train, dev, test)
  train_prediction, dev_prediction, test_prediction = entrainer_tester_classifieur(train, dev, test)
Fin.
```

L'implémentation de cet algorithme se trouve dans le fichier ![TAIR_projet.py](/TAIR_projet.py). Une version de ce code est également disponible dans le document ![TAIR_projet.ipynb](/TAIR_projet.ipynb).
