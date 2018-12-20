# Classification des documents du procès des groupes américains du tabac
### Mélodie Boillet - Text Analysis - 22/10/2018

## Introduction 

Afin de faciliter l'exploitation de documents par les avocats dans le cadre de ce procès, nous devons mettre en place une classification automatique des types de documents. Pour cela, nous disposons d'un échantillon aléatoire de documents qui a été collecté et classifié par des opérateurs selon les catégories suivantes :
  * Advertisement,
  * Email,
  * Form,
  * Letter,
  * Memo, 
  * News, 
  * Note,
  * Report,
  * Resume et
  * Scientific.

Nous allons donc mettre en place une méthodologie de projet de Machine Learning afin de prédire automatiquement les types des documents. 

Tout d'abord, nous analyserons les données dont nous disposons. Nous exposerons ensuite une solution afin de répondre à la problématique et présenterons une implémentation de cette solution. Enfin, nous analyserons les performances de cette solution et détaillerons les possibles pistes d'amélioration.

## 1. Analyse des données. 

La première étape à réaliser ici est d'étudier les données que nous avons à notre disposition. Pour cela, nous avons choisi de visualiser le nombre de textes par catégorie ainsi que le nombre de lettres moyen par type de texte.

![Categories](/images_rapport/categories)
<img src='/images_rapport/categories' alt="Categories" />



Nous voyons ici que les

![Letters](/images_rapport/letters)
