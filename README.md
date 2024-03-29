# MORFITT

## Data ([Zenodo](https://zenodo.org/record/7893841#.ZFLFDnZBybg)) | Publication ([arXiv](TODO) / [HAL](https://hal.science/hal-04125879/document) / [ACL Anthology](TODO)) 
[Yanis LABRAK](https://www.linkedin.com/in/yanis-labrak-8a7412145/), [Richard DUFOUR](https://cv.hal.science/richard-dufour), [Mickaël ROUVIER](https://cv.hal.science/mickael-rouvier)

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/115EixHBcjf-se6xQeaTwZWE1i4idTNbm?usp=sharing) or [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://github.com/qanastek/MORFITT/blob/main/TrainTransformers.py)

We introduce MORFITT, the first multi-label corpus for the classification of specialties in the medical field, in French. MORFITT is composed of 3,624 summaries of scientific articles from PubMed, annotated in 12 specialties. The article details the corpus, the experiments and the preliminary results obtained using a classifier based on the pre-trained language model CamemBERT.

For more details, please refer to our paper:

**MORFITT: A multi-label topic classification for French Biomedical literature** ([arXiv](ddd) / [HAL](ddd) / [ACL Anthology](ddd))


# Key Features

## Documents distribution

| Train |  Dev  | Test  |
|-------|-------|-------|
| 1,514 | 1,022 | 1,088 |

## Multi-label distribution

|               | Train |  Dev  |  Test | Total |
|:----------------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|  Vétérinaire  |       320      |       250      |       254      |  824  |
|   Étiologie   |       317      |       202      |       222      |  741  |
|  Psychologie  |       255      |       175      |       179      |  609  |
|   Chirurgie   |       223      |       169      |       157      |  549  |
|   Génétique   |       207      |       139      |       159      |  505  |
|  Physiologie  |       217      |       125      |       148      |  490  |
| Pharmacologie |       112      |       84       |       103      |  299  |
| Microbiologie |       115      |       72       |       86       |  273  |
|  Immunologie  |       106      |       86       |       70       |  262  |
|     Chimie    |       94       |       53       |       65       |  212  |
|   Virologie   |       76       |       57       |       67       |  200  |
| Parasitologie |       68       |       34       |       50       |  152  |
|     Total     | 2,110 | 1,446 | 1,560 | 5,116 |


## Number of labels per document distribution

<p align="left">
  <img src="https://github.com/qanastek/MORFITT/raw/main/images/distributions_nbr_elements_colors.png" alt="drawing" width="400"/>
</p>

## Co-occurences distribution

<p align="left">
  <img src="https://github.com/qanastek/MORFITT/raw/main/images/distributions_co-references-fixed.png" alt="drawing" width="400"/>
</p>

# If you use HuggingFace Transformers

```python
from datasets import load_dataset
dataset = load_dataset("qanastek/MORFITT")
print(dataset)
```

or

```python
from datasets import load_dataset
dataset_base = load_dataset(
    'csv',
    data_files={
        'train': f"./train.tsv",
        'validation': f"./dev.tsv",
        'test': f"./test.tsv",
    },
    delimiter="\t",
)
```

# License and Citation

The code is under [Apache-2.0 License](./LICENSE).

The MORFITT dataset is licensed under *Attribution-ShareAlike 4.0 International* ([CC BY-SA 4.0](https://creativecommons.org/licenses/by/4.0/)).
If you find this project useful in your research, please cite the following papers:

```plain
Yanis Labrak, Mickaël Rouvier, Richard Dufour. MORFITT : A multi-label corpus of French scientific articles in the biomedical domain. 30e Conférence sur le Traitement Automatique des Langues Naturelles (TALN) Atelier sur l'Analyse et la Recherche de Textes Scientifiques, Florian Boudin, Jun 2023, Paris, France. ⟨hal-04125879⟩
```

or using the bibtex:


```bibtex
@inproceedings{labrak:hal-04125879,
  TITLE = {{MORFITT : A multi-label corpus of French scientific articles in the biomedical domain}},
  AUTHOR = {Labrak, Yanis and Rouvier, Micka{\"e}l and Dufour, Richard},
  URL = {https://hal.science/hal-04125879},
  BOOKTITLE = {{30e Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles (TALN) Atelier sur l'Analyse et la Recherche de Textes Scientifiques}},
  ADDRESS = {Paris, France},
  ORGANIZATION = {{Florian Boudin}},
  YEAR = {2023},
  MONTH = Jun,
  KEYWORDS = {BERT ; RoBERTa ; Transformers ; Biomedical ; Clinical ; Topics ; multi-labels ; BERT ; RoBERTa ; Transformers ; Biom{\'e}dical ; Clinique ; Sp{\'e}cialit{\'e}s ; multi-labels},
  PDF = {https://hal.science/hal-04125879/file/_ARTS___TALN_RECITAL_2023__MORFITT__Multi_label_topic_classification_for_French_Biomedical_literature%20%285%29.pdf},
  HAL_ID = {hal-04125879},
  HAL_VERSION = {v1},
}
```
     
