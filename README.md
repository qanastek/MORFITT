# MORFITT

## Data ([Zenodo](ddd)) | Publication ([arXiv](ddd) / [HAL](ddd) / [ACL Anthology](ddd)) 
[Yanis LABRAK](https://www.linkedin.com/in/yanis-labrak-8a7412145/), [Richard DUFOUR](https://cv.hal.science/richard-dufour), [Mickaël ROUVIER](https://cv.hal.science/mickael-rouvier)

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/115EixHBcjf-se6xQeaTwZWE1i4idTNbm?usp=sharing) or [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://github.com/qanastek/MORFITT/blob/main/TrainTransformers.py)

We introduce MORFITT, the first multi-label corpus for the classification of specialties in the medical field, in French. MORFITT is composed of 3,624 summaries of scientific articles from PubMed, annotated in 12 specialties. The article details the corpus, the experiments and the preliminary results obtained using a classifier based on the pre-trained language model CamemBERT.

![overview](ddd)

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

The MORFITT dataset is licensed under *Creative Commons Attribution 4.0 International* ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
If you find this project useful in your research, please cite the following papers:

```plain
Yanis LABRAK & al. (COMMING SOON)
```

or using the bibtex:

```bibtex
@article{MORFITT,
}
```
     
