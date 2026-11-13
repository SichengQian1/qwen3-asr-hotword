---
language:
- en
pipeline_tag: other
library_name: montreal-forced-aligner
tags:
- montreal-forced-aligner
- speech-to-text-alignment
- forced-alignment
license: cc-by-4.0
---
# Model Card for english_mfa

<!-- Provide a quick summary of what the model is/does. -->

This MFA model is for aligning Global English across multiple dialects.

- [Model details](#model-details)
- [Uses](#uses)
- [Performance Factors](#how-to-get-started-with-the-model)
- [Dictionary Details](#dictionary-details)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Contact](#contact)

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** Michael McAuliffe
- **Funded by:** N/A
- **Model type:** Montreal Forced Aligner model
- **Language(s) (NLP):** English
- **License:** cc-by-4.0

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model is intended to be used for forced alignment of English varieties that it was trained on (US, UK, Indian, Nigerian Englishes).  Please see https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/troubleshooting.html for details on common fixes.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model cannot provide accurate assessments of goodness of pronunciations or provide transcripts as it is trained to be accepting of variation in pronunciation to provide a reasonable alignment across many varieties of English.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

This model will perform best on the variety of speech that it was trained on, with a bias towards US English.  The speakers in the training data are all adult speakers, so child speech alignment may not be accurate.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

When using this model on a variety that it was not trained on, better results can be attained by adapting the model to the data to be aligned first.  See https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/adapt_acoustic_model.html and https://github.com/mmcauliffe/mfa-adaptation for example usage and scripts.

## How to Get Started with the Model

Use the code below to get started with the model.

To get started, follow the instructions for [installing MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html). To align files using this model, use the [mfa align](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html) command.

## Dictionary Details

#### Details for english_nigeria_mfa dictionary and G2P model

- **Source:** wikipron
- **Orthography:** Latin
- **Phone set:** MFA
- **Words:** 77,407
* **Phones:** 66
* **Graphemes:** 46

##### IPA chart

###### Consonants

| Manner | Labial | Labiodental | Dental | Alveolar | Alveopalatal | Palatal | Velar | Glottal |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| **Nasal** | m | ɱ |  | n |  | ɲ | ŋ | |
| Palatalized | mʲ |  |  |  |  |  |  | |
| **Stop** | p     b |  | t̪     d̪ | t     d |  | c     ɟ | k     ɡ | |
| Aspirated | pʰ |  |  | tʰ |  | cʰ | kʰ | |
| Labialized |  |  |  | tʷ |  | cʷ     ɟʷ | kʷ     ɡʷ | |
| Palatalized | pʲ     bʲ |  |  | tʲ     dʲ |  |  |  | |
| **Affricate** |  |  |  |  | tʃ     dʒ |  |  | |
| **Sibilant** |  |  |  | s     z | ʃ |  |  | |
| **Fricative** |  | f     v | θ     ð |  |  | ç |  | h|
| Palatalized |  | fʲ     vʲ |  |  |  |  |  | |
| **Approximant** | w |  |  | ɹ |  | j |  | |
| **Lateral** |  |  |  | l ɫ |  | ʎ |  |  |

###### Vowels

|  | Front | Near-Front | Central | Near-Back | Back |
| :----: | :----: | :----: | :----: | :----: | :----: |
| **Close** | i iː |  |  |  | u uː|
|  |  |  |  | ʊ | |
| **Close-Mid** | e |  |  |  | o|
|  |  |  |  |  | |
| **Open-Mid** | ɛ ɛː |  | ɜ ɜː |  | ɔ|
|  |  |  |  |  | |
| **Open** |  |  | a aː |  |  |

##### Diphthongs
* aj
* ɔj
* aw


#### Other
* kp


#### Details for english_india_mfa dictionary and G2P model

- **Source:** wikipron
- **Orthography:** Latin
- **Phone set:** MFA
- **Words:** 123,926
* **Phones:** 64
* **Graphemes:** 62

##### IPA chart

###### Consonants

| Manner | Labial | Labiodental | Dental | Alveolar | Alveopalatal | Retroflex | Palatal | Velar | Glottal |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| **Nasal** | m | ɱ |  | n |  |  | ɲ | ŋ | |
| Palatalized | mʲ |  |  |  |  |  |  |  | |
| **Stop** | p     b |  | t̪     d̪ |  |  | ʈ     ɖ | c     ɟ | k     ɡ | ʔ|
| Labialized | pʷ |  |  |  |  | ʈʷ | cʷ     ɟʷ | kʷ     ɡʷ | |
| Palatalized | pʲ     bʲ |  |  |  |  | ʈʲ |  |  | |
| **Affricate** |  |  |  |  | tʃ     dʒ |  |  |  | |
| **Sibilant** |  |  |  | s     z | ʃ     ʒ |  |  |  | |
| **Fricative** |  | f |  |  |  |  | ç |  | h|
| Palatalized |  | fʲ |  |  |  |  |  |  | |
| **Approximant** |  | ʋ |  | ɹ |  |  | j |  | |
| **Tap** |  |  |  | ɾ |  |  |  |  | |
| **Lateral** |  |  |  | l |  |  | ʎ |  |  |

###### Vowels

|  | Front | Near-Front | Central | Near-Back | Back |
| :----: | :----: | :----: | :----: | :----: | :----: |
| **Close** | i iː |  | ʉ ʉː |  | |
|  |  | ɪ |  | ʊ | |
| **Close-Mid** | eː |  |  |  | oː|
|  |  |  | ə |  | |
| **Open-Mid** | ɛ ɛː |  | ɜ ɜː |  | |
|  |  |  |  |  | |
| **Open** |  |  | a aː |  | ɑ ɑː     ɒ ɒː |

##### Diphthongs
* aj
* ɔj
* aw


#### Details for english_uk_mfa dictionary and G2P model

- **Source:** wikipron
- **Orthography:** Latin
- **Phone set:** MFA
- **Words:** 186,963
* **Phones:** 78
* **Graphemes:** 88

##### IPA chart

###### Consonants

| Manner | Labial | Labiodental | Dental | Alveolar | Alveopalatal | Palatal | Velar | Glottal |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| **Nasal** | m m̩ | ɱ |  | n n̩ |  | ɲ | ŋ | |
| Palatalized | mʲ |  |  |  |  |  |  | |
| **Stop** | p     b |  | t̪     d̪ | t     d |  | c     ɟ | k     ɡ | ʔ|
| Aspirated | pʰ |  |  | tʰ |  | cʰ | kʰ | |
| Labialized | pʷ |  |  | tʷ |  | cʷ     ɟʷ | kʷ     ɡʷ | |
| Palatalized | pʲ     bʲ |  |  | tʲ     dʲ |  |  |  | |
| **Affricate** |  |  |  |  | tʃ     dʒ |  |  | |
| **Sibilant** |  |  |  | s     z | ʃ     ʒ |  |  | |
| **Fricative** |  | f     v | θ     ð |  |  | ç |  | h|
| Labialized |  | vʷ |  |  |  |  |  | |
| Palatalized |  | fʲ     vʲ |  |  |  |  |  | |
| **Approximant** | w |  |  | ɹ |  | j |  | |
| **Lateral** |  |  |  | l ɫ ɫ̩ |  | ʎ |  |  |

###### Vowels

|  | Front | Near-Front | Central | Near-Back | Back |
| :----: | :----: | :----: | :----: | :----: | :----: |
| **Close** | i iː |  | ʉ ʉː |  | |
|  |  | ɪ |  | ʊ | |
| **Close-Mid** | e ej |  |  |  | |
|  |  |  | ə |  | |
| **Open-Mid** | ɛ ɛː |  | ɜ ɜː |  | |
|  |  |  | ɐ |  | |
| **Open** |  |  | a |  | ɑ ɑː     ɒ ɒː |

##### Diphthongs
* ej
* əw
* ɔj
* aw
* aj


#### Details for english_us_mfa dictionary and G2P model

- **Source:** wikipron
- **Orthography:** Latin
- **Phone set:** MFA
- **Words:** 251,336
* **Phones:** 78
* **Graphemes:** 87

##### IPA chart

###### Consonants

| Manner | Labial | Labiodental | Dental | Alveolar | Alveopalatal | Palatal | Velar | Glottal |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| **Nasal** | m m̩ | ɱ |  | n n̩ |  | ɲ | ŋ | |
| Palatalized | mʲ |  |  |  |  |  |  | |
| **Stop** | p     b |  | t̪     d̪ | t     d |  | c     ɟ | k     ɡ | ʔ|
| Aspirated | pʰ |  |  | tʰ |  | cʰ | kʰ | |
| Labialized | pʷ |  |  | tʷ |  | cʷ     ɟʷ | kʷ     ɡʷ | |
| Palatalized | pʲ     bʲ |  |  | tʲ     dʲ |  |  |  | |
| **Affricate** |  |  |  |  | tʃ     dʒ |  |  | |
| **Sibilant** |  |  |  | s     z | ʃ     ʒ |  |  | |
| **Fricative** |  | f     v | θ     ð |  |  | ç |  | h|
| Palatalized |  | fʲ     vʲ |  |  |  |  |  | |
| **Approximant** | w |  |  | ɹ |  | j |  | |
| **Tap** |  |  |  | ɾ ɾ̃ |  |  |  | |
| Palatalized |  |  |  | ɾʲ |  |  |  | |
| **Lateral** |  |  |  | l ɫ ɫ̩ |  | ʎ |  |  |

###### Vowels

|  | Front | Near-Front | Central | Near-Back | Back |
| :----: | :----: | :----: | :----: | :----: | :----: |
| **Close** | i iː |  | ʉ ʉː |  | |
|  |  | ɪ |  | ʊ | |
| **Close-Mid** | ej |  |  |  | ow|
|  |  |  | ə ɚ |  | |
| **Open-Mid** | ɛ |  | ɝ |  | |
|  | æ |  | ɐ |  | |
| **Open** |  |  |  |  | ɑ ɑː     ɒ ɒː |

##### Diphthongs
* ej
* ɔj
* aw
* ow
* aj


## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

#### ARU English corpus

- **Source:** http://datacat.liverpool.ac.uk/681/
- **License:** CC BY 3.0
- **Dialects:** uk
- **Number of hours:** 7.13
- **Number of utterances:** 8,640
- **Number of speakers:** 12
  - **Female speakers:** 0
  - **Male speakers:** 0
  - **Unknown speakers:** 0


#### Common Voice English

- **Source:** https://voice.mozilla.org/en/datasets
- **License:** CC-0
- **Dialects:** us, uk, nigeria, india
- **Number of hours:** 2,322.80
- **Number of utterances:** 1,625,987
- **Number of speakers:** 71,160
  - **Female speakers:** 3,750
  - **Male speakers:** 14,586
  - **Unknown speakers:** 52,824


#### Corpus of Regional African American Language

- **Source:** https://oraal.uoregon.edu/coraal
- **License:** CC BY-NC-SA 4.0
- **Dialects:** us
- **Number of hours:** 124.31
- **Number of utterances:** 236,792
- **Number of speakers:** 193
  - **Female speakers:** 88
  - **Male speakers:** 94
  - **Unknown speakers:** 11


#### Google Nigerian English

- **Source:** https://openslr.org/70/
- **License:** CC BY-SA 4.0
- **Dialects:** nigeria
- **Number of hours:** 5.77
- **Number of utterances:** 3,359
- **Number of speakers:** 31
  - **Female speakers:** 19
  - **Male speakers:** 12
  - **Unknown speakers:** 0


#### Google UK and Ireland English

- **Source:** https://openslr.org/83/
- **License:** CC BY-SA 4.0
- **Dialects:** uk
- **Number of hours:** 31.29
- **Number of utterances:** 17,877
- **Number of speakers:** 120
  - **Female speakers:** 49
  - **Male speakers:** 71
  - **Unknown speakers:** 0


#### ICE-Nigeria

- **Source:** https://sourceforge.net/projects/ice-nigeria/
- **License:** CC BY-NC-SA 3.0
- **Dialects:** nigeria
- **Number of hours:** 52.86
- **Number of utterances:** 113,664
- **Number of speakers:** 1,276
  - **Female speakers:** 293
  - **Male speakers:** 853
  - **Unknown speakers:** 130


#### L2-ARCTIC

- **Source:** https://psi.engr.tamu.edu/l2-arctic-corpus/
- **License:** CC BY-NC 4.0
- **Dialects:** N/A
- **Number of hours:** 27.51
- **Number of utterances:** 27,042
- **Number of speakers:** 24
  - **Female speakers:** 12
  - **Male speakers:** 12
  - **Unknown speakers:** 0


#### LibriSpeech English

- **Source:** https://openslr.org/12/
- **License:** CC BY 4.0
- **Dialects:** us
- **Number of hours:** 982.10
- **Number of utterances:** 292,367
- **Number of speakers:** 2,484
  - **Female speakers:** 1,283
  - **Male speakers:** 1,201
  - **Unknown speakers:** 0


#### NCHLT English

- **Source:** https://repo.sadilar.org/handle/20.500.12185/274
- **License:** CC BY 3.0
- **Dialects:** N/A
- **Number of hours:** 56.43
- **Number of utterances:** 77,412
- **Number of speakers:** 210
  - **Female speakers:** 0
  - **Male speakers:** 0
  - **Unknown speakers:** 210


#### A Scripted Pakistani English Daily-use Speech Corpus

- **Source:** https://magichub.com/datasets/pakistani-english-scripted-speech-corpus-daily-use-sentence/
- **License:** CC BY-NC-ND 4.0
- **Dialects:** india
- **Number of hours:** 4.00
- **Number of utterances:** 2,191
- **Number of speakers:** 7
  - **Female speakers:** 3
  - **Male speakers:** 4
  - **Unknown speakers:** 0


### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

Preprocessing include fixes and orthographic standardization to various corpora.


#### Training Hyperparameters

- **Training regime:** [Training configuration](config.yaml)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

N/A

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

N/A

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

N/A

### Results

N/A

#### Summary



## Technical Specifications

### Model Architecture and Objective

HMM-GMM model

#### Software

This model was trained via the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/).

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```
@techreport{mfa_english_mfa_acoustic_2026,
	author={McAuliffe, Michael and Sonderegger, Morgan},
	title={Global English MFA acoustic model v3.3.0},
	address={\url{https://huggingface.co/MontrealCorpusTools/english_mfa}},
	year={2026},
	month={Jun},
}
```

**APA:**

```
McAuliffe, M. & Sonderegger, M. (2026). Global English MFA acoustic model v3.3.0. Available at https://huggingface.co/MontrealCorpusTools/english_mfa.
```

## Contact

For questions and issues, please file an issue either for this model at https://huggingface.co/MontrealCorpusTools/english_mfa/discussions or for larger MFA issues at https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues.