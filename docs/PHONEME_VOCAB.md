# Phoneme Vocabulary v0.1

`configs/phonemes/en_ptbr_phoneme_vocab.v0.1.json` is the first CTC output
inventory for the hotword branch. It is intentionally conservative:

- one global `<blank>` token for CTC;
- one global `<unk>` fallback token;
- CMUdict ARPAbet base phones with an `EN_` prefix;
- Montreal Forced Aligner Portuguese (Brazil) phones with a `PT_` prefix.

Token IDs are assigned by array index. The first two IDs are fixed:

```text
0: <blank>
1: <unk>
```

The v0.1 inventory has 81 total output classes:

```text
2 special tokens
39 English phones
40 Brazilian Portuguese phones
```

## English Normalization

English pronunciations should use CMUdict-style ARPAbet phones. CMUdict lexical
stress digits must be stripped before mapping into this vocabulary:

```text
AH0, AH1, AH2 -> EN_AH
OW0, OW1, OW2 -> EN_OW
ER0, ER1, ER2 -> EN_ER
```

This keeps the first CTC head smaller and avoids forcing the hotword branch to
learn stress distinctions before the basic phoneme pipeline is validated.

## Portuguese Normalization

Brazilian Portuguese pronunciations should use the MFA Portuguese (Brazil)
phone symbols and then add the `PT_` prefix. The vocabulary keeps nasal vowels
and common affricates as explicit symbols, for example:

```text
PT_ɐ̃
PT_ẽ
PT_õ
PT_tʃ
PT_dʒ
```

## Why Prefix By Language

The first training iteration keeps English and Portuguese phones distinct even
when their surface IPA symbols look similar. This makes target generation,
debugging, and hotword filtering simpler:

```text
English sample -> EN_* target phones
pt-BR sample   -> PT_* target phones
```

Later experiments can add a separate mapping from `EN_*` and `PT_*` symbols to a
shared IPA inventory for pronunciation similarity scoring.

## Precision IPA Vocabulary v0.2

`configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json` is the current
precision-prioritized candidate for the English, Latin American Spanish, and
Brazilian Portuguese CTC hotword branch.

This version changes the design target from "smallest practical output space" to
"better hotword and near-pronunciation recall". The vocabulary is still compact
relative to a BPE or LLM tokenizer, but compactness is not the primary goal. The
primary goal is to keep phones that may separate hotwords, proper names,
foreign-language terms, and near-sounding negative examples.

The v0.2 inventory has 90 total output classes:

```text
2 special tokens
88 shared IPA-like phone tokens
```

The first two IDs remain fixed:

```text
0: <blank>
1: <unk>
```

### Sources

The v0.2 vocabulary is built from mature MFA dictionaries:

- English: English (US) MFA dictionary v2.0.0;
- Spanish: Spanish (Latin America) MFA dictionary v2.0.0;
- Portuguese: Portuguese (Brazil) MFA dictionary v2.0.0.

MFA is used as the primary source because it is designed for forced alignment
and acoustic modeling, which is closer to our CTC supervision target than an LLM
tokenizer or a TTS-only phoneme frontend. CMUdict and Kokoro/Misaki remain useful
cross-checks, but they are not the authoritative source for this CTC inventory.

### Merge Method

The merge method follows multilingual CTC/global-phone-set practice and the
WHISTLE-style IPA phoneme alphabet idea:

```text
English MFA phones
Spanish Latin America MFA phones
Portuguese Brazil MFA phones
        ↓
Unicode / MFA IPA-like normalization
        ↓
exact-symbol union
        ↓
one shared CTC vocabulary
```

Language prefixes are intentionally not used:

```text
wrong: EN_a, ES_a, PT_a
right: a
```

Language and dialect are stored as sample metadata and hotword-registry fields,
not as part of the CTC token name.

### Precision Policy

Phones are not collapsed simply to reduce the output dimension. The following
contrasts are kept because they may matter for hotword recall and false-positive
control:

```text
i / ɪ
u / ʊ
e / ɛ
o / ɔ
r / ɾ
t / t̪
d / d̪
ʃ / ʒ
tʃ / dʒ
ɐ / ɐ̃
```

Brazilian Portuguese nasal vowels and nasal glides are kept:

```text
ɐ̃
ẽ
ĩ
õ
ũ
j̃
w̃
```

Spanish approximant/fricative phones such as `β` and `ɣ` are also kept in this
candidate rather than immediately mapping them back to `b` and `ɡ`. After the
real training set and FLEURS hotword registry are phonemized, token frequency
statistics should decide whether any rare or overly narrow phones should be
merged before a production training run.

### Reporting Summary

For a project report, the v0.2 design can be summarized as:

```text
The English-Spanish-Portuguese CTC phoneme vocabulary is built from mature MFA
phone inventories: English (US), Spanish (Latin America), and Portuguese
(Brazil). Following multilingual CTC global-phone-set and WHISTLE-style
phoneme supervision, the three inventories are normalized into an MFA IPA-like
symbol space and merged by exact phone identity. Identical phones share one CTC
ID, while language and dialect remain metadata rather than token prefixes.

Because the project objective is hotword and near-pronunciation recall, the
vocabulary keeps precision-relevant contrasts instead of minimizing output
classes. This produces a 90-class CTC output space over ln_post's 1024-dimensional
hidden states, suitable for training a dedicated hotword CTC branch on up to
roughly ten thousand hours of data.
```
