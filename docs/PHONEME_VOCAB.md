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
