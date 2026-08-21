import csv
from pathlib import Path

import pytest

from qwen_hotword.training.spanish_mfa_repair import (
    finalize_shared_spanish_mfa_repair,
    prepare_shared_spanish_mfa_repair,
    repair_spanish_pronunciation,
    spanish_g2p_proxy,
)

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json")


def _write_g2p_dir(path: Path, counts: dict[str, int]) -> None:
    path.mkdir(parents=True)
    (path / "words.txt").write_text(
        "".join(f"{word}\n" for word in sorted(counts)), encoding="utf-8"
    )
    with (path / "word_counts.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["word", "count"])
        for word, count in sorted(counts.items()):
            writer.writerow([word, count])


def test_spanish_proxy_rules_preserve_enye_and_diaeresis() -> None:
    assert spanish_g2p_proxy("años") == (
        "anios",
        ("enye_via_ni",),
        None,
    )
    assert spanish_g2p_proxy("pingüino") == (
        "pingwino",
        ("diaeresis_via_gw",),
        None,
    )
    assert spanish_g2p_proxy("está") == ("esta", ("strip_acute",), None)
    assert spanish_g2p_proxy("bilingüe") == (
        "bilingwe",
        ("diaeresis_via_gw",),
        None,
    )
    assert spanish_g2p_proxy("normal")[2] == "no_safe_proxy"


def test_spanish_pronunciation_cleanup_is_rule_scoped() -> None:
    repaired, counts = repair_spanish_pronunciation(
        "a ɲ j õ s", ("enye_via_ni",)
    )
    assert repaired == "a ɲ o s"
    assert counts == {
        "combining_tilde_removed": 1,
        "enye_proxy_glide_removed": 1,
    }

    untouched_glide, _ = repair_spanish_pronunciation("a ɲ j o s", ())
    assert untouched_glide == "a ɲ j o s"


def test_shared_spanish_repair_reuses_base_and_runs_one_proxy_dictionary(
    tmp_path: Path,
) -> None:
    slr_g2p = tmp_path / "slr" / "mfa_g2p"
    cv_g2p = tmp_path / "cv" / "mfa_g2p"
    _write_g2p_dir(
        slr_g2p,
        {"años": 75, "más": 329, "normal": 3, "pingüino": 2},
    )
    _write_g2p_dir(
        cv_g2p,
        {"está": 311, "mañana": 7, "señor": 4, "vergüenza": 2},
    )
    slr_dictionary = tmp_path / "slr.dict"
    cv_dictionary = tmp_path / "cv.dict"
    slr_dictionary.write_text("mas m ã s\nnormal n o ɾ m a l\n", encoding="utf-8")
    cv_dictionary.write_text("esta e s t a\n", encoding="utf-8")
    corpora = {
        "slr61": (slr_g2p, slr_dictionary),
        "common_voice_rioplatense_v26": (cv_g2p, cv_dictionary),
    }

    repair_root = tmp_path / "repair"
    prepared = prepare_shared_spanish_mfa_repair(corpora, repair_root)

    assert prepared["shared_proxy_words"] == 5
    assert (repair_root / "proxy_words.txt").read_text(encoding="utf-8").splitlines() == [
        "anios",
        "maniana",
        "pingwino",
        "senior",
        "vergwenza",
    ]
    assert prepared["corpora"]["slr61"]["resolution_counts"] == {
        "base_exact": 1,
        "base_proxy": 1,
        "mfa_proxy": 2,
    }

    proxy_dictionary = repair_root / "proxy.dict"
    proxy_dictionary.write_text(
        "anios a ɲ j õ s\n"
        "maniana m a ɲ j a n a\n"
        "pingwino p ĩ ŋ ɡ w i n o\n"
        "senior s e ɲ j o ɾ\n"
        "vergwenza b e ɾ ɣ w ẽ n s a\n",
        encoding="utf-8",
    )
    output = tmp_path / "finalized"
    finalized = finalize_shared_spanish_mfa_repair(
        corpora,
        repair_root,
        proxy_dictionary,
        VOCAB,
        output,
    )

    assert finalized["training_labels_ready"] is True
    for name in corpora:
        assert finalized["corpora"][name]["audit"]["missing_words"] == 0
        assert finalized["corpora"][name]["audit"]["oov_phone_units"] == 0
    slr_repaired = (
        output / "slr61" / "slr61_spanish_latin_america_repaired.v1.dict"
    ).read_text(encoding="utf-8")
    assert "años\ta ɲ o s\n" in slr_repaired
    assert "más\tm a s\n" in slr_repaired
    assert "pingüino\tp i ŋ ɡ w i n o\n" in slr_repaired
    assert "anios\t" not in slr_repaired
    assert "\u0303" not in slr_repaired
    assert (output / "sha256.txt").is_file()


def test_shared_spanish_repair_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="absent or empty"):
        prepare_shared_spanish_mfa_repair(
            {"slr61": (tmp_path / "missing", tmp_path / "missing.dict")},
            output,
        )
