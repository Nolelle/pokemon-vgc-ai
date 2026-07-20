"""Unit tests for `selfplay.run_selfplay`'s pure scheduling logic
(`build_config_variants`/`discover_team_pool`/`build_game_specs`) -- no server, no
network, no `RecordingVgcPlayer` construction needed for any of these.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from selfplay.run_selfplay import (
    JITTER_FIELDS,
    JITTER_RANGE,
    N_JITTER_VARIANTS,
    build_config_variants,
    build_game_specs,
    discover_team_pool,
)

# --- build_config_variants: myopic/search/jittered, OTS always rejected -------------


def test_build_config_variants_has_expected_names() -> None:
    variants = build_config_variants(seed=0)
    assert "myopic" in variants
    assert "search" in variants
    jitter_names = [name for name in variants if name.startswith("jitter-")]
    assert len(jitter_names) == N_JITTER_VARIANTS


def test_build_config_variants_myopic_disables_search_search_enables_it() -> None:
    variants = build_config_variants(seed=0)
    assert variants["myopic"].use_two_ply_search is False
    assert variants["search"].use_two_ply_search is True


def test_build_config_variants_every_variant_rejects_open_team_sheets() -> None:
    variants = build_config_variants(seed=0)
    for name, config in variants.items():
        assert config.accept_open_team_sheet is False, name


def test_build_config_variants_jittered_weights_stay_within_range() -> None:
    variants = build_config_variants(seed=1)
    base = variants["search"]
    for name, config in variants.items():
        if not name.startswith("jitter-"):
            continue
        for field_name in JITTER_FIELDS:
            base_value = getattr(base, field_name)
            jittered_value = getattr(config, field_name)
            assert jittered_value == pytest.approx(base_value, rel=JITTER_RANGE + 1e-9)
        # Jittered variants still use the search path -- only the weights differ.
        assert config.use_two_ply_search is True


def test_build_config_variants_deterministic_for_same_seed() -> None:
    a = build_config_variants(seed=7)
    b = build_config_variants(seed=7)
    for name in a:
        for field_name in JITTER_FIELDS:
            assert getattr(a[name], field_name) == getattr(b[name], field_name)


def test_build_config_variants_different_seeds_usually_differ() -> None:
    a = build_config_variants(seed=1)
    b = build_config_variants(seed=2)
    differs = any(
        getattr(a["jitter-0"], field_name) != getattr(b["jitter-0"], field_name)
        for field_name in JITTER_FIELDS
    )
    assert differs


# --- discover_team_pool -------------------------------------------------------------


def test_discover_team_pool_includes_dev_and_sorted_pool_files(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "team_01.packed.txt").write_text("b")
    (pool_dir / "team_00.packed.txt").write_text("a")
    dev_path = tmp_path / "dev.packed.txt"
    dev_path.write_text("dev")

    paths = discover_team_pool(pool_dir, dev_path)

    assert paths[0] == dev_path
    assert paths[1:] == [pool_dir / "team_00.packed.txt", pool_dir / "team_01.packed.txt"]


def test_discover_team_pool_missing_dir_returns_dev_only(tmp_path: Path) -> None:
    dev_path = tmp_path / "dev.packed.txt"
    dev_path.write_text("dev")
    paths = discover_team_pool(tmp_path / "does-not-exist", dev_path)
    assert paths == [dev_path]


# --- build_game_specs: exact meta1 fraction guarantee, determinism -------------------


def test_build_game_specs_returns_empty_for_zero_games() -> None:
    assert (
        build_game_specs(
            0,
            seed=0,
            meta1_team_path=Path("m"),
            other_team_paths=[Path("a")],
            config_names=["search"],
        )
        == []
    )


def test_build_game_specs_meta1_appears_on_exactly_the_requested_fraction() -> None:
    meta1 = Path("meta1.packed.txt")
    others = [Path("dev.packed.txt"), Path("pool1.packed.txt")]
    specs = build_game_specs(
        20,
        seed=0,
        meta1_team_path=meta1,
        other_team_paths=others,
        config_names=["search", "myopic"],
        meta1_min_fraction=0.5,
    )
    assert len(specs) == 20
    meta1_games = [s for s in specs if meta1 in (s.p1_team_path, s.p2_team_path)]
    assert len(meta1_games) == 10  # ceil(20 * 0.5)
    # Every meta1 game has meta1 on EXACTLY one side (never both -- other_team_paths
    # never itself contains meta1_team_path in this test).
    for spec in meta1_games:
        assert (spec.p1_team_path == meta1) != (spec.p2_team_path == meta1)


def test_build_game_specs_rounds_up_fractional_meta1_count() -> None:
    specs = build_game_specs(
        7,
        seed=0,
        meta1_team_path=Path("meta1"),
        other_team_paths=[Path("dev")],
        config_names=["search"],
        meta1_min_fraction=0.5,
    )
    meta1_games = [s for s in specs if Path("meta1") in (s.p1_team_path, s.p2_team_path)]
    assert len(meta1_games) == 4  # ceil(7 * 0.5) == 4, not 3


def test_build_game_specs_non_meta1_games_never_include_meta1() -> None:
    meta1 = Path("meta1")
    others = [Path("dev"), Path("pool1")]
    specs = build_game_specs(
        20, seed=0, meta1_team_path=meta1, other_team_paths=others, config_names=["search"]
    )
    non_meta1_games = [s for s in specs if meta1 not in (s.p1_team_path, s.p2_team_path)]
    for spec in non_meta1_games:
        assert spec.p1_team_path in others
        assert spec.p2_team_path in others


def test_build_game_specs_deterministic_for_same_seed() -> None:
    kwargs = dict(
        n_games=15,
        seed=99,
        meta1_team_path=Path("meta1"),
        other_team_paths=[Path("dev"), Path("pool1"), Path("pool2")],
        config_names=["search", "myopic", "jitter-0"],
    )
    specs_a = build_game_specs(**kwargs)
    specs_b = build_game_specs(**kwargs)
    assert specs_a == specs_b


def test_build_game_specs_configs_drawn_from_provided_names() -> None:
    specs = build_game_specs(
        10,
        seed=0,
        meta1_team_path=Path("meta1"),
        other_team_paths=[Path("dev")],
        config_names=["only-config"],
    )
    assert all(s.p1_config_name == "only-config" for s in specs)
    assert all(s.p2_config_name == "only-config" for s in specs)


def test_build_game_specs_raises_on_empty_other_team_paths() -> None:
    with pytest.raises(ValueError):
        build_game_specs(
            5, seed=0, meta1_team_path=Path("meta1"), other_team_paths=[], config_names=["search"]
        )


def test_build_game_specs_raises_on_empty_config_names() -> None:
    with pytest.raises(ValueError):
        build_game_specs(
            5,
            seed=0,
            meta1_team_path=Path("meta1"),
            other_team_paths=[Path("dev")],
            config_names=[],
        )
