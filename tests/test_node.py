from pathlib import Path

from vgc import node


def test_parse_node_major() -> None:
    assert node._parse_node_major("v22.22.0") == 22
    assert node._parse_node_major("23.1.0") == 23
    assert node._parse_node_major("not-a-version") is None


def test_find_node_skips_old_executable(monkeypatch) -> None:
    old = Path("/old/node")
    modern = Path("/modern/node")
    monkeypatch.setattr(node, "_candidate_paths", lambda: [old, modern])
    monkeypatch.setattr(node, "_node_major", lambda path: {old: 16, modern: 22}[path])

    assert node.find_node() == str(modern)


def test_node_environment_prepends_selected_executable(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/old/bin:/usr/bin")

    environment = node.node_environment("/modern/bin/node")

    assert environment["PATH"] == "/modern/bin:/old/bin:/usr/bin"
