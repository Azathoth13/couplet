"""Sparse attention uses application extensions independently of SC providers."""
import sys
from types import ModuleType, SimpleNamespace

from llama3fhe.acceleration import multi_mac
from llama3fhe.acceleration.fhe_ext import bsgs, keyswitch
from llama3fhe.operators.attention import sparse_fused


def test_sparse_attention_uses_packaged_acceleration(monkeypatch):
    package = ModuleType("sylph_easy")
    package.__path__ = []
    extension = ModuleType("sylph_easy.easyfhe_ext")
    extension.bsgs = object()
    extension.keyswitch = SimpleNamespace(enabled=lambda: True)
    monkeypatch.setitem(sys.modules, "sylph_easy", package)
    monkeypatch.setitem(sys.modules, "sylph_easy.easyfhe_ext", extension)
    monkeypatch.delitem(sys.modules, "multi_mac", raising=False)
    monkeypatch.setattr(keyswitch, "enabled", lambda: True)
    modules = sparse_fused._modules()
    assert modules is not None
    assert modules[0] is multi_mac
    assert modules[1] is keyswitch
    assert modules[2] is bsgs
    assert modules[1] is not extension.keyswitch
