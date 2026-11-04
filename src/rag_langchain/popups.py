"""Local SweetAlert2 confirmation component; no runtime CDN or parent-DOM injection."""

from pathlib import Path

import streamlit.components.v1 as components

_confirm = components.declare_component(
    "rag_confirm", path=str(Path(__file__).with_name("popup_frontend"))
)


def confirm_action(title, message, *, key):
    return _confirm(title=title, message=message, key=key, default=None)
