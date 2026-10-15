"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

# Short page titles for generated API pages (``fsm``, not
# ``finite_selection.fsm``), so the sidebar stays readable.
configure(globals(), templates_path=["_templates"], add_module_names=False)
