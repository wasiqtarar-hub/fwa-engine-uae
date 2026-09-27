"""Role-gated page modules.

Each module exposes ``render()``. Pages are registered with Streamlit's router
by ``app/Home.py`` **only if the signed-in role holds the page's permission**,
so a page a role cannot use is not merely hidden — it does not exist for that
session (build brief §13.3).
"""
