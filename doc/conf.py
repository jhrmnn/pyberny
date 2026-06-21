import datetime
import os
import sys
from importlib.metadata import version as get_version

import sphinxcontrib.katex as _katex
import toml
from docutils import nodes
from docutils.parsers.rst import roles

# Server-side KaTeX prerender so a math parse error fails the build.
# sphinxcontrib-katex's throwOnError=False default is overridden so a
# parse error raises instead of rendering as red HTML. Safe to keep
# unconditional: sphinx-multiversion (which reuses this conf.py against
# historical refs) only runs on push to master/tags in CI, and those
# sources are kept in sync with the gate.
katex_prerender = True
_katex.KATEX_DEFAULT_OPTIONS['throwOnError'] = True

sys.path.insert(0, os.path.abspath('../src'))
with open('../pyproject.toml') as f:
    metadata = toml.load(f)['tool']['poetry']

_HERE = os.path.abspath(os.path.dirname(__file__))

project = 'PyBerny'
author = ' '.join(metadata['authors'][0].split()[:-1])
release = version = get_version('pyberny')
description = metadata['description']
year_range = (2016, datetime.date.today().year)
year_str = (
    str(year_range[0])
    if year_range[0] == year_range[1]
    else f'{year_range[0]}-{year_range[1]}'
)
copyright = f'{year_str}, {author}'

master_doc = 'index'
extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinxcontrib.katex',
    'sphinx_multiversion',
]
intersphinx_mapping = {
    'python': (
        'https://docs.python.org/3',
        (
            'https://docs.python.org/3/objects.inv',
            os.path.join(_HERE, 'python-objects.inv'),
        ),
    ),
}
exclude_patterns = ['build', '.DS_Store']

html_theme = 'alabaster'
html_theme_options = {
    'description': description,
    'github_button': True,
    'github_user': 'jhrmnn',
    'github_repo': 'pyberny',
    'badge_branch': 'master',
    'codecov_button': True,
    'fixed_sidebar': True,
    'page_width': '60em',
}
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'versions.html',
    ]
}
html_static_path = ['_static']
html_css_files = ['pyberny.css']
templates_path = ['_templates']

smv_branch_whitelist = r'^master$'
smv_tag_whitelist = r'^\d+\.\d+\.\d+$'
smv_remote_whitelist = r'^origin$'

autodoc_default_options = {'special-members': '__call__'}
# molsym is imported at module top by berny.symmetry and runs numpy at import
# time; mock it (like numpy) so autodoc can import berny without executing it.
autodoc_mock_imports = ['numpy', 'molsym']
todo_include_todos = True
pygments_style = 'sphinx'
napoleon_numpy_docstring = False
napoleon_use_ivar = True


def skip_namedtuples(app, what, name, obj, skip, options):
    if hasattr(obj, '_source'):
        return True
    return None


def _callout_role(css_class):
    """Inline role that wraps content in <span class=css_class> while still
    parsing inline markup (citations, refs, code spans, etc.) inside the
    body. A bare `.. role:: foo` directive instead treats its content as
    plain text, which silently breaks links like `[BirkholzTCA16]_`.
    """

    def role(name, rawtext, text, lineno, inliner, options=None, content=None):
        opts = (options or {}).copy()
        opts.setdefault('classes', []).append(css_class)
        children, messages = inliner.parse(text, lineno, inliner, inliner.parent)
        return [nodes.inline(rawtext, '', *children, **opts)], messages

    return role


def setup(app):
    app.connect('autodoc-skip-member', skip_namedtuples)
    roles.register_local_role('sm', _callout_role('sm'))
    roles.register_local_role('pb', _callout_role('pb'))
