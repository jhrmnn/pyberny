import datetime
import os
import subprocess
import sys

import toml

sys.path.insert(0, os.path.abspath('../src'))
with open('../pyproject.toml') as f:
    metadata = toml.load(f)['tool']['poetry']

project = 'PyBerny'
author = ' '.join(metadata['authors'][0].split()[:-1])
release = version = (
    subprocess.run(['poetry', 'version'], capture_output=True, cwd='..')
    .stdout.decode()
    .split()[1]
)
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
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
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
    'travis_button': True,
    'fixed_sidebar': True,
    'page_width': '60em',
}
html_sidebars = {
    '**': ['about.html', 'navigation.html', 'relations.html', 'searchbox.html']
}
html_static_path = ['_static']

autodoc_default_options = {'special-members': '__call__'}
autodoc_mock_imports = ['numpy']
todo_include_todos = True
pygments_style = 'sphinx'
napoleon_numpy_docstring = False
napoleon_use_ivar = True


def skip_namedtuples(app, what, name, obj, skip, options):
    if hasattr(obj, '_source'):
        return True


def setup(app):
    app.connect('autodoc-skip-member', skip_namedtuples)
