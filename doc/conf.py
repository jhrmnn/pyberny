#!/usr/bin/env python3
import datetime

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode'
]
source_suffix = '.rst'
master_doc = 'index'
project = 'pyberny'
author = 'Jan Hermann'
copyright = f'2017-{datetime.date.today().year}, {author}'
version = '0.1'
release = version
language = None
exclude_patterns = ['build', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = True
html_theme = 'alabaster'
html_theme_options = {
    'description': 'Molecular geometry optimizer',
    'github_button': True,
    'github_user': 'azag0',
    'github_repo': 'pyberny',
}
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
    ]
}
htmlhelp_basename = f'{project}doc'


def skip_namedtuples(app, what, name, obj, skip, options):
    if hasattr(obj, '_source'):
        return True


def setup(app):
    app.connect('autodoc-skip-member', skip_namedtuples)
