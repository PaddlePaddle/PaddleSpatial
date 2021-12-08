# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'PaddleSpatial'
author = u'2021, Baidu Inc.'
copyright = author


# The master toctree document.
master_doc = 'index'

# The suffix of source filenames.
source_suffix = '.rst'


release = '0.1'
version = '0.1.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

# Import mock dependencies packages
autodoc_mock_imports = ['paddle', 'pgl']




intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
