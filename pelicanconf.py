AUTHOR = 'Vedaang Chopra'
SITENAME = 'Vedaang Chopra'
SITEURL = ''

PATH = 'content'
ARTICLE_PATHS = ['writing']
USE_FOLDER_AS_CATEGORY = True
THEME = 'themes/minimalist'

TIMEZONE = 'America/New_York'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Articles
ARTICLE_URL = 'writing/{slug}.html'
ARTICLE_SAVE_AS = 'writing/{slug}.html'
ARTICLE_LANG_URL = 'writing/{slug}-{lang}.html'
ARTICLE_LANG_SAVE_AS = 'writing/{slug}-{lang}.html'

# Blog Index (now Writing)
INDEX_SAVE_AS = 'writing.html'
INDEX_URL = 'writing.html'

# Direct Templates
DIRECT_TEMPLATES = ['index', 'categories', 'authors', 'archives', 'home', 'projects']
HOME_SAVE_AS = 'index.html'
PROJECTS_SAVE_AS = 'projects.html'

# Pages
PAGE_URL = '{slug}.html'
PAGE_SAVE_AS = '{slug}.html'
PAGE_PATHS = ['pages', 'resume']

DISPLAY_PAGES_ON_MENU = False
DISPLAY_CATEGORIES_ON_MENU = False

# Menu
MENUITEMS = (('Resume', '/resume.html'),
             ('Writing', '/writing.html'),
             ('Uses', '/uses.html'),)

DEFAULT_PAGINATION = 10

# Load Projects Data
import yaml
import os
from datetime import datetime

CURRENT_YEAR = datetime.now().year

PROJECTS = []
try:
    projects_file = os.path.join(os.path.dirname(__file__), 'content', 'projects', 'projects.yml')
    if os.path.exists(projects_file):
        with open(projects_file, 'r') as f:
            PROJECTS = yaml.safe_load(f)
except Exception as e:
    print(f"Error loading projects: {e}")

# Social
SOCIAL = (('GitHub', 'https://github.com/Vedaang-Chopra'),
          ('LinkedIn', '#'),) # Add actual links later

# Plugins & Extensions
MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.codehilite': {'css_class': 'highlight'},
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {},
    },
    'output_format': 'html5',
}

# Static Paths
STATIC_PATHS = ['images', 'projects', 'resume', 'extra']

EXTRA_PATH_METADATA = {
    'extra/CNAME': {'path': 'CNAME'},
    'extra/.nojekyll': {'path': '.nojekyll'},
}
