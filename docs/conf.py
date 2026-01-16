# Configuration file for the Sphinx documentation builder.
import os
import sys
from datetime import date
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))
version = ET.parse("../package.xml").getroot()[1].text
print("Found version:", version)

project = "EmbodiedAgents"
copyright = f"{date.today().year}, Automatika Robotics"
author = "Automatika Robotics"
release = version

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_copybutton",  # install with `pip install sphinx-copybutton`
    "autodoc2",  # install with `pip install sphinx-autodoc2`
    "myst_parser",  # install with `pip install myst-parser`
    "sphinx_sitemap",  # install with `pip install sphinx-sitemap`
]

autodoc2_packages = [
    {
        "module": "agents",
        "path": "../agents",
        "exclude_dirs": ["__pycache__", "utils"],
        "exclude_files": [
            "callbacks.py",
            "publisher.py",
            "component_base.py",
            "model_component.py",
            "model_base.py",
            "db_base.py",
            "executable.py",
        ],
    },
]

autodoc2_docstrings = "all"
autodoc2_class_docstring = "both"  # bug in autodoc2, should be `merge`
autodoc2_render_plugin = "myst"
autodoc2_hidden_objects = ["private", "dunder", "undoc"]
autodoc2_module_all_regexes = [
    r"agents.config",
    r"agents.models",
    r"agents.vectordbs",
    r"agents.ros",
    r"agents.clients\.[^\.]+",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README*"]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_html_meta = {
    "google-site-verification": "cQVj-BaADcGVOGB7GOvfbkgJjxni10C2fYWCZ03jOeo"
}
myst_heading_anchors = 7  # to remove cross reference errors with md

html_baseurl = "https://automatika-robotics.github.io/embodied-agents/"
language = "en"
html_theme = "sphinx_book_theme"  # install with `pip install sphinx-book-theme`
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.png"
html_theme_options = {
    "logo": {
        "image_light": "_static/EMBODIED_AGENTS_LIGHT.png",
        "image_dark": "_static/EMBODIED_AGENTS_DARK.png",
    },
    "icon_links": [
        {
            "name": "Automatika",
            "url": "https://automatikarobotics.com/",
            "icon": "_static/automatika-logo.png",
            "type": "local",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/automatika-robotics/embodied-agents",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/B9ZU6qjzND",
            "icon": "fa-brands fa-discord",
        },
    ],
    "path_to_docs": "docs",
    "repository_url": "https://github.com/automatika-robotics/embodied-agents",
    "repository_branch": "main",
    "use_source_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}

# --- LLMS.TXT CONFIGURATION ---
# Defines the order of manual documentation for the curriculum
LLMS_TXT_SELECTION = [
    "intro.md",
    "installation.md",
    "quickstart.md",
    # Basics - The Core API
    "basics/components.md",
    "basics/clients.md",
    "basics/models.md",
    # Examples - Increasing complexity
    "examples/foundation/conversational.md",
    "examples/foundation/prompt_engineering.md",
    "examples/foundation/semantic_router.md",
    "examples/foundation/goto.md",
    "examples/foundation/semantic_map.md",
    "examples/foundation/tool_calling.md",
    "examples/foundation/complete.md",
    "examples/foundation/planning_model.md",
    "examples/foundation/vla.md",
    "examples/foundation/vla_with_event.md",
    "examples/events/multiprocessing.md",
    "examples/events/fallback.md",
    "examples/events/event_driven_description.md",
]


def format_for_llm(filename: str, content: str) -> str:
    """Helper to wrap content in a readable format for LLMs."""
    # Clean up HTML image tags to reduce noise
    lines = content.split("\n")
    cleaned_lines = [line for line in lines if "<img src=" not in line]
    cleaned_content = "\n".join(cleaned_lines).strip()

    return f"## File: {filename}\n```markdown\n{cleaned_content}\n```\n\n"


def generate_llms_txt(app, exception):
    """Generates llms.txt combining manual docs and autodoc2 API docs."""
    if exception is not None:
        return  # Do not generate if build failed

    print("[llms.txt] Starting generation...")

    src_dir = Path(app.srcdir)
    out_dir = Path(app.outdir)
    full_text = []

    # Add Preamble
    preamble = (
        "# EmbodiedAgents Documentation\n\n"
        "The following text contains the documentation for the EmbodiedAgents framework "
        "by Automatika Robotics. It is optimized for context ingestion.\n\n"
    )
    full_text.append(preamble)

    # Process Manual Docs (Curated List)
    print(f"[llms.txt] Processing {len(LLMS_TXT_SELECTION)} manual files...")
    for relative_path in LLMS_TXT_SELECTION:
        file_path = src_dir / relative_path
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            full_text.append(format_for_llm(relative_path, content))
        else:
            print(f"[llms.txt] Warning: Manual file not found: {relative_path}")

    # Write output to the build root
    output_path = out_dir / "llms.txt"
    try:
        output_path.write_text("".join(full_text), encoding="utf-8")
        print(f"[llms.txt] Successfully generated: {output_path}")
    except Exception as e:
        print(f"[llms.txt] Error writing file: {e}")


def copy_markdown_files(app, exception):
    """Copy source markdown files"""
    if exception is None:  # Only run if build succeeded
        # Source dir is where your .md files are
        src_dir = app.srcdir  # This points to your `source/` folder
        dst_dir = app.outdir  # This is typically `build/html`

        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".md"):
                    src_path = os.path.join(root, file)
                    # Compute path relative to the source dir
                    rel_path = os.path.relpath(src_path, src_dir)
                    # Destination path inside the build output
                    dst_path = os.path.join(dst_dir, rel_path)

                    # Make sure the target directory exists
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)


def create_robots_txt(app, exception):
    """Create robots.txt file to take advantage of sitemap crawl"""
    if exception is None:
        dst_dir = app.outdir  # Typically 'build/html/'
        robots_path = os.path.join(dst_dir, "robots.txt")
        content = f"""User-agent: *

Sitemap: {html_baseurl}/sitemap.xml
"""
        with open(robots_path, "w") as f:
            f.write(content)


def setup(app):
    """Plugin to post build and copy markdowns as well"""
    app.connect("build-finished", copy_markdown_files)
    app.connect("build-finished", create_robots_txt)
    app.connect("build-finished", generate_llms_txt)
