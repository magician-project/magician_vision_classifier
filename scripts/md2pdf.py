#!/usr/bin/env python3
"""Render a Markdown report to PDF.

Used for the dated deliverables in the repo root (`21-8-report.md`, `31-6-report.md`).
Images referenced from the Markdown are resolved relative to the source file, so a report
can link straight at `experiments/<campaign>/<run>/..._confusion_row_normalized.png` and
the plots land in the PDF.

WHY wkhtmltopdf AND NOT A PURE-PYTHON RENDERER. xhtml2pdf was tried first and got the
tables wrong in two ways that matter for these reports: a Markdown table with an empty
header row (`| | |`) collapsed to a pair of empty cells -- silently losing the whole
Method table -- and a long first column overlapped its neighbour instead of wrapping.
wkhtmltopdf is a real layout engine and gets both right. It needs to be installed:

    sudo apt-get install -y wkhtmltopdf

Usage:
    python scripts/md2pdf.py 21-8-report.md                # -> 21-8-report.pdf
    python scripts/md2pdf.py report.md out.pdf
"""

import argparse
import os
import shutil
import subprocess
import sys

try:
    import markdown
except ImportError:
    sys.exit('python-markdown is required:  pip install markdown')

# DejaVu carries no U+2B50; swap for U+2605 so the marker renders instead of a blank box.
GLYPH_SUBS = {'⭐': '★'}

CSS = """
html, body { font-family: "DejaVu Sans", sans-serif; font-size: 9.2pt; line-height: 1.45;
             color: #1a1a1a; margin: 0; }
h1 { font-size: 19pt; margin: 0 0 4px 0; }
h2 { font-size: 13pt; margin: 20px 0 6px 0; border-bottom: 1px solid #c4c4c4;
     padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 10.6pt; margin: 14px 0 5px 0; page-break-after: avoid; }
p  { margin: 0 0 7px 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.1pt;
       background: #f1f3f5; padding: 0 2px; border-radius: 2px; }
blockquote { margin: 9px 0; padding: 8px 12px; background: #f3f7fb;
             border-left: 3px solid #2b7bba; }
blockquote p { margin: 0; }

table { width: 100%; border-collapse: collapse; margin: 8px 0 12px 0;
        font-size: 7.6pt; page-break-inside: auto; }
thead { display: table-header-group; }          /* repeat header when a table spans pages */
tr    { page-break-inside: avoid; }
th    { background: #eaeef2; border: 1px solid #b6bcc2; padding: 3px 4px;
        text-align: left; font-size: 7.4pt; }
td    { border: 1px solid #d2d6da; padding: 3px 4px; vertical-align: top; }
td:first-child { white-space: nowrap; }         /* model names must not wrap mid-token */

img { max-width: 100%; }
/* Confusion matrices are dense; give them the full text width or the labels are unreadable. */
img[alt$="confusion matrix"] { width: 184mm; display: block; margin: 2px 0 2px -3mm;
                               page-break-inside: avoid; }
a  { color: #2b6ea8; text-decoration: none; }
hr { border: none; border-top: 1px solid #e2e2e2; margin: 14px 0; }
em { color: #444; }
"""


def render(src, out):
    if shutil.which('wkhtmltopdf') is None:
        sys.exit('wkhtmltopdf not found:  sudo apt-get install -y wkhtmltopdf')

    base = os.path.dirname(os.path.abspath(src)) or '.'
    md = open(src, encoding='utf-8').read()
    for a, b in GLYPH_SUBS.items():
        md = md.replace(a, b)
    body = markdown.markdown(md, extensions=['tables', 'fenced_code', 'sane_lists',
                                             'attr_list'])
    html = (f'<html><head><meta charset="utf-8"><base href="file://{base}/">'
            f'<style>{CSS}</style></head><body>{body}</body></html>')

    # Written beside the source so the <base href> and any relative image paths agree.
    tmp = os.path.join(base, '.md2pdf_render.html')
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(html)
    try:
        r = subprocess.run(
            ['wkhtmltopdf', '--enable-local-file-access', '--quiet',
             '--page-size', 'A4', '--margin-top', '14mm', '--margin-bottom', '15mm',
             '--margin-left', '13mm', '--margin-right', '13mm',
             '--footer-center', f'{os.path.basename(src)}  ·  page [page] of [topage]',
             '--footer-font-size', '7', '--footer-font-name', 'DejaVu Sans',
             '--footer-spacing', '5', '--encoding', 'utf-8', tmp, out],
            capture_output=True, text=True)
    finally:
        os.remove(tmp)

    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        sys.exit(f'wkhtmltopdf failed (rc={r.returncode})')
    print(f'wrote {out} ({os.path.getsize(out) / 1024 / 1024:.2f} MB)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', help='Markdown file to render')
    ap.add_argument('output', nargs='?', help='output PDF (default: alongside the source)')
    a = ap.parse_args()
    if not os.path.exists(a.source):
        sys.exit(f'{a.source} not found')
    render(a.source, a.output or a.source.rsplit('.', 1)[0] + '.pdf')


if __name__ == '__main__':
    main()
