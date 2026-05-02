"""
Render the markdown summaries into clean styled HTML you can open in a browser
(and screenshot for a mentor / write-up).

By default scans both summary folders and converts every .md it finds:
  custom_automation/analysis/phase2_cap_results/*.md           → .html
  custom_automation/analysis/neuronpedia_validation_results/*.md → .html

The HTML uses inline CSS (no external deps), so the files are self-contained
and travel cleanly via email or chat.

Usage:
    # Render every summary in both result folders.
    python custom_automation/analysis/render_html.py

    # Render specific files only.
    python custom_automation/analysis/render_html.py path/to/foo.md path/to/bar.md

    # Output to a different directory than next to the source file.
    python custom_automation/analysis/render_html.py --out my_html/
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_DIRS = [
    ANALYSIS_DIR / "phase2_cap_results",
    ANALYSIS_DIR / "neuronpedia_validation_results",
]


# ---------------------------------------------------------------------------
# Markdown → HTML (just the subset we use: # headings, tables, paragraphs)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^[-+]?\d[\d,_]*(\.\d+)?\s*%?\s*(±\s*\d+(\.\d+)?\s*%?)?$")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


def _is_numeric(cell: str) -> bool:
    s = cell.strip()
    if not s or s == "—" or s == "-":
        return True
    return bool(_NUM_RE.match(s))


def _render_inline(text: str) -> str:
    """Markdown-ish inline: `code`, **bold**, plain text. Escape HTML first."""
    safe = html.escape(text)
    safe = _BOLD_RE.sub(r"<strong>\1</strong>", safe)
    safe = _INLINE_CODE_RE.sub(r"<code>\1</code>", safe)
    return safe


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-+:?", c.strip()) for c in cells if c.strip())


def _row_alignments(separator_cells: list[str]) -> list[str]:
    out: list[str] = []
    for c in separator_cells:
        c = c.strip()
        if c.startswith(":") and c.endswith(":"):
            out.append("center")
        elif c.endswith(":"):
            out.append("right")
        elif c.startswith(":"):
            out.append("left")
        else:
            out.append("left")
    return out


def _split_pipe_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def md_to_html(md_text: str) -> tuple[str, str]:
    """Return (title, body_html). Title comes from the first H1 heading."""
    lines = md_text.splitlines()
    out: list[str] = []
    title = ""
    i = 0
    while i < len(lines):
        line = lines[i]

        # Heading
        h_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if h_match:
            level = len(h_match.group(1))
            text = h_match.group(2).strip()
            if level == 1 and not title:
                title = text
            out.append(f"<h{level}>{_render_inline(text)}</h{level}>")
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^-{3,}\s*$", line):
            out.append("<hr/>")
            i += 1
            continue

        # Table: looks ahead for `| --- | --- |` separator
        if line.strip().startswith("|") and i + 1 < len(lines):
            header_cells = _split_pipe_row(line)
            sep_cells = _split_pipe_row(lines[i + 1])
            if _is_separator_row(sep_cells) and len(header_cells) == len(sep_cells):
                aligns = _row_alignments(sep_cells)
                rows: list[list[str]] = []
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    rows.append(_split_pipe_row(lines[j]))
                    j += 1
                # Auto-detect numeric columns: if every cell in a column (excluding empties)
                # parses as a number, right-align it regardless of separator hint.
                n_cols = len(header_cells)
                col_numeric = [True] * n_cols
                for r in rows:
                    for k in range(min(n_cols, len(r))):
                        if r[k].strip() and not _is_numeric(r[k]):
                            col_numeric[k] = False
                effective_aligns = [
                    "right" if col_numeric[k] and aligns[k] == "left" else aligns[k]
                    for k in range(n_cols)
                ]
                # Render
                out.append('<div class="table-wrap"><table>')
                out.append("  <thead><tr>")
                for k, h in enumerate(header_cells):
                    out.append(f'    <th style="text-align:{effective_aligns[k]}">{_render_inline(h)}</th>')
                out.append("  </tr></thead>")
                out.append("  <tbody>")
                for r in rows:
                    out.append("    <tr>")
                    for k in range(n_cols):
                        cell = r[k] if k < len(r) else ""
                        out.append(
                            f'      <td style="text-align:{effective_aligns[k]}">'
                            f'{_render_inline(cell)}</td>'
                        )
                    out.append("    </tr>")
                out.append("  </tbody>")
                out.append("</table></div>")
                i = j
                continue

        # Blank line
        if not line.strip():
            i += 1
            continue

        # Paragraph (consume consecutive non-empty, non-special lines)
        para: list[str] = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].strip().startswith(("#", "|", "-"))
        ):
            para.append(lines[i])
            i += 1
        joined = " ".join(p.strip() for p in para)
        out.append(f"<p>{_render_inline(joined)}</p>")

    return title, "\n".join(out)


# ---------------------------------------------------------------------------
# Page wrapper
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #fbfbfd;
  --fg: #1d1d1f;
  --muted: #6e6e73;
  --line: #d2d2d7;
  --row-alt: #f5f5f7;
  --accent: #0071e3;
  --code-bg: #f0f0f3;
}
* { box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  margin: 0;
  padding: 32px 40px 64px;
  max-width: 1400px;
}
h1 { font-size: 28px; margin: 8px 0 24px; font-weight: 600; }
h2 { font-size: 20px; margin: 32px 0 12px; font-weight: 600; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { font-size: 16px; margin: 24px 0 8px; font-weight: 600; color: var(--muted); }
p { margin: 8px 0 16px; color: #3a3a3c; }
hr { border: 0; border-top: 1px solid var(--line); margin: 32px 0; }
code {
  background: var(--code-bg);
  padding: 1px 6px;
  border-radius: 4px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12.5px;
  color: #3a3a3c;
}
strong { color: var(--fg); }

.table-wrap {
  overflow-x: auto;
  margin: 8px 0 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13.5px;
}
thead th {
  background: #ececef;
  color: var(--fg);
  font-weight: 600;
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
  position: sticky;
  top: 0;
  white-space: nowrap;
}
tbody td {
  padding: 8px 14px;
  border-bottom: 1px solid var(--row-alt);
  vertical-align: top;
}
tbody tr:nth-child(even) td { background: var(--row-alt); }
tbody tr:hover td { background: #e8f2fd; }
tbody tr:last-child td { border-bottom: 0; }

td code, th code { background: transparent; padding: 0; font-size: 12.5px; }

.footer {
  color: var(--muted);
  font-size: 12px;
  margin-top: 40px;
}
"""


def wrap_page(title: str, body: str, source_path: Path | None = None) -> str:
    page_title = title or "Results"
    src_note = (
        f'<div class="footer">Rendered from <code>{html.escape(str(source_path))}</code></div>'
        if source_path else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(page_title)}</title>
  <style>{CSS}</style>
</head>
<body>
{body}
{src_note}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render_one(src: Path, out_dir: Path | None = None) -> Path:
    md_text = src.read_text(encoding="utf-8")
    title, body = md_to_html(md_text)
    if out_dir is None:
        dest = src.with_suffix(".html")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / (src.stem + ".html")
    dest.write_text(wrap_page(title, body, source_path=src), encoding="utf-8")
    return dest


def discover_md_files() -> list[Path]:
    found: list[Path] = []
    for d in DEFAULT_DIRS:
        if d.is_dir():
            found.extend(sorted(d.glob("*.md")))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Specific .md files to render. Default: scan both summary folders.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory. Default: write each .html next to its source .md.")
    args = parser.parse_args()

    if args.paths:
        sources = [Path(p) for p in args.paths]
        missing = [p for p in sources if not p.exists()]
        if missing:
            print("Missing files: " + ", ".join(str(p) for p in missing))
            sys.exit(1)
    else:
        sources = discover_md_files()
        if not sources:
            print(f"No .md files found in {DEFAULT_DIRS}.")
            print("Run `summarize_phase2_cap_sweep.py` and/or `summarize_neuronpedia_validation.py` first.")
            sys.exit(1)

    for src in sources:
        dest = render_one(src, out_dir=args.out)
        print(f"  {src}  ->  {dest}")
    print(f"Rendered {len(sources)} file(s).")


if __name__ == "__main__":
    main()