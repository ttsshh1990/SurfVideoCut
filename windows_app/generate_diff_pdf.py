#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional dependency
    sync_playwright = None

try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.annotations import Popup, Text
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None
    PdfWriter = None
    Popup = None
    Text = None


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
BLOCK_TAGS = {
    "p",
    "li",
    "td",
    "th",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "caption",
    "blockquote",
    "pre",
    "figcaption",
}
SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "head",
    "meta",
    "link",
    "iframe",
    "object",
}
INLINE_BOLD_TAGS = {"b", "strong"}
INLINE_ITALIC_TAGS = {"i", "em"}
INLINE_UNDERLINE_TAGS = {"u"}


@dataclass
class Block:
    id: str
    source: str
    order: int
    text: str
    normalized: str
    heading: bool = False
    heading_level: int | None = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    list_item: bool = False
    table_cell: bool = False
    kind: str = ""
    table_pos: tuple[int, int, int] | None = None


@dataclass
class Match:
    docx_index: int
    html_index: int
    match_type: str
    score: float
    formatting_diffs: list[str]


@dataclass
class BrowserRenderResult:
    blocks: list[Block]
    width_px: float
    height_px: float
    rects_by_order: dict[int, tuple[float, float, float, float]]


def table_pos_key(block: Block) -> tuple[int, int, int] | None:
    return block.table_pos if block.table_cell and block.table_pos is not None else None


def table_index_key(block: Block) -> int | None:
    return block.table_pos[0] if block.table_cell and block.table_pos is not None else None


def split_lead_label_text(text: str, *, table_cell: bool = False) -> tuple[str, str] | None:
    if table_cell:
        return None
    lines = [part.strip() for part in str(text or "").splitlines() if part.strip()]
    if len(lines) < 2:
        return None
    lead = lines[0]
    rest = " ".join(lines[1:]).strip()
    if not rest:
        return None
    lead_words = len([word for word in lead.split() if word])
    looks_like_label = len(lead) <= 64 and lead_words <= 8 and not re.search(r"[.!?:;]$", lead)
    body_substantial = len(rest) >= 80
    if looks_like_label and body_substantial:
        return lead, rest
    return None


def normalize_text(text: str) -> str:
    return (
        str(text or "")
        .replace("\u00A0", " ")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201C", '"')
        .replace("\u201D", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2022", "-")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("$ ", "$")
        .replace("€ ", "€")
        .replace("£ ", "£")
        .replace("¥ ", "¥")
        .replace("~ ", "~")
    )


def normalize_for_compare(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\bwww\.", "", text, flags=re.I)
    text = re.sub(r"\(\s+(?=\d)", "(", text)
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    text = re.sub(r"(\d)\s+\)", r"\1)", text)
    text = re.sub(r"([+-])\s+(?=\d)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def visible_meaningful(text: str) -> bool:
    normalized = normalize_for_compare(text)
    return bool(normalized and re.search(r"[a-z0-9]", normalized))


def tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^0-9a-z]+", " ", normalize_for_compare(text))
    return [token for token in cleaned.split() if token]


def bigrams(text: str) -> list[str]:
    cleaned = normalize_for_compare(text).replace(" ", "")
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


def jaccard_tokens(a: str, b: str) -> float:
    set_a = set(tokenize(a))
    set_b = set(tokenize(b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def dice_bigrams(a: str, b: str) -> float:
    grams_a = bigrams(a)
    grams_b = bigrams(b)
    if not grams_a and not grams_b:
        return 1.0
    if not grams_a or not grams_b:
        return 0.0
    counts: dict[str, int] = {}
    for gram in grams_a:
        counts[gram] = counts.get(gram, 0) + 1
    inter = 0
    for gram in grams_b:
        count = counts.get(gram, 0)
        if count:
            inter += 1
            counts[gram] = count - 1
    return (2 * inter) / (len(grams_a) + len(grams_b))


def similarity(a: str, b: str) -> float:
    len_ratio = min(len(a), len(b)) / max(len(a), len(b), 1)
    return (0.5 * jaccard_tokens(a, b)) + (0.35 * dice_bigrams(a, b)) + (0.15 * len_ratio)


def summarize_formatting_diff(docx: Block, html: Block) -> list[str]:
    diffs: list[str] = []
    checks = [
        ("heading", docx.heading, html.heading),
        ("bold", docx.bold, html.bold),
        ("italic", docx.italic, html.italic),
        ("underline", docx.underline, html.underline),
        ("list item", docx.list_item, html.list_item),
        ("table cell", docx.table_cell, html.table_cell),
    ]
    for label, doc_value, html_value in checks:
        if doc_value != html_value:
            diffs.append(
                f"DOCX {'has' if doc_value else 'does not have'} {label}; "
                f"HTML {'has' if html_value else 'does not have'} it."
            )
    if docx.heading and html.heading and docx.heading_level != html.heading_level:
        diffs.append(
            f"DOCX heading level is {docx.heading_level or '?'}; "
            f"HTML heading level is {html.heading_level or '?'}."
        )
    return diffs


def xml_local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def style_flag(style_text: str, needle: str) -> bool:
    return needle in style_text.lower()


def collect_word_text(paragraph: ET.Element) -> tuple[str, bool, bool, bool]:
    text_parts: list[str] = []
    bold = False
    italic = False
    underline = False
    for run in paragraph.findall(".//w:r", WORD_NS):
        for t in run.findall("w:t", WORD_NS):
            text_parts.append(t.text or "")
        for tab in run.findall("w:tab", WORD_NS):
            text_parts.append("\t")
        for br in run.findall("w:br", WORD_NS):
            text_parts.append("\n")
        run_props = run.find("w:rPr", WORD_NS)
        if run_props is None:
            continue
        if run_props.find("w:b", WORD_NS) is not None:
            bold = True
        if run_props.find("w:i", WORD_NS) is not None:
            italic = True
        if run_props.find("w:u", WORD_NS) is not None:
            underline = True
    return "".join(text_parts), bold, italic, underline


def extract_docx_blocks(path: Path) -> list[Block]:
    with zipfile.ZipFile(path) as archive:
        xml_text = archive.read("word/document.xml")
    root = ET.fromstring(xml_text)
    body = root.find("w:body", WORD_NS)
    if body is None:
        return []

    blocks: list[Block] = []
    order = 0
    table_ordinal = 0
    for child in body:
        local = xml_local_name(child.tag)
        if local == "p":
            para_props = child.find("w:pPr", WORD_NS)
            style_val = ""
            list_item = False
            if para_props is not None:
                style = para_props.find("w:pStyle", WORD_NS)
                if style is not None:
                    style_val = (
                        style.get(f"{{{WORD_NS['w']}}}val", "")
                        or style.get("w:val", "")
                        or style.get("val", "")
                    )
                list_item = para_props.find("w:numPr", WORD_NS) is not None
            text, bold, italic, underline = collect_word_text(child)
            if not visible_meaningful(text):
                continue
            heading_match = re.search(r"heading\s*([1-6])?", style_val, flags=re.I)
            is_title = re.search(r"title", style_val, flags=re.I)
            heading = bool(heading_match or is_title)
            heading_level = int(heading_match.group(1)) if heading_match and heading_match.group(1) else 1 if is_title else None
            blocks.append(
                Block(
                    id=f"docx-{order}",
                    source="docx",
                    order=order,
                    text=normalize_text(text).strip(),
                    normalized=normalize_for_compare(text),
                    heading=heading,
                    heading_level=heading_level,
                    bold=bold,
                    italic=italic,
                    underline=underline,
                    list_item=list_item,
                    kind="p",
                )
            )
            order += 1
        elif local == "tbl":
            for row in child.findall(".//w:tr", WORD_NS):
                row_index = list(child.findall(".//w:tr", WORD_NS)).index(row)
                for col_index, cell in enumerate(row.findall("w:tc", WORD_NS)):
                    cell_parts: list[str] = []
                    bold = False
                    italic = False
                    underline = False
                    for paragraph in cell.findall("w:p", WORD_NS):
                        text, p_bold, p_italic, p_underline = collect_word_text(paragraph)
                        if visible_meaningful(text):
                            cell_parts.append(normalize_text(text).strip())
                        bold = bold or p_bold
                        italic = italic or p_italic
                        underline = underline or p_underline
                    cell_text = "\n".join(part for part in cell_parts if part)
                    if not visible_meaningful(cell_text):
                        continue
                    blocks.append(
                        Block(
                            id=f"docx-{order}",
                            source="docx",
                            order=order,
                            text=cell_text,
                            normalized=normalize_for_compare(cell_text),
                            bold=bold,
                            italic=italic,
                            underline=underline,
                            table_cell=True,
                            kind="td",
                            table_pos=(table_ordinal, row_index, col_index),
                        )
                    )
                    order += 1
            table_ordinal += 1
    return blocks


def get_descendant_text(element: ET.Element) -> str:
    parts: list[str] = []
    for text in element.itertext():
        parts.append(text)
    return normalize_text("".join(parts))


def node_has_descendant_block(element: ET.Element) -> bool:
    for child in list(element):
        if not isinstance(child.tag, str):
            continue
        if xml_local_name(child.tag).lower() in BLOCK_TAGS:
            return True
        if node_has_descendant_block(child):
            return True
    return False


def detect_inline_flags(element: ET.Element) -> tuple[bool, bool, bool]:
    bold = False
    italic = False
    underline = False
    for node in element.iter():
        if not isinstance(node.tag, str):
            continue
        tag = xml_local_name(node.tag).lower()
        style = (node.get("style") or "").lower()
        if tag in INLINE_BOLD_TAGS or style_flag(style, "font-weight: bold"):
            bold = True
        if tag in INLINE_ITALIC_TAGS or style_flag(style, "font-style: italic"):
            italic = True
        if tag in INLINE_UNDERLINE_TAGS or style_flag(style, "text-decoration: underline"):
            underline = True
    return bold, italic, underline


def is_hidden(element: ET.Element) -> bool:
    style = (element.get("style") or "").lower()
    hidden_attr = (element.get("hidden") or "").lower()
    return "display:none" in style or "visibility:hidden" in style or hidden_attr in {"hidden", "true"}


def annotate_html_parents(element: ET.Element, parent: ET.Element | None = None) -> None:
    setattr(element, "_parent", parent)
    for child in list(element):
        annotate_html_parents(child, element)


def nearest_html_ancestor(element: ET.Element, tag_name: str) -> ET.Element | None:
    current = getattr(element, "_parent", None)
    while current is not None:
        if isinstance(current.tag, str) and xml_local_name(current.tag).lower() == tag_name:
            return current
        current = getattr(current, "_parent", None)
    return None


def html_table_position(element: ET.Element, root: ET.Element) -> tuple[int, int, int] | None:
    tag = xml_local_name(element.tag).lower() if isinstance(element.tag, str) else ""
    if tag not in {"td", "th"}:
        return None
    tr = nearest_html_ancestor(element, "tr")
    table = nearest_html_ancestor(element, "table")
    if tr is None or table is None:
        return None
    tables = [node for node in root.iter() if isinstance(node.tag, str) and xml_local_name(node.tag).lower() == "table"]
    rows = [node for node in table.iter() if isinstance(node.tag, str) and xml_local_name(node.tag).lower() == "tr"]
    cells = [node for node in list(tr) if isinstance(node.tag, str) and xml_local_name(node.tag).lower() in {"td", "th"}]
    return (
        max(0, tables.index(table)),
        max(0, rows.index(tr)),
        max(0, cells.index(element)),
    )


def walk_html_blocks(element: ET.Element, blocks: list[Block], order_ref: list[int], hidden_ancestor: bool = False) -> None:
    if not isinstance(element.tag, str):
        return
    tag = xml_local_name(element.tag).lower()
    if tag in SKIP_TAGS:
        return

    hidden_here = hidden_ancestor or is_hidden(element)
    if hidden_here:
        return

    if tag in BLOCK_TAGS:
        capture_self = tag in {"td", "th", "li", "pre", "blockquote", "caption", "figcaption"}
        if not capture_self:
            capture_self = not node_has_descendant_block(element)
        if capture_self:
            text = get_descendant_text(element).strip()
            if visible_meaningful(text):
                bold, italic, underline = detect_inline_flags(element)
                heading_level = int(tag[1]) if re.fullmatch(r"h[1-6]", tag) else None
                table_cell = tag in {"td", "th"}
                split = split_lead_label_text(text, table_cell=table_cell)
                base_kwargs: dict[str, Any] = {
                    "source": "html",
                    "heading": heading_level is not None,
                    "heading_level": heading_level,
                    "bold": bold,
                    "italic": italic,
                    "underline": underline,
                    "list_item": tag == "li",
                    "table_cell": table_cell,
                    "kind": tag,
                    "table_pos": html_table_position(element, getattr(element, "_root", element)),
                }
                if split:
                    lead, rest = split
                    order = order_ref[0]
                    blocks.append(
                        Block(
                            id=f"html-{order}",
                            order=order,
                            text=lead,
                            normalized=normalize_for_compare(lead),
                            **base_kwargs,
                        )
                    )
                    order_ref[0] += 1
                    order = order_ref[0]
                    blocks.append(
                        Block(
                            id=f"html-{order}",
                            order=order,
                            text=rest,
                            normalized=normalize_for_compare(rest),
                            **{
                                **base_kwargs,
                                "heading": False,
                                "heading_level": None,
                                "bold": False,
                                "italic": False,
                                "underline": False,
                            },
                        )
                    )
                    order_ref[0] += 1
                else:
                    order = order_ref[0]
                    blocks.append(
                        Block(
                            id=f"html-{order}",
                            order=order,
                            text=text,
                            normalized=normalize_for_compare(text),
                            **base_kwargs,
                        )
                    )
                    order_ref[0] += 1
            return

    for child in list(element):
        walk_html_blocks(child, blocks, order_ref, hidden_here)


def extract_html_blocks(path: Path) -> list[Block]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    annotate_html_parents(root)
    for node in root.iter():
        setattr(node, "_root", root)
    body = None
    if xml_local_name(root.tag).lower() == "html":
        for child in root:
            if isinstance(child.tag, str) and xml_local_name(child.tag).lower() == "body":
                body = child
                break
    if body is None:
        body = root
    blocks: list[Block] = []
    walk_html_blocks(body, blocks, [0])
    return blocks


def browser_render_and_extract(html_path: Path, output_pdf: Path) -> BrowserRenderResult:
    if sync_playwright is None:
        raise RuntimeError(
            "Playwright is not installed. Install it with `pip install playwright` and `playwright install chromium`."
        )

    js = """
() => {
  const BLOCK_TAGS = new Set(['P','LI','TD','TH','H1','H2','H3','H4','H5','H6','CAPTION','BLOCKQUOTE','PRE','FIGCAPTION']);
  const SKIP_TAGS = new Set(['SCRIPT','STYLE','NOSCRIPT','TEMPLATE','SVG','CANVAS','HEAD','META','LINK','IFRAME','OBJECT']);
  const boldTags = new Set(['B','STRONG']);
  const italicTags = new Set(['I','EM']);
  const underlineTags = new Set(['U']);

  const normalizeText = (text) => String(text || '')
    .replace(/\\u00A0/g, ' ')
    .replace(/\\bwww\\./gi, '')
    .replace(/[ \\t\\r\\f\\v]+/g, ' ')
    .trim();

  const splitLeadLabelText = (text, tableCell = false) => {
    if (tableCell) return null;
    const lines = String(text || '').split(/\\n+/).map(s => s.trim()).filter(Boolean);
    if (lines.length < 2) return null;
    const lead = lines[0];
    const rest = lines.slice(1).join(' ').trim();
    if (!rest) return null;
    const leadWords = lead.split(/\\s+/).filter(Boolean).length;
    const looksLikeLabel = lead.length <= 64 && leadWords <= 8 && !/[.!?:;]$/.test(lead);
    const bodySubstantial = rest.length >= 80;
    return looksLikeLabel && bodySubstantial ? { lead, rest } : null;
  };

  const visibleMeaningful = (text) => /[A-Za-z0-9]/.test(normalizeText(text));

  function isHidden(el) {
    const style = window.getComputedStyle(el);
    return style.display === 'none' || style.visibility === 'hidden';
  }

  function hasDescendantBlock(el) {
    for (const child of el.children) {
      if (BLOCK_TAGS.has(child.tagName)) return true;
      if (hasDescendantBlock(child)) return true;
    }
    return false;
  }

  function detectInlineFlags(el) {
    let bold = false;
    let italic = false;
    let underline = false;
    for (const node of el.querySelectorAll('*')) {
      const style = window.getComputedStyle(node);
      if (boldTags.has(node.tagName) || parseInt(style.fontWeight || '400', 10) >= 600) bold = true;
      if (italicTags.has(node.tagName) || style.fontStyle === 'italic') italic = true;
      if (underlineTags.has(node.tagName) || (style.textDecorationLine || '').includes('underline')) underline = true;
    }
    const selfStyle = window.getComputedStyle(el);
    if (boldTags.has(el.tagName) || parseInt(selfStyle.fontWeight || '400', 10) >= 600) bold = true;
    if (italicTags.has(el.tagName) || selfStyle.fontStyle === 'italic') italic = true;
    if (underlineTags.has(el.tagName) || (selfStyle.textDecorationLine || '').includes('underline')) underline = true;
    return { bold, italic, underline };
  }

  const blocks = [];
  let order = 0;

  function walk(el, hiddenAncestor = false) {
    if (!(el instanceof HTMLElement)) return;
    if (SKIP_TAGS.has(el.tagName)) return;
    const hidden = hiddenAncestor || isHidden(el);
    if (hidden) return;

    if (BLOCK_TAGS.has(el.tagName)) {
      let captureSelf = ['TD','TH','LI','PRE','BLOCKQUOTE','CAPTION','FIGCAPTION'].includes(el.tagName);
      if (!captureSelf) captureSelf = !hasDescendantBlock(el);
      if (captureSelf) {
        const text = normalizeText(el.innerText || el.textContent || '');
        if (visibleMeaningful(text)) {
          const rect = el.getBoundingClientRect();
          const flags = detectInlineFlags(el);
          const headingMatch = /^H([1-6])$/.exec(el.tagName);
          const tableEl = ['TD', 'TH'].includes(el.tagName) ? el.closest('table') : null;
          const trEl = ['TD', 'TH'].includes(el.tagName) ? el.closest('tr') : null;
          const tableRows = tableEl ? Array.from(tableEl.querySelectorAll('tr')) : [];
          const cellSiblings = trEl ? Array.from(trEl.children).filter(child => child.tagName === 'TD' || child.tagName === 'TH') : [];
          const block = {
            id: `html-${order}`,
            source: 'html',
            order,
            text,
            normalized: '',
            heading: Boolean(headingMatch),
            heading_level: headingMatch ? Number(headingMatch[1]) : null,
            bold: flags.bold,
            italic: flags.italic,
            underline: flags.underline,
            list_item: el.tagName === 'LI',
            table_cell: el.tagName === 'TD' || el.tagName === 'TH',
            kind: el.tagName.toLowerCase(),
            table_pos: tableEl && trEl ? {
              table: Math.max(0, Array.from(document.querySelectorAll('table')).indexOf(tableEl)),
              row: Math.max(0, tableRows.indexOf(trEl)),
              col: Math.max(0, cellSiblings.indexOf(el))
            } : null,
            x: rect.left + window.scrollX,
            y: rect.top + window.scrollY,
            width: rect.width,
            height: rect.height
          };
          const split = splitLeadLabelText(text, block.table_cell);
          if (split) {
            el.setAttribute('data-docx-compare-order', String(order));
            blocks.push({
              ...block,
              id: `html-${order}`,
              order,
              text: split.lead,
            });
            order += 1;
            blocks.push({
              ...block,
              id: `html-${order}`,
              order,
              text: split.rest,
              heading: false,
              heading_level: null,
              bold: false,
              italic: false,
              underline: false,
            });
            order += 1;
          } else {
            el.setAttribute('data-docx-compare-order', String(order));
            blocks.push(block);
            order += 1;
          }
          return;
        }
      }
    }
    for (const child of el.children) walk(child, hidden);
  }

  document.documentElement.style.setProperty('-webkit-print-color-adjust', 'exact');
  document.documentElement.style.setProperty('print-color-adjust', 'exact');
  walk(document.body || document.documentElement);
  return {
    blocks,
    width: Math.max(
      document.documentElement.scrollWidth,
      document.body ? document.body.scrollWidth : 0
    ),
    height: Math.max(
      document.documentElement.scrollHeight,
      document.body ? document.body.scrollHeight : 0
    ),
  };
}
"""

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        page.wait_for_load_state("networkidle")
        page.emulate_media(media="screen")
        result = page.evaluate(js)
        width_px = max(float(result["width"]), 1.0)
        height_px = max(float(result["height"]), 1.0)
        page.pdf(
            path=str(output_pdf),
            width=f"{math.ceil(width_px)}px",
            height=f"{math.ceil(height_px)}px",
            print_background=True,
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"},
            prefer_css_page_size=False,
        )
        browser.close()

    blocks: list[Block] = []
    rects_by_order: dict[int, tuple[float, float, float, float]] = {}
    for item in result["blocks"]:
        order = int(item["order"])
        rects_by_order[order] = (
            float(item["x"]),
            float(item["y"]),
            float(item["width"]),
            float(item["height"]),
        )
        blocks.append(
            Block(
                id=item["id"],
                source="html",
                order=order,
                text=str(item["text"]),
                normalized=normalize_for_compare(str(item["text"])),
                heading=bool(item["heading"]),
                heading_level=int(item["heading_level"]) if item["heading_level"] is not None else None,
                bold=bool(item["bold"]),
                italic=bool(item["italic"]),
                underline=bool(item["underline"]),
                list_item=bool(item["list_item"]),
                table_cell=bool(item["table_cell"]),
                kind=str(item["kind"]),
                table_pos=(
                    int(item["table_pos"]["table"]),
                    int(item["table_pos"]["row"]),
                    int(item["table_pos"]["col"]),
                ) if item.get("table_pos") else None,
            )
        )
    return BrowserRenderResult(
        blocks=blocks,
        width_px=width_px,
        height_px=height_px,
        rects_by_order=rects_by_order,
    )


def build_html_indices(
    html_blocks: list[Block],
) -> tuple[
    dict[str, list[int]],
    dict[str, list[int]],
    dict[int, list[int]],
    dict[tuple[int, int, int], list[int]],
    dict[int, list[int]],
]:
    exact_map: dict[str, list[int]] = {}
    token_index: dict[str, list[int]] = {}
    length_buckets: dict[int, list[int]] = {}
    table_pos_map: dict[tuple[int, int, int], list[int]] = {}
    table_index_map: dict[int, list[int]] = {}
    for index, block in enumerate(html_blocks):
        exact_map.setdefault(block.normalized, []).append(index)
        pos_key = table_pos_key(block)
        if pos_key is not None:
            table_pos_map.setdefault(pos_key, []).append(index)
        table_idx = table_index_key(block)
        if table_idx is not None:
            table_index_map.setdefault(table_idx, []).append(index)
        for token in dict.fromkeys(token for token in tokenize(block.normalized) if len(token) >= 3):
            token_index.setdefault(token, []).append(index)
        bucket = max(1, round(len(block.normalized) / 30))
        length_buckets.setdefault(bucket, []).append(index)
    return exact_map, token_index, length_buckets, table_pos_map, table_index_map


def get_approx_candidates(
    doc_block: Block,
    html_blocks: list[Block],
    token_index: dict[str, list[int]],
    length_buckets: dict[int, list[int]],
    table_pos_map: dict[tuple[int, int, int], list[int]],
    table_index_map: dict[int, list[int]],
    used_html: set[int],
) -> list[int]:
    pos_key = table_pos_key(doc_block)
    if pos_key is not None and pos_key in table_pos_map:
        return [index for index in table_pos_map[pos_key] if index not in used_html]
    table_idx = table_index_key(doc_block)
    if table_idx is not None and table_idx in table_index_map:
        return [index for index in table_index_map[table_idx] if index not in used_html]
    candidates: set[int] = set()
    tokens = list(dict.fromkeys(token for token in tokenize(doc_block.normalized) if len(token) >= 3))
    for token in tokens[:3]:
        for index in token_index.get(token, []):
            if len(candidates) >= 120:
                break
            candidates.add(index)
    bucket = max(1, round(len(doc_block.normalized) / 30))
    for probe in range(bucket - 1, bucket + 2):
        for index in length_buckets.get(probe, []):
            if len(candidates) >= 200:
                break
            candidates.add(index)

    filtered: list[int] = []
    doc_len = max(len(doc_block.normalized), 1)
    for index in candidates:
        if index in used_html:
            continue
        html_len = max(len(html_blocks[index].normalized), 1)
        ratio = min(doc_len, html_len) / max(doc_len, html_len)
        if ratio >= 0.45:
            filtered.append(index)
    return filtered


def compare_blocks(docx_blocks: list[Block], html_blocks: list[Block]) -> tuple[list[Match], list[Block], list[Block]]:
    exact_map, token_index, length_buckets, table_pos_map, table_index_map = build_html_indices(html_blocks)
    used_html: set[int] = set()
    matches: list[Match] = []
    unmatched_docx: list[Block] = []

    for doc_index, doc_block in enumerate(docx_blocks):
        pos_key = table_pos_key(doc_block)
        table_idx = table_index_key(doc_block)
        if pos_key is not None:
            exact_pos_candidates = table_pos_map.get(pos_key, [])
            exact_pos_index = next(
                (index for index in exact_pos_candidates if index not in used_html and html_blocks[index].normalized == doc_block.normalized),
                None,
            )
            if exact_pos_index is not None:
                used_html.add(exact_pos_index)
                matches.append(
                    Match(
                        docx_index=doc_index,
                        html_index=exact_pos_index,
                        match_type="exact",
                        score=1.0,
                        formatting_diffs=summarize_formatting_diff(doc_block, html_blocks[exact_pos_index]),
                    )
                )
                continue
        if table_idx is not None:
            exact_table_candidates = table_index_map.get(table_idx, [])
            exact_table_index = next(
                (index for index in exact_table_candidates if index not in used_html and html_blocks[index].normalized == doc_block.normalized),
                None,
            )
            if exact_table_index is not None:
                used_html.add(exact_table_index)
                matches.append(
                    Match(
                        docx_index=doc_index,
                        html_index=exact_table_index,
                        match_type="exact",
                        score=1.0,
                        formatting_diffs=summarize_formatting_diff(doc_block, html_blocks[exact_table_index]),
                    )
                )
                continue
        exact_candidates = [] if table_idx is not None else exact_map.get(doc_block.normalized, [])
        exact_index = next((index for index in exact_candidates if index not in used_html), None)
        if exact_index is not None:
            used_html.add(exact_index)
            matches.append(
                Match(
                    docx_index=doc_index,
                    html_index=exact_index,
                    match_type="exact",
                    score=1.0,
                    formatting_diffs=summarize_formatting_diff(doc_block, html_blocks[exact_index]),
                )
            )
            continue

        best_index = None
        best_score = 0.0
        for html_index in get_approx_candidates(doc_block, html_blocks, token_index, length_buckets, table_pos_map, table_index_map, used_html):
            score = similarity(doc_block.normalized, html_blocks[html_index].normalized)
            if score > best_score:
                best_score = score
                best_index = html_index
        if best_index is not None and best_score >= 0.73:
            used_html.add(best_index)
            matches.append(
                Match(
                    docx_index=doc_index,
                    html_index=best_index,
                    match_type="approx",
                    score=best_score,
                    formatting_diffs=summarize_formatting_diff(doc_block, html_blocks[best_index]),
                )
            )
        else:
            unmatched_docx.append(doc_block)

    unmatched_html = [block for index, block in enumerate(html_blocks) if index not in used_html]
    matches.sort(key=lambda match: match.html_index)
    return matches, unmatched_docx, unmatched_html


def shorten(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", normalize_text(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def prnewswire_only_difference(doc_text: str, html_text: str) -> bool:
    if "/PRNewswire/" not in html_text or "/PRNewswire/" in doc_text:
        return False
    html_without_wire = normalize_for_compare(html_text.replace("/PRNewswire/", ""))
    doc_normalized = normalize_for_compare(doc_text)
    html_without_wire = html_without_wire.replace(", --", " -").replace("--", "-")
    doc_normalized = doc_normalized.replace(" - ", " -")
    return (
        "synopsys, inc." in doc_normalized
        and "synopsys, inc." in html_without_wire
        and similarity(doc_normalized, html_without_wire) >= 0.985
    )


def text_difference_comment(doc_block: Block, html_block: Block, score: float) -> str | None:
    if doc_block.normalized == html_block.normalized:
        return None
    if prnewswire_only_difference(doc_block.text, html_block.text):
        return 'Text differs from DOCX match. HTML has “/PRNewswire/“ while DOCX does not.'
    if score >= 0.999:
        return None
    return (
        "Text differs from the DOCX match. "
        f"Similarity score: {score:.2f}. "
        f"DOCX text: {shorten(doc_block.text)}"
    )


def appendix_summary_blocks(docx_blocks: list[Block], unmatched_docx: list[Block]) -> list[Block]:
    summary_blocks: list[Block] = []
    seen_ids: set[str] = set()

    def add(block: Block) -> None:
        if block.id in seen_ids:
            return
        seen_ids.add(block.id)
        summary_blocks.append(block)

    for block in unmatched_docx:
        if normalize_for_compare(block.text) == "synopsys, inc.":
            continue
        add(block)

    if len(summary_blocks) >= 28:
        return summary_blocks

    label_texts = {
        "three months ended",
        "january 31,",
        "adjustments:",
        "amortization of acquired intangible assets",
        "stock-based compensation",
    }
    for block in docx_blocks:
        normalized = block.normalized
        if not block.table_cell or block.table_pos is None:
            continue
        table_idx, row_idx, _col_idx = block.table_pos
        if "gaap to non-gaap reconciliation" in normalized:
            add(block)
        if table_idx != 1 or row_idx < 19:
            continue
        numeric_summary_value = False
        if re.fullmatch(r"\$?\d\.\d{2}", block.text.strip()):
            if row_idx in {22, 25}:
                numeric_summary_value = True
            elif block.table_pos[2] == 1:
                numeric_summary_value = True
        if (
            "per diluted share attributed to synopsys" in normalized
            or normalized in label_texts
            or numeric_summary_value
        ):
            add(block)
        if len(summary_blocks) >= 28:
            break

    return summary_blocks


def build_comments(
    docx_blocks: list[Block],
    html_blocks: list[Block],
    matches: list[Match],
    unmatched_docx: list[Block],
    unmatched_html: list[Block],
) -> tuple[dict[int, list[str]], list[tuple[Block, str]]]:
    html_comments: dict[int, list[str]] = {}
    for match in matches:
        doc_block = docx_blocks[match.docx_index]
        html_block = html_blocks[match.html_index]
        comments: list[str] = []
        text_comment = text_difference_comment(doc_block, html_block, match.score)
        if text_comment:
            comments.append(text_comment)
        if comments:
            html_comments[match.html_index] = comments

    for block in unmatched_html:
        html_comments.setdefault(block.order, []).append(
            "This HTML block has no corresponding content in the DOCX."
        )

    appendix_comments = [
        (block, "This DOCX content was not found in the webpage HTML.")
        for block in appendix_summary_blocks(docx_blocks, unmatched_docx)
    ]
    return html_comments, appendix_comments


def pdf_safe_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    encoded = text.encode("latin-1", errors="replace")
    return encoded.decode("latin-1")


class PdfBuilder:
    def __init__(self, page_width: int = 612, page_height: int = 792) -> None:
        self.page_width = page_width
        self.page_height = page_height
        self.margin_left = 54
        self.margin_right = 54
        self.margin_top = 54
        self.margin_bottom = 54
        self.font_size = 11
        self.leading = 14
        self.title_size = 16
        self.subtitle_size = 10
        self.pages: list[dict[str, object]] = []
        self.current_page: dict[str, object] | None = None
        self.current_y = 0

    def new_page(self) -> None:
        page: dict[str, object] = {"ops": [], "annots": []}
        self.pages.append(page)
        self.current_page = page
        self.current_y = self.page_height - self.margin_top

    def ensure_page(self) -> None:
        if self.current_page is None:
            self.new_page()

    def available_height(self) -> int:
        return int(self.current_y - self.margin_bottom)

    def wrap_text(self, text: str, font_size: int | None = None, indent: int = 0) -> list[str]:
        size = font_size or self.font_size
        usable_width = self.page_width - self.margin_left - self.margin_right - indent
        chars_per_line = max(30, int(usable_width / (size * 0.54)))
        text = re.sub(r"[ \t]+", " ", normalize_text(text)).strip()
        if not text:
            return [""]
        wrapped: list[str] = []
        for paragraph in text.splitlines() or [""]:
            if not paragraph.strip():
                wrapped.append("")
                continue
            wrapped.extend(textwrap.wrap(paragraph, width=chars_per_line, break_long_words=False, break_on_hyphens=False))
        return wrapped or [""]

    def _append_text_op(self, x: int, y: int, text: str, font_size: int) -> None:
        assert self.current_page is not None
        op = f"BT /F1 {font_size} Tf 1 0 0 1 {x} {y} Tm ({pdf_safe_text(text)}) Tj ET"
        self.current_page["ops"].append(op)

    def add_wrapped_text(
        self,
        text: str,
        *,
        font_size: int | None = None,
        indent: int = 0,
        gap_after: int = 0,
    ) -> tuple[int, int]:
        self.ensure_page()
        size = font_size or self.font_size
        lines = self.wrap_text(text, font_size=size, indent=indent)
        required = len(lines) * self.leading + gap_after
        if self.available_height() < required:
            self.new_page()
        assert self.current_page is not None
        first_y = self.current_y
        x = self.margin_left + indent
        for line in lines:
            self._append_text_op(x, self.current_y, line, size)
            self.current_y -= self.leading
        self.current_y -= gap_after
        return x, first_y

    def add_annotation(self, x: int, y: int, contents: str) -> None:
        self.ensure_page()
        assert self.current_page is not None
        rect = [x, y - 4, x + 16, y + 12]
        self.current_page["annots"].append({"rect": rect, "contents": contents})

    def add_block(self, label: str, body: str, comment: str | None = None) -> None:
        header_x, header_y = self.add_wrapped_text(label, font_size=self.subtitle_size, gap_after=0)
        if comment:
            self.add_annotation(header_x - 20, header_y, comment)
        self.add_wrapped_text(body, indent=12, gap_after=8)

    def build(self) -> bytes:
        objects: list[bytes] = []

        def add_object(data: bytes) -> int:
            objects.append(data)
            return len(objects)

        font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

        page_ids: list[int] = []
        page_object_indices: list[int] = []
        content_ids: list[int] = []
        annot_ids_per_page: list[list[int]] = []

        for page in self.pages:
            stream_text = "\n".join(page["ops"]) + "\n"
            stream_bytes = stream_text.encode("latin-1", errors="replace")
            content_id = add_object(
                b"<< /Length "
                + str(len(stream_bytes)).encode("ascii")
                + b" >>\nstream\n"
                + stream_bytes
                + b"endstream"
            )
            content_ids.append(content_id)

            page_annot_ids: list[int] = []
            for annot in page["annots"]:
                rect = annot["rect"]
                contents = pdf_safe_text(str(annot["contents"]))
                date_text = datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%SZ")
                annot_id = add_object(
                    (
                        "<< /Type /Annot /Subtype /Text "
                        f"/Rect [{' '.join(f'{value:.2f}' for value in rect)}] "
                        f"/Contents ({contents}) "
                        "/Name /Comment "
                        "/T (DOCX Compare) "
                        f"/M ({date_text}) "
                        "/Open false "
                        "/C [1 0.94 0.35] >>"
                    ).encode("latin-1", errors="replace")
                )
                page_annot_ids.append(annot_id)
            annot_ids_per_page.append(page_annot_ids)

            page_ids.append(add_object(b""))
            page_object_indices.append(len(objects) - 1)

        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
        pages_id = add_object(
            f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("ascii")
        )
        catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))

        for page_id, obj_index, content_id, annot_ids in zip(page_ids, page_object_indices, content_ids, annot_ids_per_page):
            annots = ""
            if annot_ids:
                annots = " /Annots [" + " ".join(f"{annot_id} 0 R" for annot_id in annot_ids) + "]"
            page_obj = (
                f"<< /Type /Page /Parent {pages_id} 0 R "
                f"/MediaBox [0 0 {self.page_width} {self.page_height}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R{annots} >>"
            ).encode("ascii")
            objects[obj_index] = page_obj

        output = bytearray(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")
        offsets = [0]
        for number, obj in enumerate(objects, start=1):
            offsets.append(len(output))
            output.extend(f"{number} 0 obj\n".encode("ascii"))
            output.extend(obj)
            output.extend(b"\nendobj\n")
        xref_offset = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        return bytes(output)


def annotate_existing_pdf(
    pdf_path: Path,
    html_comments: dict[int, list[str]],
    appendix_comments: list[tuple[Block, str]],
    render_result: BrowserRenderResult,
) -> None:
    if PdfReader is None or PdfWriter is None or Text is None:
        raise RuntimeError(
            "pypdf is not installed. Install it with `pip install pypdf` to embed PDF comments."
        )

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    if not writer.pages:
        raise RuntimeError("The rendered PDF has no pages.")

    page = writer.pages[0]
    page_width_pt = float(page.mediabox.right) - float(page.mediabox.left)
    page_height_pt = float(page.mediabox.top) - float(page.mediabox.bottom)
    scale_x = page_width_pt / max(render_result.width_px, 1.0)
    scale_y = page_height_pt / max(render_result.height_px, 1.0)

    for order, comments in html_comments.items():
        rect = render_result.rects_by_order.get(order)
        if not rect:
            continue
        x_px, y_px, width_px, height_px = rect
        x_pt = x_px * scale_x
        y_top_pt = page_height_pt - (y_px * scale_y)
        note_w = 16
        note_h = 16
        note_rect = (
            max(6, x_pt - note_w - 2),
            max(6, y_top_pt - note_h),
            max(22, x_pt - 2),
            max(22, y_top_pt),
        )
        annotation = Text(
            text="\n\n".join(comments),
            rect=note_rect,
            open=False,
        )
        text_annotation = writer.add_annotation(page_number=0, annotation=annotation)
        popup_rect = (
            min(page_width_pt - 20, x_pt + max(width_px * scale_x, 180)),
            max(20, y_top_pt - 140),
            min(page_width_pt - 10, x_pt + max(width_px * scale_x, 360)),
            max(80, y_top_pt - 20),
        )
        writer.add_annotation(
            page_number=0,
            annotation=Popup(
                rect=popup_rect,
                open=False,
                parent=text_annotation,
            ),
        )

    if appendix_comments:
        lines = [f"- {shorten(block.text, 160)}" for block, _comment in appendix_comments[:20]]
        if len(appendix_comments) > 20:
            lines.append(f"- ... and {len(appendix_comments) - 20} more DOCX-only blocks")
        summary_text = (
            "DOCX content with no corresponding HTML block:\n"
            + "\n".join(lines)
        )
        writer.add_annotation(
            page_number=0,
            annotation=Text(
                text=summary_text,
                rect=(18, page_height_pt - 26, 34, page_height_pt - 10),
                open=False,
            ),
        )

    with pdf_path.open("wb") as handle:
        writer.write(handle)


def render_pdf(
    html_blocks: list[Block],
    html_comments: dict[int, list[str]],
    appendix_comments: list[tuple[Block, str]],
    output_path: Path,
    docx_path: Path,
    html_path: Path,
) -> None:
    pdf = PdfBuilder()
    pdf.add_wrapped_text("Webpage PDF With DOCX Difference Comments", font_size=16, gap_after=6)
    pdf.add_wrapped_text(
        f"HTML source: {html_path.name} | DOCX source: {docx_path.name}",
        font_size=10,
        gap_after=14,
    )
    pdf.add_wrapped_text(
        "Each yellow note icon is an embedded PDF comment that flags text, formatting, or presence differences versus the DOCX source.",
        font_size=10,
        gap_after=14,
    )

    for block in html_blocks:
        tag_bits = []
        if block.heading:
            tag_bits.append(f"heading{block.heading_level or ''}")
        if block.list_item:
            tag_bits.append("list")
        if block.table_cell:
            tag_bits.append("table")
        if block.bold:
            tag_bits.append("bold")
        if block.italic:
            tag_bits.append("italic")
        if block.underline:
            tag_bits.append("underline")
        meta = f"[HTML {block.order + 1}]"
        if tag_bits:
            meta += " " + ", ".join(tag_bits)
        comments = html_comments.get(block.order, [])
        pdf.add_block(meta, block.text, "\n\n".join(comments) if comments else None)

    if appendix_comments:
        pdf.new_page()
        pdf.add_wrapped_text("DOCX-Only Content Appendix", font_size=16, gap_after=8)
        pdf.add_wrapped_text(
            "These blocks were present in the DOCX but had no match in the webpage content. Each block below also carries an embedded PDF comment.",
            font_size=10,
            gap_after=14,
        )
        for index, (block, comment) in enumerate(appendix_comments, start=1):
            pdf.add_block(f"[DOCX only {index}]", block.text, comment)

    output_path.write_bytes(pdf.build())


def build_summary(
    docx_blocks: list[Block],
    html_blocks: list[Block],
    matches: list[Match],
    unmatched_docx: list[Block],
    unmatched_html: list[Block],
) -> dict[str, object]:
    exact = sum(1 for match in matches if match.match_type == "exact")
    approx = sum(1 for match in matches if match.match_type == "approx")
    return {
        "docx_blocks": len(docx_blocks),
        "html_blocks": len(html_blocks),
        "exact_matches": exact,
        "approx_matches": approx,
        "docx_only": len(unmatched_docx),
        "html_only": len(unmatched_html),
        "matches": [asdict(match) for match in matches],
        "docx_only_blocks": [asdict(block) for block in unmatched_docx],
        "html_only_blocks": [asdict(block) for block in unmatched_html],
    }


def default_output_name(html_path: Path) -> str:
    return f"{html_path.stem}__docx_diff_comments.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a saved HTML webpage into a PDF and embed comments for differences versus a DOCX file."
    )
    parser.add_argument(
        "--docx",
        type=Path,
        default=Path("2026-02-25 SNPS_Q1'26_EarningsRelease_Final.docx"),
        help="Source DOCX file.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("Q1'26 Earnings Release Proof_022426 4pm.html"),
        help="Source saved HTML webpage file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination PDF file. Defaults to <html-stem>__docx_diff_comments.pdf.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "playwright", "simple"),
        default="auto",
        help="PDF rendering mode. Use `playwright` for browser-faithful rendering on Windows/macOS/Linux.",
    )
    return parser.parse_args()


def run_compare(
    *,
    docx_path: Path,
    html_path: Path,
    output_path: Path | None = None,
    summary_json_path: Path | None = None,
    renderer: str = "auto",
) -> dict[str, object]:
    output_path = output_path or Path(default_output_name(html_path))
    docx_blocks = extract_docx_blocks(docx_path)
    use_playwright = renderer == "playwright" or (
        renderer == "auto" and sync_playwright is not None and PdfReader is not None
    )

    render_result: BrowserRenderResult | None = None
    if use_playwright:
        render_result = browser_render_and_extract(html_path, output_path)
        html_blocks = render_result.blocks
    else:
        html_blocks = extract_html_blocks(html_path)

    matches, unmatched_docx, unmatched_html = compare_blocks(docx_blocks, html_blocks)
    html_comments, appendix_comments = build_comments(
        docx_blocks,
        html_blocks,
        matches,
        unmatched_docx,
        unmatched_html,
    )

    if use_playwright:
        annotate_existing_pdf(
            pdf_path=output_path,
            html_comments=html_comments,
            appendix_comments=appendix_comments,
            render_result=render_result,
        )
    else:
        render_pdf(
            html_blocks=html_blocks,
            html_comments=html_comments,
            appendix_comments=appendix_comments,
            output_path=output_path,
            docx_path=docx_path,
            html_path=html_path,
        )

    summary = build_summary(docx_blocks, html_blocks, matches, unmatched_docx, unmatched_html)
    summary["renderer"] = "playwright" if use_playwright else "simple"
    summary["output_pdf"] = str(output_path)
    if summary_json_path:
        summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_json"] = str(summary_json_path)
    return summary


def main() -> int:
    args = parse_args()
    summary = run_compare(
        docx_path=args.docx,
        html_path=args.html,
        output_path=args.output,
        summary_json_path=args.summary_json,
        renderer=args.renderer,
    )

    print(f"PDF written: {summary['output_pdf']}")
    print(
        "Summary: "
        f"{summary['docx_blocks']} DOCX blocks, "
        f"{summary['html_blocks']} HTML blocks, "
        f"{summary['exact_matches']} exact matches, "
        f"{summary['approx_matches']} approximate matches, "
        f"{summary['docx_only']} DOCX-only, "
        f"{summary['html_only']} HTML-only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
