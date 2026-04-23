"""
Generate PDF documents for the RAG Customer Support Assistant project.
Converts markdown docs to styled PDFs with embedded diagram images.
Uses reportlab for PDF generation.
"""
import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, Preformatted, HRFlowable
)
from reportlab.lib.colors import black, white, grey, lightgrey

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
IMG_DIR = os.path.join(DOCS_DIR, "images")

# Colors
PRIMARY = HexColor("#667EEA")
SECONDARY = HexColor("#764BA2")
ACCENT = HexColor("#38B2AC")
DARK = HexColor("#1A202C")
LIGHT_BG = HexColor("#F7FAFC")
CODE_BG = HexColor("#EDF2F7")
BORDER = HexColor("#CBD5E0")


def get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('DocTitle', parent=styles['Title'],
        fontSize=24, textColor=PRIMARY, spaceAfter=6, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle('DocSubtitle', parent=styles['Normal'],
        fontSize=12, textColor=SECONDARY, alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle('H1', parent=styles['Heading1'],
        fontSize=18, textColor=PRIMARY, spaceBefore=20, spaceAfter=10,
        fontName='Helvetica-Bold', borderWidth=1, borderColor=PRIMARY, borderPadding=4))
    styles.add(ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=14, textColor=SECONDARY, spaceBefore=14, spaceAfter=8,
        fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle('H3', parent=styles['Heading3'],
        fontSize=11, textColor=DARK, spaceBefore=10, spaceAfter=6,
        fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=DARK, alignment=TA_JUSTIFY,
        spaceAfter=6, fontName='Helvetica'))
    styles.add(ParagraphStyle('BulletItem', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=DARK, leftIndent=20,
        bulletIndent=10, spaceAfter=3, fontName='Helvetica'))
    styles.add(ParagraphStyle('CodeBlock', parent=styles['Code'],
        fontSize=8, leading=10, backColor=CODE_BG, borderWidth=0.5,
        borderColor=BORDER, borderPadding=6, fontName='Courier',
        leftIndent=10, rightIndent=10, spaceAfter=8, spaceBefore=4))
    styles.add(ParagraphStyle('FooterStyle', parent=styles['Normal'],
        fontSize=8, textColor=grey, alignment=TA_CENTER))
    return styles


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(grey)
    canvas.drawCentredString(A4[0]/2, 15*mm, f"Page {doc.page}")
    canvas.restoreState()


def parse_markdown_to_elements(md_text, styles, images_map):
    """Parse markdown text into reportlab flowable elements."""
    elements = []
    lines = md_text.split('\n')
    i = 0
    in_code = False
    code_lines = []
    in_table = False
    table_rows = []

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith('```'):
            if in_code:
                code_text = '\n'.join(code_lines)
                if code_text.strip():
                    elements.append(Preformatted(code_text, styles['CodeBlock']))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Table rows
        if '|' in line and line.strip().startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if all(set(c) <= set('-: ') for c in cells):
                i += 1
                continue
            table_rows.append(cells)
            # Check if next line is still table
            if i + 1 < len(lines) and '|' in lines[i+1] and lines[i+1].strip().startswith('|'):
                i += 1
                continue
            # End of table - render it
            if table_rows:
                elements.append(build_table(table_rows, styles))
                table_rows = []
            i += 1
            continue

        # Headers
        if line.startswith('# ') and not line.startswith('##'):
            text = clean_md(line[2:].strip())
            elements.append(Paragraph(text, styles['H1']))
            # Check if there's a matching image
            for key, img_path in images_map.items():
                if key in text.lower():
                    elements.append(add_image(img_path))
            i += 1
            continue

        if line.startswith('## '):
            text = clean_md(line[3:].strip())
            elements.append(Spacer(1, 6))
            elements.append(HRFlowable(width="100%", thickness=1, color=PRIMARY))
            elements.append(Paragraph(text, styles['H1']))
            # Insert relevant diagram
            for key, img_path in images_map.items():
                if key in text.lower():
                    elements.append(add_image(img_path))
            i += 1
            continue

        if line.startswith('### '):
            text = clean_md(line[4:].strip())
            elements.append(Paragraph(text, styles['H2']))
            for key, img_path in images_map.items():
                if key in text.lower():
                    elements.append(add_image(img_path))
            i += 1
            continue

        if line.startswith('#### '):
            text = clean_md(line[5:].strip())
            elements.append(Paragraph(text, styles['H3']))
            i += 1
            continue

        # Horizontal rule
        if line.strip() == '---':
            elements.append(Spacer(1, 6))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
            elements.append(Spacer(1, 6))
            i += 1
            continue

        # Bullet points
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            text = clean_md(line.strip()[2:])
            elements.append(Paragraph(f"\u2022 {text}", styles['BulletItem']))
            i += 1
            continue

        # Numbered lists
        m = re.match(r'^\s*(\d+)\.\s+(.*)', line)
        if m:
            text = clean_md(m.group(2))
            elements.append(Paragraph(f"{m.group(1)}. {text}", styles['BulletItem']))
            i += 1
            continue

        # Empty line
        if not line.strip():
            elements.append(Spacer(1, 4))
            i += 1
            continue

        # Normal paragraph
        text = clean_md(line.strip())
        if text:
            elements.append(Paragraph(text, styles['Body']))
        i += 1

    return elements


def clean_md(text):
    """Convert markdown formatting to reportlab XML tags."""
    # First strip markdown links - keep just the text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Extract bold/italic/code segments and replace with placeholders
    segments = []
    
    def save_bold(m):
        idx = len(segments)
        segments.append(f'<b>{_esc(m.group(1))}</b>')
        return f'\x00PLACEHOLDER{idx}\x00'
    
    def save_italic(m):
        idx = len(segments)
        segments.append(f'<i>{_esc(m.group(1))}</i>')
        return f'\x00PLACEHOLDER{idx}\x00'
    
    def save_code(m):
        idx = len(segments)
        escaped = _esc(m.group(1))
        segments.append(f'<font face="Courier" size="9" color="#E53E3E">{escaped}</font>')
        return f'\x00PLACEHOLDER{idx}\x00'
    
    # Apply in order: code first (most specific), then bold, then italic
    text = re.sub(r'`(.+?)`', save_code, text)
    text = re.sub(r'\*\*(.+?)\*\*', save_bold, text)
    text = re.sub(r'\*(.+?)\*', save_italic, text)
    
    # Now escape the remaining text
    text = _esc(text)
    
    # Restore placeholders
    for i, seg in enumerate(segments):
        text = text.replace(f'\x00PLACEHOLDER{i}\x00', seg)
    
    return text


def _esc(text):
    """Escape XML special characters."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def build_table(rows, styles):
    """Build a styled reportlab table from parsed rows."""
    if not rows:
        return Spacer(1, 1)
    # Ensure all rows have same column count
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append('')

    # Convert cells to Paragraphs
    data = []
    for ri, row in enumerate(rows):
        data.append([Paragraph(clean_md(c), styles['Body']) for c in row])

    col_width = (A4[0] - 2*inch) / max_cols
    t = Table(data, colWidths=[col_width]*max_cols)

    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


def add_image(img_path, max_width=6*inch):
    """Add an image flowable, scaled to fit."""
    if not os.path.exists(img_path):
        return Spacer(1, 1)
    img = Image(img_path)
    ratio = img.imageWidth / img.imageHeight
    w = min(max_width, img.imageWidth * 0.5)
    h = w / ratio
    if h > 4*inch:
        h = 4*inch
        w = h * ratio
    img.drawWidth = w
    img.drawHeight = h
    return img


def build_title_page(title, subtitle, styles):
    """Create a title page."""
    elements = []
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph(title, styles['DocTitle']))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(subtitle, styles['DocSubtitle']))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(HRFlowable(width="60%", thickness=2, color=PRIMARY))
    elements.append(Spacer(1, 0.3*inch))
    meta = [
        "Version: 1.0",
        "Date: April 2026",
        "Author: Moin Uddin",
        "",
        "RAG-Based Customer Support Assistant",
        "with LangGraph &amp; Human-in-the-Loop Escalation",
    ]
    for line in meta:
        elements.append(Paragraph(line, ParagraphStyle('Meta',
            fontSize=11, textColor=DARK, alignment=TA_CENTER, spaceAfter=4)))
    elements.append(PageBreak())
    return elements


def generate_pdf(md_filename, pdf_filename, title, subtitle, images_map):
    """Generate a PDF from a markdown file with embedded diagrams."""
    md_path = os.path.join(DOCS_DIR, md_filename)
    pdf_path = os.path.join(DOCS_DIR, pdf_filename)

    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Remove the YAML-like header (title, version, date lines at top)
    # Skip until first ## or ---
    lines = md_text.split('\n')
    start = 0
    for idx, line in enumerate(lines):
        if line.startswith('## ') and idx > 2:
            start = idx
            break
    md_text = '\n'.join(lines[start:])

    styles = get_styles()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                           leftMargin=1*inch, rightMargin=1*inch,
                           topMargin=0.8*inch, bottomMargin=0.8*inch)

    elements = build_title_page(title, subtitle, styles)
    elements.extend(parse_markdown_to_elements(md_text, styles, images_map))

    doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"[OK] Generated: {pdf_path}")


if __name__ == "__main__":
    # Image mappings: keyword in section title -> image path
    hld_images = {
        "architecture diagram": os.path.join(IMG_DIR, "system_architecture.png"),
        "data flow": os.path.join(IMG_DIR, "data_flow.png"),
    }
    lld_images = {
        "workflow design": os.path.join(IMG_DIR, "langgraph_workflow.png"),
        "conditional routing": os.path.join(IMG_DIR, "decision_tree.png"),
        "hitl design": os.path.join(IMG_DIR, "hitl_flow.png"),
    }
    tech_images = {
        "system architecture": os.path.join(IMG_DIR, "system_architecture.png"),
        "workflow explanation": os.path.join(IMG_DIR, "langgraph_workflow.png"),
        "conditional logic": os.path.join(IMG_DIR, "decision_tree.png"),
        "hitl implementation": os.path.join(IMG_DIR, "hitl_flow.png"),
        "data flow": os.path.join(IMG_DIR, "data_flow.png"),
    }

    print("Generating PDFs with embedded diagrams...")
    generate_pdf("HLD.md", "HLD.pdf", "High-Level Design (HLD)",
                 "RAG-Based Customer Support Assistant", hld_images)
    generate_pdf("LLD.md", "LLD.pdf", "Low-Level Design (LLD)",
                 "RAG-Based Customer Support Assistant", lld_images)
    generate_pdf("Technical_Documentation.md", "Technical_Documentation.pdf",
                 "Technical Documentation",
                 "RAG-Based Customer Support Assistant", tech_images)
    print("\nAll PDFs generated in docs/ folder!")
