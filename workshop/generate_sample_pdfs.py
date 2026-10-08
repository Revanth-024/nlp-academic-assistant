"""Instructor authoring only: derive three teaching PDFs from canonical Markdown.

Uses optional ReportLab (bundled Codex authoring runtime), not an application
dependency. Run from repository root with a ReportLab-equipped Python:
``python -m workshop.generate_sample_pdfs``. No independent facts are authored.
"""
from pathlib import Path
import hashlib
import json
import re
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SOURCE_IDS = {"HCE-DOC04", "HCE-DOC08", "HCE-DOC09"}


def display_text(text):
    # Typography only; do not paraphrase or alter facts.
    text = text.translate(str.maketrans({"–": "-", "—": "-", "‑": "-", "“": '"', "”": '"', "’": "'", "→": "->"}))
    text = escape(text)
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)


def main():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.pdfgen.canvas import Canvas
    manifest_path = ROOT / "data/document_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("WorkshopBody", fontName="Helvetica", fontSize=10.5, leading=15,
                              spaceAfter=9, textColor=colors.HexColor("#172331")))
    styles.add(ParagraphStyle("WorkshopSmall", parent=styles["WorkshopBody"], fontSize=8.5, leading=12))
    styles["Title"].fontName = "Helvetica-Bold"
    styles["Title"].fontSize, styles["Title"].leading = 19, 24
    styles["Heading2"].spaceBefore, styles["Heading2"].spaceAfter = 12, 7
    styles["Heading2"].keepWithNext = True
    derived = []
    for source in manifest["academic_documents"]:
        if source["document_id"] not in SOURCE_IDS:
            continue
        canonical = ROOT / "data" / source["filename"]
        filename = "sample_pdfs/" + canonical.stem + ".pdf"
        output = ROOT / "data" / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = canonical.read_text(encoding="utf-8").splitlines()
        story = [Paragraph(escape(source["title"]), styles["Title"]),
                 Paragraph("Derived teaching PDF. Canonical source: " + escape(source["filename"]), styles["WorkshopSmall"]),
                 Spacer(1, 7)]
        index = 1  # Heading already rendered as the title.
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                index += 1
                continue
            if line.startswith("|"):
                rows = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    row = lines[index].strip()
                    if not re.fullmatch(r"[|\s:\-]+", row):
                        rows.append([Paragraph(display_text(cell.strip()), styles["WorkshopSmall"])
                                     for cell in row.strip("|").split("|")])
                    index += 1
                widths = [115, A4[0] - 203] if len(rows[0]) == 2 else None
                table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7edf3")),
                                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                           ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#c8d1db")),
                                           ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                           ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                                           ("TOPPADDING", (0, 0), (-1, -1), 5),
                                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                story.extend([table, Spacer(1, 10)])
                continue
            if line.startswith("#"):
                story.append(Paragraph(display_text(line.lstrip("# ")), styles["Heading2"]))
            else:
                paragraph = [line]
                while index + 1 < len(lines) and lines[index + 1].strip() and not lines[index + 1].startswith(("#", "|", "- ")):
                    index += 1
                    paragraph.append(lines[index].strip())
                story.append(Paragraph(display_text(" ".join(paragraph)), styles["WorkshopBody"]))
            index += 1

        def footer(canvas, doc):
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#526272"))
            canvas.drawString(44, 25, "Fictional Hindu College of Engineering | Derived workshop document")
            canvas.drawRightString(A4[0] - 44, 25, f"Page {doc.page}")

        def invariant_canvas(*args, **kwargs):
            kwargs["invariant"] = 1
            return Canvas(*args, **kwargs)

        pdf = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=44, rightMargin=44,
                                topMargin=40, bottomMargin=44, title=source["title"],
                                author="Synthetic NLP Workshop", pageCompression=1)
        pdf.build(story, onFirstPage=footer, onLaterPages=footer, canvasmaker=invariant_canvas)
        record = {"document_id": source["document_id"].replace("DOC", "PDF"), "filename": filename,
                  "title": source["title"], "category": "derived_teaching_pdf",
                  "canonical_document_id": source["document_id"], "canonical_filename": source["filename"],
                  "canonical_source_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
                  "intended_workshop_uses": ["pdf_extraction", "document_cleaning", "chunking", "provenance"]}
        derived.append(record)
        print(f"Created {filename} ({output.stat().st_size} bytes) from {source['filename']}")
    # Additive inventory only: original 14 documents and 4 papers stay untouched.
    manifest["derived_sample_pdfs"] = derived
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
