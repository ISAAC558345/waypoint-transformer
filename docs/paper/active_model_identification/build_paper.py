"""Render the canonical design manuscript and export an editable LaTeX source."""

from __future__ import annotations

import hashlib
import json
import re
from html import unescape
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
REPO_DEST = "docs/paper/active_model_identification"
PDF_NAME = "Task_Directed_Active_Model_Identification_MultiUAV.pdf"
PDF_PATH = HERE / PDF_NAME
DATA = json.loads((HERE / "manuscript.json").read_text(encoding="utf-8"))
W, H = 612, 792
MARGIN, GAP = 43, 18
COL = (W - 2 * MARGIN - GAP) / 2

# Embed a Unicode serif family so that formulas do not depend on a viewer's
# installation of the legacy, unembedded PDF Symbol font.
FONT, BOLD, ITALIC = "PaperSerif", "PaperSerif-Bold", "PaperSerif-Italic"
FONT_CANDIDATES = [
    (Path("C:/Windows/Fonts"), ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")),
    (Path("/usr/share/fonts/truetype/liberation2"),
     ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf")),
    (Path("/usr/share/fonts/truetype/liberation"),
     ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf")),
]
for font_root, font_files in FONT_CANDIDATES:
    if all((font_root / filename).is_file() for filename in font_files):
        for name, filename in zip((FONT, BOLD, ITALIC, "PaperSerif-BoldItalic"), font_files):
            pdfmetrics.registerFont(TTFont(name, str(font_root / filename)))
        break
else:
    raise RuntimeError("Install Times New Roman (Windows) or Liberation Serif (Linux) to render the PDF.")
pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=BOLD, italic=ITALIC,
                             boldItalic="PaperSerif-BoldItalic")
MATH = "PaperMath"
required_math = "ΣθΔ∫ελπβδΛμσγ≤≥∝∈′"
math_candidates = [
    (Path("C:/Windows/Fonts/cambria.ttc"), 1),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"), 0),
    (font_root / font_files[0], 0),
]
for math_path, collection_index in math_candidates:
    if math_path.is_file():
        math_font = TTFont(MATH, str(math_path), subfontIndex=collection_index)
        if all(ord(c) in math_font.face.charToGlyph for c in required_math):
            pdfmetrics.registerFont(math_font)
            break
else:
    raise RuntimeError("Install Cambria Math (Windows) or DejaVu Serif (Linux) for mathematical glyphs.")

STYLES = {
    "body": ParagraphStyle("body", fontName=FONT, fontSize=9.65, leading=11.6,
                           alignment=TA_JUSTIFY, firstLineIndent=9, spaceAfter=5),
    "abstract": ParagraphStyle("abstract", fontName=BOLD, fontSize=9.25,
                               leading=11.2, alignment=TA_JUSTIFY, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName=FONT, fontSize=10.5, leading=12.6,
                         alignment=TA_CENTER, spaceBefore=10, spaceAfter=6, keepWithNext=True),
    "h2": ParagraphStyle("h2", fontName=ITALIC, fontSize=9.9, leading=12,
                         spaceBefore=6, spaceAfter=4, keepWithNext=True),
    "equation": ParagraphStyle("equation", fontName=MATH, fontSize=9.0,
                               leading=12.2, alignment=TA_CENTER),
    "caption": ParagraphStyle("caption", fontName=FONT, fontSize=8.3,
                              leading=10.1, spaceBefore=4, spaceAfter=7),
    "tablecap": ParagraphStyle("tablecap", fontName=FONT, fontSize=8.5,
                               leading=10.4, alignment=TA_CENTER, spaceBefore=7,
                               spaceAfter=5, keepWithNext=True),
    "cell": ParagraphStyle("cell", fontName=FONT, fontSize=8.0, leading=9.6),
    "reference": ParagraphStyle("reference", fontName=FONT, fontSize=8.0,
                                leading=9.7, spaceAfter=5, leftIndent=16,
                                firstLineIndent=-16, splitLongWords=True),
    "algorithm": ParagraphStyle("algorithm", fontName=FONT, fontSize=8.5,
                                leading=10.4, spaceAfter=4, leftIndent=12,
                                firstLineIndent=-12),
}
for style in STYLES.values():
    style.allowWidows = 0
    style.allowOrphans = 0

REFERENCES = [
    ("wagenmaker2021task", "A. J. Wagenmaker, M. Simchowitz, and K. Jamieson, \"Task-Optimal Exploration in Linear Dynamical Systems,\" in Proc. ICML, PMLR 139, pp. 10641-10652, 2021.", "https://proceedings.mlr.press/v139/wagenmaker21a.html"),
    ("memmel2024asid", "M. Memmel, A. Wagenmaker, C. Zhu, D. Fox, and A. Gupta, \"ASID: Active Exploration for System Identification in Robotic Manipulation,\" in Proc. ICLR, 2024.", "https://proceedings.iclr.cc/paper_files/paper/2024/hash/ae8afec5dabb07091d8f85128250f2ff-Abstract-Conference.html"),
    ("aoyama2025poke", "M. Y. Aoyama, J. Moura, J. D. A. Ferrandis, and S. Vijayakumar, \"Poke and Strike: Learning Task-Informed Exploration Policies,\" in Proc. CoRL, PMLR 305, pp. 3008-3034, 2025.", "https://proceedings.mlr.press/v305/aoyama25a.html"),
    ("tang2026auto", "S. Tang, W.-H. Chen, and C. Liu, \"Auto-optimization of energy generation for wave energy converters with active learning,\" Ocean Engineering, vol. 351, Art. 124313, 2026, doi: 10.1016/j.oceaneng.2026.124313.", "https://doi.org/10.1016/j.oceaneng.2026.124313"),
    ("esrafilian2019learning", "O. Esrafilian, R. Gangula, and D. Gesbert, \"Learning to Communicate in UAV-Aided Wireless Networks: Map-Based Approaches,\" IEEE Internet Things J., vol. 6, no. 2, pp. 1791-1802, 2019, doi: 10.1109/JIOT.2018.2879682.", "https://arxiv.org/abs/1806.05165"),
    ("shrestha2023spectrum", "R. Shrestha, D. Romero, and S. P. Chepuri, \"Spectrum Surveying: Active Radio Map Estimation With Autonomous UAVs,\" IEEE Trans. Wireless Commun., vol. 22, no. 1, pp. 627-641, 2023, doi: 10.1109/TWC.2022.3197087.", "https://arxiv.org/abs/2201.04125"),
    ("song2026partialckm", "Y. Song, H. Lu, C. Zhang, B. Zheng, and Y. Zeng, \"Trajectory Optimization for Cellular-Connected UAV in Complex Environment With Partial CKM,\" IEEE Trans. Commun., vol. 74, pp. 8208-8221, 2026, doi: 10.1109/TCOMM.2026.3689212.", "https://arxiv.org/abs/2512.06452"),
    ("dai2024wind", "X. Dai, B. Duo, X. Yuan, and M. Di Renzo, \"Energy-Efficient UAV Communications in the Presence of Wind: 3D Modeling and Trajectory Design,\" IEEE Trans. Wireless Commun., vol. 23, no. 3, pp. 1840-1854, 2024, doi: 10.1109/TWC.2023.3292290.", "https://arxiv.org/abs/2304.06909"),
    ("li2026predictive", "B. Li et al., \"Sensing-Assisted Predictive Beamforming for UAV-Enabled Ocean Monitoring Networks,\" arXiv:2606.16717, preprint, 2026.", "https://arxiv.org/abs/2606.16717"),
    ("schulman2017ppo", "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, \"Proximal Policy Optimization Algorithms,\" arXiv:1707.06347, 2017.", "https://arxiv.org/abs/1707.06347"),
    ("yu2022mappo", "C. Yu et al., \"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,\" in Advances in Neural Information Processing Systems, vol. 35, pp. 24611-24624, 2022.", "https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html"),
]

EQUATIONS = {
    1: 'D = Σ<sub>k,j</sub> w<sub>j</sub> l<sub>j</sub> 1{t<sub>j</sub><super>recv</super> ≤ d<sub>j</sub>}.',
    2: 'v<sub>m</sub><super>air</super>(t) = v<sub>m</sub>(t) - w<sub>m</sub>(t),',
    3: 'P<sub>m</sub><super>actual</super>(t) = θ<sub>m</sub>P<sub>m</sub><super>nom</super>(v<sub>m</sub><super>air</super>, a<sub>m</sub>) + P<sub>m</sub><super>aux</super>(t),',
    4: 'E<sub>m</sub>(t + Δt) = E<sub>m</sub>(t) - ∫<sub>t</sub><super>t+Δt</super>P<sub>m</sub><super>actual</super>(s) ds,',
    5: 'y<sub>m,i</sub> = ΔE<sub>m,i</sub><super>meter</super> - E<sub>m,i</sub><super>aux</super> = z<sub>m,i</sub>θ<sub>m</sub> + ε<sub>m,i</sub>,',
    6: 'z<sub>m,i</sub> = ∫<sub>I<sub>i</sub></sub>P<sub>m</sub><super>nom</super>(s) ds,   ε<sub>m,i</sub> ~ N(0, σ<sub>m,i</sub><super>2</super>).',
    7: 'J(π) = E<sub>π</sub>[D/B<sub>ref</sub> - λ<sub>E</sub> Σ<sub>m</sub>E<sub>m</sub><super>used</super>/E<sub>ref</sub>],',
    8: 'Pr(E<sub>m,o</sub> + E<sub>m,return</sub> ≤ E<sub>m</sub> - E<sub>m,reserve</sub> | β<sub>n</sub>)<br/>≥ 1 - δ,',
    9: 'b<sub>n</sub>(θ) ∝ 1<sub>[θmin, θmax]</sub>(θ)<br/>× exp[-Λ<sub>n</sub>(θ - μ<sub>n</sub>)<super>2</super>/2],',
    10: 'Λ<sub>n+1</sub> = Λ<sub>n</sub> + z<sub>n</sub><super>2</super>/σ<sub>n</sub><super>2</super>,',
    11: 'μ<sub>n+1</sub> = (Λ<sub>n</sub>μ<sub>n</sub> + z<sub>n</sub>y<sub>n</sub>/σ<sub>n</sub><super>2</super>)/Λ<sub>n+1</sub>.',
    12: 'Q<sub>n</sub>(β,o) = E[r<sub>n</sub> + γ<super>Δn/Δ0</super>V<sub>n+1</sub>(β′) | β,o],',
    13: 'V<sub>n</sub><super>cal</super> = max<sub>o ∈ Ocal</sub> Q<sub>n</sub>(β<sub>n</sub>,o)<br/>- max<sub>o ∈ Oexec</sub> Q<sub>n</sub>(β<sub>n</sub>,o).',
}


def architecture():
    labels = [
        "Received data and onboard telemetry",
        "Parameter belief and predictive validation",
        "Task-value scheduler and legal options",
        "Service / calibration / backhaul / return",
        "Local flight and energy feasibility guards",
        "Physical motion and transport accounting",
        "New observations and delayed messages",
    ]
    drawing = Drawing(COL, 179)
    x, width, height, gap = 14, COL - 28, 18, 7
    for i, label in enumerate(labels):
        y = 179 - (i + 1) * height - i * gap
        drawing.add(Rect(x, y, width, height, strokeColor=colors.HexColor("#586674"),
                         fillColor=colors.HexColor("#F6F7F8"), strokeWidth=0.5))
        drawing.add(String(COL / 2, y + 6, label, fontName=FONT,
                           fontSize=7.6, textAnchor="middle"))
        if i < len(labels) - 1:
            drawing.add(Line(COL / 2, y, COL / 2, y - gap, strokeWidth=0.6))
            drawing.add(Polygon([COL/2, y-gap, COL/2-2, y-gap+3, COL/2+2, y-gap+3],
                                fillColor=colors.black, strokeColor=None))
    drawing.add(Line(x, 20, 4, 20, strokeWidth=0.5))
    drawing.add(Line(4, 20, 4, 170, strokeWidth=0.5))
    drawing.add(Line(4, 170, x, 170, strokeWidth=0.5))
    drawing.add(Polygon([x,170,x-3,168,x-3,172],fillColor=colors.black,strokeColor=None))
    return drawing, labels


def make_table(block):
    data = [[Paragraph("<b>" + escape(c) + "</b>", STYLES["cell"]) for c in block["headers"]]]
    data += [[Paragraph(escape(c), STYLES["cell"]) for c in row] for row in block["rows"]]
    table = Table(data, colWidths=[COL*x for x in block["widths"]], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LINEABOVE", (0,0), (-1,0), 0.7, colors.black),
        ("LINEBELOW", (0,0), (-1,0), 0.45, colors.black),
        ("LINEBELOW", (0,-1), (-1,-1), 0.7, colors.black),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    return table


def page_decorations(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.drawRightString(W-MARGIN, H-26, str(doc.page))
    canvas.setFont(ITALIC, 7.4)
    canvas.setFillColor(colors.HexColor("#505050"))
    canvas.drawString(MARGIN, 23, "Research design manuscript | 13 November 2026 | Empirical evaluation pending")
    if doc.page == 1:
        title_style = ParagraphStyle("title", fontName=FONT, fontSize=20, leading=23,
                                     alignment=TA_CENTER)
        title = Paragraph(escape(DATA["title"]), title_style)
        _, th = title.wrap(W-2*MARGIN, 110)
        title.drawOn(canvas, MARGIN, H-MARGIN-th)
        canvas.setFillColor(colors.black)
        canvas.setFont(FONT, 11)
        canvas.drawCentredString(W/2, H-MARGIN-th-19, DATA["author"])
        canvas.setFont(ITALIC, 8.2)
        canvas.drawCentredString(W/2, H-MARGIN-th-35,
                                  "Research design | Task-directed model verification | Experiments pending")
    canvas.restoreState()


def build_pdf():
    story = []
    for block in DATA["blocks"]:
        kind = block["type"]
        if kind in ("p", "abstract", "h1", "h2"):
            prefix = "<b>Abstract - </b>" if kind == "abstract" else ""
            story.append(Paragraph(prefix + block["text"], STYLES["body" if kind == "p" else kind]))
        elif kind == "equation":
            equation = Paragraph(EQUATIONS[block["number"]], STYLES["equation"])
            number = Paragraph(f'({block["number"]})', STYLES["equation"])
            table = Table([[equation, number]], colWidths=[COL-22, 22])
            table.setStyle(TableStyle([
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING", (0,0), (-1,-1), 0),
                ("RIGHTPADDING", (0,0), (-1,-1), 0),
                ("TOPPADDING", (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ]))
            story.append(table)
        elif kind == "table":
            story += [Paragraph(escape(block["caption"]), STYLES["tablecap"]),
                      make_table(block), Spacer(1,7)]
        elif kind == "figure":
            drawing, _ = architecture()
            story.append(KeepTogether([Spacer(1,5), drawing,
                         Paragraph(escape(block["caption"]), STYLES["caption"])]))
        elif kind == "algorithm":
            cells = [Paragraph(f"<b>{escape(block['caption'])}</b>", STYLES["caption"])]
            cells += [Paragraph(f"{i}. {escape(step)}", STYLES["algorithm"])
                      for i, step in enumerate(block["steps"],1)]
            story.append(KeepTogether(cells + [Spacer(1,7)]))
    story.append(Paragraph("References", STYLES["h1"]))
    for i, (_, reference, url) in enumerate(REFERENCES, 1):
        story.append(Paragraph(f'[{i}] {escape(reference)} '
                     f'<link href="{url}" color="#294A65">[source]</link>', STYLES["reference"]))

    doc = BaseDocTemplate(str(PDF_PATH), pagesize=(W,H), leftMargin=MARGIN,
                          rightMargin=MARGIN, title=DATA["title"], author=DATA["author"],
                          subject=DATA["subtitle"], pageCompression=1)
    first = [Frame(MARGIN+i*(COL+GAP), 43, COL, H-43-160, id=f"first{i}",
                   leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0) for i in range(2)]
    later = [Frame(MARGIN+i*(COL+GAP),43,COL,H-43-43,id=f"later{i}",
                   leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0) for i in range(2)]
    doc.addPageTemplates([
        PageTemplate(id="first", frames=first, onPage=page_decorations, autoNextPageTemplate="later"),
        PageTemplate(id="later", frames=later, onPage=page_decorations),
    ])
    doc.build(story)


def latex_escape(text):
    text = unescape(re.sub(r"<[^>]+>", "", text))
    replacements = {"\\":"\\textbackslash{}", "&":"\\&", "%":"\\%", "$":"\\$",
                    "#":"\\#", "_":"\\_", "{":"\\{", "}":"\\}",
                    "~":"\\textasciitilde{}", "^":"\\textasciicircum{}"}
    text = "".join(replacements.get(c,c) for c in text)
    return re.sub(r"\[(\d+)\]", lambda m: r"\cite{"+REFERENCES[int(m.group(1))-1][0]+"}", text)


def export_latex():
    output = [
        "% Generated from manuscript.json. The distributed PDF is rendered by build_paper.py.",
        "% Optional TeX build: pdflatex manuscript; bibtex manuscript; pdflatex manuscript twice.",
        r"\documentclass[10pt,twocolumn,letterpaper]{article}",
        r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}",
        r"\usepackage{mathptmx}",
        r"\usepackage[letterpaper,margin=0.60in,columnsep=0.25in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,tabularx,graphicx}",
        r"\usepackage{tikz}", r"\usetikzlibrary{arrows.meta,positioning}",
        r"\usepackage{xurl}", r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\emergencystretch}{2em}",
        r"\renewcommand{\thetable}{\Roman{table}}",
        r"\title{"+latex_escape(DATA["title"])+"}",
        r"\author{"+latex_escape(DATA["author"])+"}",
        r"\date{\small Research design manuscript | 13 November 2026 | Empirical evaluation pending}",
        r"\begin{document}", r"\maketitle",
        r"\nocite{"+",".join(x[0] for x in REFERENCES)+"}",
    ]
    for block in DATA["blocks"]:
        kind = block["type"]
        if kind == "abstract":
            output += [r"\begin{abstract}",latex_escape(block["text"]),r"\end{abstract}"]
        elif kind == "p":
            output += [latex_escape(block["text"]), ""]
        elif kind == "h1":
            title = re.sub(r"^[IVX]+\. ","",block["text"])
            command = r"\section*" if title == "Reproducibility Note" else r"\section"
            output.append(command+"{"+latex_escape(title)+"}")
        elif kind == "h2":
            title = re.sub(r"^[A-Z]\. ","",block["text"])
            output.append(r"\subsection{"+latex_escape(title)+"}")
        elif kind == "equation":
            output += [r"\begin{equation}",block["latex"],r"\end{equation}"]
        elif kind == "table":
            caption = re.sub(r"^TABLE [IVX]+\. ","",block["caption"])
            output += [r"\begin{table}[t]",r"\centering\footnotesize",
                       r"\caption{"+latex_escape(caption)+"}",
                       r"\begin{tabularx}{\columnwidth}{"+"X"*len(block["headers"])+"}",
                       r"\toprule", " & ".join(latex_escape(x) for x in block["headers"])+r" \\",
                       r"\midrule"]
            output += [" & ".join(latex_escape(x) for x in row)+r" \\" for row in block["rows"]]
            output += [r"\bottomrule",r"\end{tabularx}",r"\end{table}"]
        elif kind == "figure":
            _, labels = architecture()
            output += [r"\begin{figure}[t]",r"\centering",
                       r"\begin{tikzpicture}[node distance=3mm,>=Stealth]",
                       r"\tikzset{box/.style={draw,rectangle,text width=.84\columnwidth,align=center,font=\scriptsize,minimum height=6mm}}"]
            for i, label in enumerate(labels):
                placement = "" if i==0 else f",below=of n{i-1}"
                output.append(f"\\node[box{placement}] (n{i}) {{{latex_escape(label)}}};")
                if i:
                    output.append(f"\\draw[->] (n{i-1}) -- (n{i});")
            output += [r"\draw[->] (n6.west) -- ++(-3mm,0) |- (n0.west);",
                       r"\end{tikzpicture}",
                       r"\caption{"+latex_escape(re.sub(r"^Fig\. 1\. ","",block["caption"]))+"}",
                       r"\end{figure}"]
        elif kind == "algorithm":
            output += [r"\begin{figure}[t]",r"\small\hrule\medskip",
                       r"\textbf{"+latex_escape(block["caption"])+"}",r"\begin{enumerate}"]
            output += [r"\item "+latex_escape(s) for s in block["steps"]]
            output += [r"\end{enumerate}\hrule",r"\end{figure}"]
    output += [r"\bibliographystyle{IEEEtran}",r"\bibliography{references_verified}",r"\end{document}"]
    (HERE / "manuscript.tex").write_text("\n".join(output)+"\n",encoding="utf-8")


def verify():
    reader = PdfReader(PDF_PATH)
    texts = [p.extract_text() or "" for p in reader.pages]
    text = "\n".join(texts)
    for phrase in ["Empirical evaluation pending", "6.421", "1.546", "7.921", "References", "ASID", "M0B"]:
        if phrase not in text:
            raise RuntimeError(f"Expected content missing: {phrase}")
    for marker in ["\ufffd", "\u25a0"]:
        if marker in text:
            raise RuntimeError("Unreadable character found in PDF extraction")
    citations = sorted(set(int(x) for x in re.findall(r"\[(\d+)\]",text)))
    if citations != list(range(1,len(REFERENCES)+1)):
        raise RuntimeError(f"Reference numbering mismatch: {citations}")
    # Independently check the analytical example. These are arithmetic checks,
    # not simulated episodes and not claimed performance improvements.
    tight_passive = 6-.01*(.5*100+.5*125)
    tight_probe = .5*(10-.01*(4.8+180))+.5*(6-.01*(6+125))
    loose_direct = 10-.01*(.5*180+.5*225)
    loose_probe = .5*(10-.01*(4.8+180))+.5*(10-.01*(6+225))
    assert abs(tight_passive-4.875)<1e-10
    assert abs(tight_probe-6.421)<1e-10
    assert abs((tight_probe-tight_passive)-1.546)<1e-10
    assert abs(loose_direct-7.975)<1e-10
    assert abs(loose_probe-7.921)<1e-10
    assert abs((loose_probe-loose_direct)+.054)<1e-10
    entries = re.findall(r"@\w+\{([^,]+),",(HERE/"references_verified.bib").read_text(encoding="utf-8"))
    assert set(entries)==set(r[0] for r in REFERENCES)
    report = {
        "status":"design_manuscript", "pages":len(reader.pages),
        "words":len(text.split()), "page_characters":[len(x) for x in texts],
        "reference_count":len(REFERENCES),
        "hyperlinks":sum(len(p.get("/Annots",[])) for p in reader.pages),
        "pdf_sha256":hashlib.sha256(PDF_PATH.read_bytes()).hexdigest(),
        "source_git_blob":"92ba3fe88ad8a6e99dbf45b655268aa300f802f9",
        "new_simulation_steps":0,"new_training_updates":0,
        "analytic_utility":{"tight_direct":tight_passive,"tight_probe":tight_probe,
                            "loose_direct":loose_direct,"loose_probe":loose_probe},
        "pdf_renderer":"ReportLab; editable LaTeX exported from the same canonical content",
        "unicode_fonts_embedded":True,
        "latex_compiled":False,
    }
    (HERE/"verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=True))


if __name__ == "__main__":
    export_latex()
    build_pdf()
    verify()
