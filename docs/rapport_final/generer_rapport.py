"""Produit le PDF et le HTML depuis contenu.json, sans modifier les expériences.
Dépendances de rédaction uniquement : reportlab et pypdf (hors application).
"""
import json
from pathlib import Path
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
fonts=Path(reportlab.__file__).parent/'fonts'
pdfmetrics.registerFont(TTFont('Vera',str(fonts/'Vera.ttf')))
pdfmetrics.registerFont(TTFont('VeraB',str(fonts/'VeraBd.ttf')))
pdfmetrics.registerFontFamily('Vera',normal='Vera',bold='VeraB',italic='Vera',boldItalic='VeraB')
s=getSampleStyleSheet()
s.add(ParagraphStyle(name='BodyR',fontName='Vera',fontSize=10.5,leading=14.5,spaceAfter=9,alignment=TA_JUSTIFY))
s.add(ParagraphStyle(name='TitleR',fontName='VeraB',fontSize=17,leading=21,spaceAfter=16))
s.add(ParagraphStyle(name='SubR',fontName='VeraB',fontSize=11.5,leading=16,spaceBefore=7,spaceAfter=7))
s.add(ParagraphStyle(name='CellR',fontName='Vera',fontSize=8.5,leading=12))
s.add(ParagraphStyle(name='CaptionR',fontName='Vera',fontSize=8.5,leading=11,spaceAfter=10,alignment=TA_CENTER))
s.add(ParagraphStyle(name='CoverR',fontName='Vera',fontSize=12,leading=19,spaceAfter=23,alignment=TA_CENTER))
content=json.loads((HERE/'contenu.json').read_text())
story=[]; html=[]
for i,page in enumerate(content):
    if i: story.append(PageBreak())
    html.append('<section class="page">')
    if i==0:story.append(Spacer(1,65))
    story.append(Paragraph(escape(page['title']),s['TitleR']))
    html.append('<h1>'+escape(page['title'])+'</h1>')
    for block in page['blocks']:
        if block[0] in ['p','h']:
            tag='p' if block[0]=='p' else 'h2'
            style=s['CoverR'] if i==0 else s['BodyR' if tag=='p' else 'SubR']
            story.append(Paragraph(block[1],style))
            html.append(f'<{tag}>{block[1]}</{tag}>')
        elif block[0]=='table':
            rows=block[1]; cols=len(rows[0]); widths=([120,157,205] if cols==3 and i==2 else [482/cols]*cols)
            t=Table([[Paragraph(escape(c),s['CellR']) for c in row] for row in rows],colWidths=widths,hAlign='CENTER')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAE6F3')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#BDB8CA')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.extend([t,Spacer(1,11)])
            html.append('<table>'+''.join('<tr>'+''.join('<td>'+escape(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</table>')
        elif block[0]=='image':
            _,path,caption,w,h=block
            story.extend([Image(str(ROOT/path),width=w,height=h,kind='proportional'),Paragraph(escape(caption),s['CaptionR'])])
            html.append(f'<figure><img src="../../{path}" style="max-height:{h}pt"><figcaption>{escape(caption)}</figcaption></figure>')
    html.append('</section>')
def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Vera',8)
    canvas.drawString(57,33,'INF1853 — IAnimaux | Automne 2026')
    canvas.drawRightString(538,33,str(doc.page));canvas.restoreState()
pdf=HERE/'IAnimaux_Rapport_final.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=(595.28,841.89),rightMargin=56.7,leftMargin=56.7,topMargin=48,bottomMargin=55,title='IAnimaux — Rapport du mini-projet',author='Megan Daoust et Luzia Coelho Bento Muaquesse')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
r=PdfReader(pdf)
if len(r.pages)!=10:raise ValueError(f'Pagination à corriger : {len(r.pages)} pages')
for index,(page,expected) in enumerate(zip(r.pages,content),1):
    text=page.extract_text()
    if expected['title'] not in text:raise ValueError(f'Page {index} décalée')
css='''@page{size:A4;margin:18mm 20mm}body{font:11pt/1.4 Arial,sans-serif;color:#222;margin:auto;max-width:170mm}.page{break-after:page}.page:last-child{break-after:auto}h1{font-size:17pt}h2{font-size:12pt}p{text-align:justify}table{width:100%;border-collapse:collapse;font-size:9pt;margin:12px 0}td{border:1px solid #bbb;padding:7px}tr:first-child{background:#eae6f3}figure{text-align:center;margin:12px 0}img{max-width:100%;object-fit:contain}figcaption{font-size:9pt}.page:first-child{text-align:center;padding-top:20mm}.page:first-child p{text-align:center}'''
(HERE/'IAnimaux_Rapport_final.html').write_text('<!doctype html><html lang="fr"><meta charset="utf-8"><title>IAnimaux — Rapport final</title><style>'+css+'</style>'+''.join(html)+'</html>')
(HERE/'verification.json').write_text(json.dumps({'pages':len(r.pages),'cover_and_references_included':True,'section_titles_verified':True,'authors':['Megan Daoust','Luzia Coelho Bento Muaquesse'],'missing_metadata':['Numéro de groupe'],'submission_filename':'Groupe{k}Final.pdf (à renommer après confirmation du groupe)'},ensure_ascii=False,indent=2)+'\n')
print(f'PDF vérifié : {len(r.pages)} pages, couverture et références incluses.')
