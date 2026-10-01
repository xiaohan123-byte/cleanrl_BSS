"""Read-only structural validation and contact sheet of the delivered deck."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main', 'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
stats = []
with ZipFile(root / '引言汇报_红色学术_6页.pptx') as z:
    for i in range(1, 7):
        slide = ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
        notes = ET.fromstring(z.read(f'ppt/notesSlides/notesSlide{i}.xml'))
        texts = [t.text for t in slide.findall('.//a:t', ns)]
        note_texts = [t.text for t in notes.findall('.//a:t', ns)]
        s = {'slide': i, 'native_shapes': len(slide.findall('.//p:sp', ns)), 'text_runs': len(texts), 'pictures': len(slide.findall('.//p:pic', ns)), 'notes': ''.join(note_texts)}
        assert s['text_runs'] > 10 and s['native_shapes'] > 10
        assert len(s['notes']) > 50
        stats.append(s)
    assert len([n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')]) == 6
(root / 'preview/package_check.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
render_dir = root / 'preview/powerpoint'
if not (render_dir / 'slide-1.png').exists():
    render_dir = root / 'preview'
canvas = Image.new('RGB', (1640, 1510), '#E7E7EB')
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
for i in range(6):
    im = Image.open(render_dir / f'slide-{i+1}.png').convert('RGB')
    im.thumbnail((790, 445))
    x = 20 + (i % 2)*810
    y = 20 + (i // 2)*500
    canvas.paste(im, (x, y))
    draw.text((x, y+457), f'{i+1:02d} / 06', fill='#751324', font=font)
canvas.save(root / '六页预览.png')
print(json.dumps(stats, ensure_ascii=False))
