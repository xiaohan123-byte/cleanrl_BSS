"""Assemble the self-contained HTML, export config and presenter script."""
from pathlib import Path
from html.parser import HTMLParser
import json

ROOT = Path(__file__).resolve().parent

class Slides(HTMLParser):
    def __init__(self):
        super().__init__()
        self.slides = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'section' and 'data-label' in a:
            self.slides.append((a['data-label'], a['data-speaker-notes']))

source = (ROOT / 'intro.html').read_text(encoding='utf-8')
parser = Slides()
parser.feed(source)
assert len(parser.slides) == 6
runtime = (ROOT / 'deck-stage.js').read_text(encoding='utf-8')
standalone = source.replace('<script src="deck-stage.js"></script>', '<script>\n' + runtime.replace('</script>', '<\\/script>') + '\n</script>')
(ROOT / '引言汇报.html').write_text(standalone, encoding='utf-8')

md = ['# 引言汇报逐页讲稿', '', '共 6 页，不另设封面。建议总时长约 3 分钟；第 4 页可略多讲，结构页快速过渡。', '']
times = ['约 30 秒', '约 35 秒', '约 35 秒', '约 40 秒', '约 30 秒', '约 15 秒']
for (label, notes), duration in zip(parser.slides, times):
    md.extend([f'## {label}（{duration}）', '', notes, ''])
md.extend(['---', '', '编辑说明：讲稿同步写入 PPT 的备注页。HTML 按 N 可显示/隐藏讲稿，方向键翻页，F 全屏。', '', '内容校准：文章结构按当前 main.tex 的实际章节排列；管理分析的结论尚待实验验证。', '', '背景数据：[IEA《Global EV Outlook 2026》](https://www.iea.org/reports/global-ev-outlook-2026/trends-in-electric-cars)。4400万辆为电动汽车总体规模，包含纯电动与插电式混合动力，不代表可换电车辆数。', ''])
(ROOT / '逐页讲稿.md').write_text('\n'.join(md), encoding='utf-8')

config = {
    'width': 1920, 'height': 1080,
    'slides': [{'showJs': f"document.querySelector('deck-stage').goTo({i})", 'selector': 'deck-stage > [data-deck-active]'} for i in range(6)],
    'hideSelectors': ['.note-toggle', '#notesPanel'],
    'resetTransformSelector': 'deck-stage',
    'filename': '引言汇报_红色学术_6页',
}
(ROOT / 'export_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'slides': len(parser.slides), 'note_characters': [len(n) for _, n in parser.slides]}, ensure_ascii=False))
