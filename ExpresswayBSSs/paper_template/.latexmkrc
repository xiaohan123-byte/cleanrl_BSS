# .latexmkrc — latexmk 配置
# 默认使用 xelatex（支持中文 + 现代字体）
$pdf_mode = 5;
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
# 输出目录
$aux_dir = 'build';
$out_dir = 'build';
# 清理辅助文件
$clean_full_ext = 'synctex.gz synctex.gz(busy) run.xml tex.bak bib bak bbl bcf fdb_latexmk fls nav snm run tdo %R-blx.bib';
