清理辅助文件
latexmk -C mian7.tex

命令行运行：
latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error mian7.tex

在.tex文件开头写一行：
%!LW recipe=latexmk (XeLaTeX)
可以让latex workshopo走这个recipe