$ErrorActionPreference = 'Stop'
$deckRoot = $PSScriptRoot
$deckFile = Join-Path $deckRoot '引言汇报_红色学术_6页.pptx'
$renderDir = Join-Path $deckRoot 'preview/powerpoint'
New-Item -ItemType Directory -Path $renderDir -Force | Out-Null
$powerpoint = New-Object -ComObject PowerPoint.Application
$originalCount = $powerpoint.Presentations.Count
$presentation = $null
try {
    $presentation = $powerpoint.Presentations.Open($deckFile, -1, 0, 0)
    $stats = @()
    foreach ($slide in $presentation.Slides) {
        $slide.Export((Join-Path $renderDir ('slide-' + $slide.SlideIndex + '.png')), 'PNG', 1920, 1080)
        $textCount = 0
        foreach ($shape in $slide.Shapes) { if ($shape.HasTextFrame -and $shape.TextFrame.HasText) { $textCount++ } }
        $stats += [PSCustomObject]@{ slide = $slide.SlideIndex; shapes = $slide.Shapes.Count; textShapes = $textCount }
    }
    $presentation.SaveAs((Join-Path $deckRoot '引言汇报_预览.pdf'), 32)
    $stats | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $renderDir 'stats.json') -Encoding UTF8
    $stats | ConvertTo-Json -Compress | Write-Output
}
finally {
    if ($null -ne $presentation) { $presentation.Close(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) }
    if ($originalCount -eq 0 -and $powerpoint.Presentations.Count -eq 0) { $powerpoint.Quit() }
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerpoint)
}
