"""Export only completed experiment comparisons; never launches a solver."""
import argparse
import json
from pathlib import Path
from src.experiment_plots import FAMILIES, export_experiment_plots


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--families",nargs="+",choices=FAMILIES)
    args=parser.parse_args(argv)
    path=export_experiment_plots(args.output_root,args.families)
    manifest=json.loads(Path(path).read_text(encoding="utf-8"))
    print(json.dumps({"manifest":path,"status":manifest["status"],
                      "complete_figures":manifest.get("complete_figure_count",0),
                      "skipped_figures":manifest.get("skipped_figure_count",0)},ensure_ascii=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
