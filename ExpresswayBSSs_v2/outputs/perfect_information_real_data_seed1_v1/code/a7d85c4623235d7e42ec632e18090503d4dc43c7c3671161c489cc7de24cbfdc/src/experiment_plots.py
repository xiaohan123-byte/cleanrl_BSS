"""Publication-format plots from completed, paired experimental records only.

No solver or synthetic-data generator is imported. Missing comparisons create
manifest entries, never zero-filled bars. Training replicates remain separate.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path

from .experiment_control import atomic_json, utc_now

FAMILIES = ("core", "joint", "terminal", "training", "sensitivity")
COST_METRICS = ("service_income_yuan", "charging_cost_yuan", "adjustment_cost_yuan", "reservation_failure_cost_yuan")
OPERATION_METRICS = (("net_profit_yuan", "Net profit (CNY)", 1.),
                     ("reservation_failure_rate", "Reservation failure (%)", 100.),
                     ("random_service_rate", "Random service (%)", 100.),
                     ("solver_seconds_mean", "Mean solve time (s)", 1.))


class IncompletePlot(ValueError):
    pass


def _read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stat(group, metric):
    stat = group.get("statistics", {}).get(metric)
    if not stat or type(stat.get("n")) is not int or stat["n"] < 1 or not _number(stat.get("mean")):
        raise IncompletePlot("missing finite completed statistic: " + metric)
    if stat["n"] > len(group.get("scenario_seeds", [])):
        raise IncompletePlot("statistical n exceeds distinct scenarios: " + metric)
    result = {"mean":float(stat["mean"]),"n":stat["n"],"ci95":None}
    low, high = stat.get("ci95_low"), stat.get("ci95_high")
    if stat["n"] >= 2 and _number(low) and _number(high) and low <= stat["mean"] <= high:
        result["ci95"] = [float(low),float(high)]
    return result


def _require_groups(report, names, metrics, expected_seeds):
    groups = report.get("groups", {})
    missing = [name for name in names if groups.get(name, {}).get("status") != "complete"]
    if missing:
        raise IncompletePlot("comparison groups incomplete or missing: " + ", ".join(missing))
    reference = list(expected_seeds)
    if not reference or len(reference) != len(set(reference)):
        raise IncompletePlot("frozen test scenario seeds are unavailable")
    selected = []
    for name in names:
        group = groups[name]
        seeds = group.get("scenario_seeds", [])
        if len(seeds) != len(set(seeds)) or set(seeds) != set(reference):
            raise IncompletePlot("comparison does not use the complete paired test set: " + name)
        selected.append({"group":name,"values":{metric:_stat(group,metric) for metric in metrics}})
    return selected


@contextmanager
def _plot_style():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titlesize":10,"axes.labelsize":9,
                         "legend.fontsize":8,"pdf.fonttype":42,"ps.fonttype":42,
                         "savefig.dpi":300,"axes.grid":False}):
        yield plt


def _save(figure, directory, identifier, fixture):
    prefix = "fixture_" if fixture else ""
    paths = {extension:directory/(prefix+identifier+"."+extension) for extension in ("png","pdf")}
    directory.mkdir(parents=True,exist_ok=True)
    for extension,path in paths.items():
        pending = path.with_name(path.stem+".tmp."+extension)
        figure.savefig(pending,format=extension,dpi=300,bbox_inches="tight",
                       metadata={"Creator":"ExpresswayBSSs experiment plot exporter"})
        os.replace(pending,path)
    return {key:str(value.resolve()) for key,value in paths.items()}


def _ci(ax, xs, values, scale=1., color="black"):
    for x,stat in zip(xs,values):
        if stat["ci95"] is None:
            continue
        center = stat["mean"] * scale
        low,high = (value*scale for value in stat["ci95"])
        ax.errorbar([x],[center],yerr=[[center-low],[high-center]],fmt="none",
                    ecolor=color,elinewidth=.9,capsize=3,capthick=.9)


def _profit_cost(directory, identifier, title, labels, selected, fixture):
    for row in selected:
        values = row["values"]
        counts = {value["n"] for value in values.values()}
        expected = values[COST_METRICS[0]]["mean"] - sum(values[key]["mean"] for key in COST_METRICS[1:])
        if len(counts) != 1 or not math.isclose(expected, values["net_profit_yuan"]["mean"], rel_tol=1e-8, abs_tol=1e-6):
            raise IncompletePlot("income/cost decomposition does not reconcile on the same scenarios")
    with _plot_style() as plt:
        fig,axes = plt.subplots(1,2,figsize=(10.4,3.8))
        colors = ["#6d8295","#469c93","#d4983f","#865a9c"]
        xs = list(range(len(selected)))
        profit = [row["values"]["net_profit_yuan"] for row in selected]
        axes[0].bar(xs,[value["mean"] for value in profit],color=colors[:len(xs)],width=.64)
        _ci(axes[0],xs,profit)
        axes[0].axhline(0,color="#777777",linewidth=.6)
        axes[0].set_ylabel("Net profit (CNY)")
        axes[0].set_title("Actual operating net profit")
        income = [row["values"][COST_METRICS[0]]["mean"] for row in selected]
        axes[1].bar(xs,income,width=.64,color="#469c93",label="Service income")
        bottom = [0.]*len(xs)
        for metric,label,color in zip(COST_METRICS[1:],["Electricity","Path changes","Reservation failure"],
                                      ["#647c98","#d4983f","#bb6774"]):
            values = [-row["values"][metric]["mean"] for row in selected]
            axes[1].bar(xs,values,bottom=bottom,width=.64,color=color,label=label)
            bottom = [a+b for a,b in zip(bottom,values)]
        axes[1].axhline(0,color="#777777",linewidth=.6)
        axes[1].set_ylabel("Income (+) / costs (-), CNY")
        axes[1].set_title("Actual income and cost decomposition")
        axes[1].legend(loc="best",frameon=False)
        for ax in axes:
            ax.set_xticks(xs,labels)
            ax.grid(axis="y",alpha=.2)
        fig.suptitle(("FIXTURE — not experimental results | " if fixture else "")+title)
        counts = ", ".join(str(value["n"]) for value in profit)
        fig.text(.01,.005,"Scenario counts: "+counts+". Error bars: recorded 95% CI when n ≥ 2; otherwise omitted. Cost bars: means.",fontsize=7)
        fig.tight_layout(rect=(0,.04,1,.93))
        outputs = _save(fig,directory,identifier,fixture)
        plt.close(fig)
    return outputs


def _training_data(suite_root, kind, variant, replicate, plan):
    directory = suite_root/"models/base"/(kind+"_"+variant+"_rep"+str(replicate))
    selection_path = directory/"selection.json"
    selection = _read(selection_path)
    count = plan["budget"]["outer_iterations"]
    if not selection or len(selection.get("candidates",[])) != count:
        raise IncompletePlot("all planned outer fits/validations have not finished: "+str(directory))
    if selection.get("kind") != kind or selection.get("variant") != variant or selection.get("training_replicate") != replicate:
        raise IncompletePlot("training selection metadata does not match its figure")
    rows,sources = [],[selection_path]
    for iteration in range(count):
        fit_path = directory/("iteration_"+str(iteration)+"_fit.json")
        validation_path = directory/("iteration_"+str(iteration)+"_validation.json")
        fit,validation = _read(fit_path),_read(validation_path)
        if not fit or not validation or fit.get("iteration") != iteration or validation.get("iteration") != iteration:
            raise IncompletePlot("missing complete fit/validation for outer iteration "+str(iteration))
        if (fit.get("training_replicate") != replicate
                or selection["candidates"][iteration] != validation
                or not fit.get("only_current_policy_labels")):
            raise IncompletePlot("fit/selection provenance does not match its validation record")
        diagnostics = fit.get("diagnostics",{})
        history = diagnostics.get("history",[])
        expected_epochs = diagnostics.get("epochs")
        epochs = [row.get("epoch") for row in history]
        losses = [row.get("mse_scaled") for row in history]
        jobs = validation.get("validation_jobs",[])
        score = validation.get("validation_net_profit_yuan")
        planned_epochs = plan["budget"]["fit_epochs"] if kind == "relu" else 1
        if (diagnostics.get("kind") != kind or expected_epochs != planned_epochs
                or not history or epochs[-1] != expected_epochs
                or any(not isinstance(epoch,int) or epoch<=0 for epoch in epochs)
                or any(a>=b for a,b in zip(epochs,epochs[1:]))
                or any(not _number(value) or value<0 for value in losses)
                or not _number(score) or len(jobs)!=len(set(jobs))
                or len(jobs)!=plan["budget"]["validation_days"]):
            raise IncompletePlot("fit/validation records are incomplete or invalid")
        rows.append({"iteration":iteration,"epochs":epochs,"training_mse_scaled":losses,
                     "validation_net_profit_yuan":float(score),"validation_n":len(jobs),
                     "target_scale_yuan":diagnostics.get("target_scale_yuan",plan.get("target_scale_yuan"))})
        sources += [fit_path,validation_path]
    return rows,sources


def _training_plot(directory,identifier,rows,kind,variant,replicate,fixture):
    scales = {row["target_scale_yuan"] for row in rows}
    if len(scales)!=1 or not _number(next(iter(scales))):
        raise IncompletePlot("training target scales differ or are unavailable")
    scale = next(iter(scales))
    with _plot_style() as plt:
        fig,axes = plt.subplots(1,2,figsize=(9.8,3.5))
        for row in rows:
            axes[0].plot(row["epochs"],row["training_mse_scaled"],marker="o" if len(row["epochs"])==1 else None,
                         label="Outer "+str(row["iteration"]+1))
        axes[0].set_xlabel("Recorded least-squares fit" if kind=="linear" else "Regression epoch")
        axes[0].set_ylabel("MSE of return / "+format(scale,"g")+" CNY")
        axes[0].legend(frameon=False)
        axes[1].plot([row["iteration"]+1 for row in rows],[row["validation_net_profit_yuan"] for row in rows],marker="o",color="#469c93")
        axes[1].set_xlabel("Outer policy iteration")
        axes[1].set_ylabel("Mean validation net profit (CNY)")
        axes[1].set_xticks([row["iteration"]+1 for row in rows])
        for ax in axes:ax.grid(alpha=.2)
        title=kind+" / "+variant+" — training replicate "+str(replicate+1)
        fig.suptitle(("FIXTURE — not experimental results | " if fixture else "")+title)
        fig.text(.01,.005,"Each outer fit uses a fresh policy batch. Validation means are observed scores; no confidence intervals are inferred.",fontsize=7)
        fig.tight_layout(rect=(0,.045,1,.91))
        outputs=_save(fig,directory,identifier,fixture)
        plt.close(fig)
    return outputs


def _sensitivity_groups(report,factor,values,replicate):
    groups = report.get("groups",{})
    output=[]
    for value in values:
        tag=factor+"_"+format(value,"g").replace(".","p")
        if factor=="horizon":
            options=[("base/"+tag+"_zero","base/"+tag+"_relu_rep"+str(replicate)),
                     ("sensitivity/"+tag+"_zero","sensitivity/"+tag+"_relu_rep"+str(replicate))]
        else:
            options=[(tag+"/zero",tag+"/relu_full_rep"+str(replicate)),
                     ("sensitivity/"+tag+"_zero","sensitivity/"+tag+"_relu_rep"+str(replicate))]
        pair=next((pair for pair in options if all(name in groups for name in pair)),options[0])
        output.extend(pair)
    return output


def _sensitivity_plot(directory,identifier,factor,values,selected,replicate,fixture):
    x=[value/12. for value in values] if factor=="horizon" else list(values)
    labels={"horizon":"Prediction horizon (h)","demand":"Demand multiplier",
            "battery":"Batteries / charging slots per station","power":"Station power multiplier"}
    with _plot_style() as plt:
        fig,axes=plt.subplots(2,2,figsize=(9.6,6.2))
        for ax,(metric,ylabel,scale) in zip(axes.flat,OPERATION_METRICS):
            for offset,label,color in ((0,"Zero terminal","#647c98"),(1,"Full ReLU","#469c93")):
                stats=[selected[2*i+offset]["values"][metric] for i in range(len(values))]
                ax.plot(x,[stat["mean"]*scale for stat in stats],marker="o",label=label,color=color)
                _ci(ax,x,stats,scale,color)
            ax.set_xlabel(labels[factor]);ax.set_ylabel(ylabel);ax.set_xticks(x);ax.grid(alpha=.2)
        axes[0,0].legend(frameon=False)
        title=factor.capitalize()+" sensitivity — training replicate "+str(replicate+1)
        fig.suptitle(("FIXTURE — not experimental results | " if fixture else "")+title)
        note="Base network fixed across horizons." if factor=="horizon" else "Full ReLU retrained separately for each physical/demand configuration."
        fig.text(.01,.005,note+" Error bars use recorded scenario-level 95% CI only when n ≥ 2.",fontsize=7)
        fig.tight_layout(rect=(0,.045,1,.94))
        outputs=_save(fig,directory,identifier,fixture)
        plt.close(fig)
    return outputs


def export_experiment_plots(output_root, families=None):
    """Write PNG+PDF only for complete requested comparisons; return manifest path.

    The manifest lists exact plotted values, source hashes and every skipped
    comparison. A partial suite can yield some complete figures without being
    labelled a complete experiment. Default output: output_root/figures/.
    """
    root=Path(output_root).resolve();suite=root/"formal_suite";directory=root/"figures"
    families=tuple(families or FAMILIES)
    if set(families)-set(FAMILIES):raise ValueError("unknown plot family")
    report_path,plan_path=suite/"report.json",suite/"plan.json"
    report,plan=_read(report_path),_read(plan_path)
    manifest={"schema_version":1,"generated_at":utc_now().isoformat(),"status":"pending",
              "requested_families":list(families),"scope":"figure export only; does not mark the experiment suite complete",
              "source_report":str(report_path),"source_report_sha256":_hash(report_path),
              "source_plan":str(plan_path),"source_plan_sha256":_hash(plan_path),
              "observed_suite_status":report.get("status") if report else "not_started", "figures":[]}
    manifest_path=directory/"manifest.json"
    if not report or not plan:
        manifest["complete_figure_count"]=0
        manifest["skipped_figure_count"]=0
        manifest["reason"]="formal report/plan unavailable; no experimental figure can be drawn"
        atomic_json(manifest_path,manifest)
        return str(manifest_path)
    fixture=bool(report.get("fixture") or plan.get("fixture"))
    manifest["fixture"]=fixture
    replicates=range(plan["budget"]["training_replicates"])
    seeds=plan.get("seeds",{}).get("test",[])
    metrics=("net_profit_yuan",*COST_METRICS)
    def attempt(identifier,prepare,draw,**metadata):
        record={"id":identifier,**metadata}
        try:
            data,sources=prepare()
            outputs=draw(data)
            record.update(status="complete",outputs=outputs,plotted_values=data,
                          sources=[{"path":str(path),"sha256":_hash(path)} for path in sources])
        except IncompletePlot as exc:
            record.update(status="skipped",reason=str(exc))
        manifest["figures"].append(record)
        finished=sum(item["status"]=="complete" for item in manifest["figures"])
        manifest["complete_figure_count"]=finished
        manifest["status"]="partial" if finished else "pending"
        atomic_json(manifest_path,manifest)
    def comparison(identifier,title,labels,names,**metadata):
        attempt(identifier,
            lambda:(_require_groups(report,names,metrics,seeds),[report_path,plan_path]),
            lambda data:_profit_cost(directory,identifier,title,labels,data,fixture),
            required_groups=names,**metadata)
    if "core" in families:
        for rep in replicates:
            comparison("core_rep"+str(rep)+"_profit_cost","Core methods — training replicate "+str(rep+1),
                ["Zero","Linear","Full ReLU"],["base/zero","base/linear_full_rep"+str(rep),"base/relu_full_rep"+str(rep)],
                family="core",training_replicate=rep)
    if "joint" in families:
        comparison("joint_profit_cost","Route and charging ablation (zero terminal)",
            ["Fixed route\nFixed charging","Online route\nFixed charging","Fixed route\nOnline charging","Joint\noptimization"],
            ["base/joint_ablation_dayahead_baseline","base/joint_ablation_joint_baseline",
             "base/joint_ablation_dayahead_joint","base/joint_ablation_joint_joint"],family="joint")
    if "terminal" in families:
        for rep in replicates:
            comparison("terminal_rep"+str(rep)+"_profit_cost","Terminal-value ablation — training replicate "+str(rep+1),
                ["Zero","Simple\ninventory","Inventory\nReLU","Full ReLU"],
                ["base/zero","base/simple_inventory","base/relu_inventory_only_rep"+str(rep),"base/relu_full_rep"+str(rep)],
                family="terminal",training_replicate=rep)
    if "training" in families:
        for kind,variant in (("linear","full"),("relu","full"),("relu","inventory_only")):
            for rep in replicates:
                identifier="training_"+kind+"_"+variant+"_rep"+str(rep)
                attempt(identifier,
                    lambda k=kind,v=variant,r=rep:_training_data(suite,k,v,r,plan),
                    lambda data,k=kind,v=variant,r=rep,i=identifier:_training_plot(directory,i,data,k,v,r,fixture),
                    family="training",training_replicate=rep,kind=kind,variant=variant)
    if "sensitivity" in families:
        for factor,key in (("horizon","horizon_periods"),("demand","demand_multipliers"),
                           ("battery","battery_slots"),("power","station_power_multipliers")):
            values=plan.get("sensitivity",{}).get(key,[])
            for rep in replicates:
                identifier="sensitivity_"+factor+"_rep"+str(rep)
                names=_sensitivity_groups(report,factor,values,rep)
                def prepare(names=names,values=values):
                    if len(values)<2:raise IncompletePlot("at least two frozen factor values are required")
                    return _require_groups(report,names,[item[0] for item in OPERATION_METRICS],seeds),[report_path,plan_path]
                attempt(identifier,prepare,
                    lambda data,f=factor,v=values,r=rep,i=identifier:_sensitivity_plot(directory,i,f,v,data,r,fixture),
                    family="sensitivity",factor=factor,values=values,training_replicate=rep,required_groups=names)
    finished=sum(item["status"]=="complete" for item in manifest["figures"])
    manifest["complete_figure_count"]=finished
    manifest["skipped_figure_count"]=len(manifest["figures"])-finished
    manifest["status"]="complete" if finished and finished==len(manifest["figures"]) else ("partial" if finished else "pending")
    atomic_json(manifest_path,manifest)
    return str(manifest_path)
