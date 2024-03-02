# open csv
import csv
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import cast

from hydra import compose, initialize
from rich.console import Console

from confs.conf import GlobalConfig, get_act_cfg
from env import MODELS_PATH, REL_CONFS_PATH
from modules.utils.rich import makeStrTable

with initialize(version_base=None, config_path=REL_CONFS_PATH):
    cfg = cast(GlobalConfig, compose(config_name="env"))

console = Console()

with open("queue.csv", newline="") as f:
    reader = csv.reader(f)
    queue = list(reader)
outputs = []

table = makeStrTable(["Project", "Act", "Version"], queue)
console.print(table)

for i, (project, act, conf) in enumerate(queue):
    program = f"{MODELS_PATH}/{project}/{act}/main.py"
    # cfg = f"{CONFS_PATH}/{project}/{act}/{conf}.yaml"
    start_time = time.time()
    process = subprocess.Popen(
        ["python", program, project, act, conf], stdout=subprocess.PIPE
    )

    last_line = None
    for line in process.stdout:  # type: ignore
        last_line = line.rstrip()
        if cfg.runner.pipe_output:
            print(last_line.decode())
    process.wait()
    if last_line is None:
        vis_metrics = ["", "None"]
        results = ["", "", "ERROR"]
    else:
        results = last_line.decode().split(",")
        if len(results) != 3 or results[-1] != "SUCCESS":
            print("==ERROR: Unexpected output==", file=sys.stderr)
            vis_metrics = ["", ""]
        else:
            metrics = results[:-1]
            metrics = [float(m) for m in metrics]
            vis_metrics = [f"{m:.3f}" for m in metrics]

    delta = time.time() - start_time
    delta = timedelta(seconds=int(delta)).__str__()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    act_cfg = get_act_cfg(project, act)
    outputs.append(
        [
            project,
            f"{act}_{act_cfg.name}",
            conf,
            date,
            delta,
            vis_metrics[0],
            vis_metrics[1],
            results[-1],
        ]
    )

    # DROP from queue
    if cfg.runner.delete_queue:
        with open("queue.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(queue[i + 1 :])
    # Log
    with open("log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(outputs)

    # Run summary
    table = makeStrTable(
        ["Project", "Act", "Version", "Date", "Time", "Metric1", "Metric2", "Status"],
        [outputs[-1]],
    )
    console.print(table)

# Run summary
table = makeStrTable(
    ["Project", "Act", "Version", "Date", "Time", "Metric1", "Metric2", "Status"],
    outputs,
)
console.print(table)
