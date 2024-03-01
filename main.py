# open csv
import csv
import subprocess

from rich.console import Console

from env import CONFS_PATH, MODELS_PATH
from modules.utils.rich import makeStrTable

DELETE = False


with open("queue.csv", newline="") as f:
    reader = csv.reader(f)
    queue = list(reader)
table = makeStrTable(["Project", "Act", "Version"], queue)

console = Console()
console.print(table)


# queue = [["task1.py", "arg"], ["task2.py", "hello.txt"], ["task3.py", "task3"]]

for i, (project, act, conf) in enumerate(queue):
    program = f"{MODELS_PATH}/{project}/{act}/main.py"
    # cfg = f"{CONFS_PATH}/{project}/{act}/{conf}.yaml"
    print(program)
    subprocess.run(["python", program, project, act, conf])

    # * DROP from queue
    if DELETE:
        with open("queue.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(queue[i + 1 :])
