import pprint
import sys
from sys import argv

sys.path.append("/app")

from modules.utils.conf import get_cfg, syncHash

print("CIFAR_10/00_basic")
cfg = get_cfg(argv)
print(cfg)
hash = syncHash()
