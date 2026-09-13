"""Background RSS/swap sampler. Usage: from memlog import start; start(tag)."""

import atexit
import threading
import time

import psutil

_peak = {"rss": 0, "swap": 0}


def start(tag, interval=5.0, logfile=None):
  proc = psutil.Process()
  f = open(logfile, "a") if logfile else None

  def loop():
    while True:
      rss = proc.memory_info().rss
      sw = psutil.swap_memory().used
      _peak["rss"] = max(_peak["rss"], rss)
      _peak["swap"] = max(_peak["swap"], sw)
      line = (
          f"[memlog {tag}] t={time.strftime('%H:%M:%S')} "
          f"rss={rss/2**30:.2f}GiB swap_used={sw/2**30:.2f}GiB "
          f"avail={psutil.virtual_memory().available/2**30:.2f}GiB"
      )
      print(line, flush=True)
      if f:
        f.write(line + "\n")
        f.flush()
      time.sleep(interval)

  t = threading.Thread(target=loop, daemon=True)
  t.start()

  def report():
    print(
        f"[memlog {tag}] PEAK rss={_peak['rss']/2**30:.2f}GiB "
        f"swap_used={_peak['swap']/2**30:.2f}GiB",
        flush=True,
    )

  atexit.register(report)
  return t
