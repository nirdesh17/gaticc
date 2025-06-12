import os, sys, argparse
import gati

def get_tbl(rows): return "\n".join(str(r) for r in rows) + "\n"

def main():
  p = argparse.ArgumentParser()
  p.add_argument('-m', '--models', required=True)
  p.add_argument('-o', '--output', default="test_summary.results.txt")
  a = p.parse_args()

  if not os.path.isdir(a.models): sys.exit("bad dir")
  gati.set_keep_quiet(True)

  fail, ok = [], []

  for f in os.listdir(a.models):
    path = os.path.join(a.models, f)
    try: gati.summary(path); ok.append(f)
    except: fail.append(f)

  txt  = "Failed Summaries\n" + get_tbl(fail)
  txt += "Passed Summaries\n" + get_tbl(ok)
  txt += f"Results: {len(ok)}/{len(ok)+len(fail)}\n"

  open(a.output, "w").write(txt)
  print(txt)

if __name__ == "__main__": main()
