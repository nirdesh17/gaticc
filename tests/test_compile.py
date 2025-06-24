import os, sys, argparse
import gati

archs = [
  {"ramsize": 512, "sa-arch": "9,4,4",  "vasize": 32, "accbuf-size": 4096, "fcbuf-size": 32768, "im2colbuf-size": 1024},
  {"ramsize": 512, "sa-arch": "9,8,8",  "vasize": 32, "accbuf-size": 4096, "fcbuf-size": 32768, "im2colbuf-size": 1024},
  {"ramsize": 512, "sa-arch": "16,1,16","vasize": 32, "accbuf-size": 4096, "fcbuf-size": 32768, "im2colbuf-size": 1024},
]

def fmt(arch, name): return f'SA={arch["sa-arch"]:<8}  Model={name}'
def tbl(title, rows): return "\n".join([title, "-"*len(title)] + ([fmt(a,n) for a,n in rows] if rows else ["None"])) + "\n"

def main():
  p = argparse.ArgumentParser()
  p.add_argument('-m', '--models', nargs='+', required=True)
  p.add_argument('-o', '--output', default="test_compile.results.txt")
  a = p.parse_args()
  models = a.models
  if len(models) == 1 and os.path.isdir(models[0]):
    models = [os.path.join(models[0], f) for f in os.listdir(models[0])]
  out = a.output

  gati.set_keep_quiet(True)
  fail, ok = [], []

  for arch in archs:
    gati.set_arch(config=arch)
    for f in models:
      try: gati.compile(f, "/tmp/model.gml"); ok.append((arch, f))
      except: fail.append((arch, f))

  txt = tbl("Failed", fail) + tbl("Passed", ok) + f"Results: {len(ok)} passed / {len(ok)+len(fail)} total\n"
  open(out, "w").write(txt)
  print(txt)

if __name__ == "__main__": main()
