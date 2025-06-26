import os, sys, json, subprocess, argparse

mut = [
  'mnist_6_28_int8.onnx', 'mnist_int8_pad2.onnx', 'mnist_int8_stride2.onnx',
  'mnist_int8_stride3.onnx', 'mnistpad1_6_28_int8.onnx', 'imagenet_vgg_16_224_int8.onnx',
  'cifar10_vgg11.onnx', 'cifar10_vgg16.onnx', 'cifar10_vgg19.onnx',
  'mnist_int8_k1x7_nomaxpool.onnx', 'mnist_int8_stride2_pad0.onnx',
  'mnist_int8_stride2_pad2.onnx', 'mnist_average_pool_int8.onnx', 'mnist_int8_pad4343.onnx'
]

def run_all(args):
  models = [f for f in os.listdir(args.models[0]) if f in mut] if len(args.models)==1 and os.path.isdir(args.models[0]) else [os.path.basename(f) for f in args.models]
  models_dir = args.models[0] if os.path.isdir(args.models[0]) else os.path.dirname(args.models[0]) or '.'
  failed, matches = [], []
  for m in models:
    print(f"Running: {os.path.join(models_dir, m)}")
    res = f"/tmp/{m}.res.json"
    try:
      subprocess.run([sys.executable, sys.argv[0], '--run-one', '-m', os.path.join(models_dir, m), '-a', args.arch, '-b', args.bitstream, '-l', args.hostname, '--out', res], check=True)
      with open(res) as f: out = json.load(f)
      (matches if out.get("status") == "ok" else failed).append((m, out.get("match", 0)))
    except KeyboardInterrupt:
      print(f"Skipped: {m}")
      failed.append((m, 0))
    except: failed.append((m, 0))
  with open("test_run.results.txt", "w") as f: f.write(fmt(failed, matches))
  print("\n" + fmt(failed, matches))

def run_one(args):
  import numpy as np, gati
  gati.set_keep_quiet(True)
  gati.set_arch(config={"sa-arch": args.arch})
  if args.hostname != 'localhost': gati.set_remote(args.hostname)
  onnx = args.models[0]
  gml = "/tmp/model.gml"
  res = {"status": "fail"}
  try:
    if gati.compile(onnx, gml) != 0: raise Exception("compile failed")
    gati.flash(args.bitstream)
    if "mnist" in onnx: data, lbl = np.load("mnist_10.npy"), "mnist_10_labels.txt"
    elif "imagenet" in onnx: data, lbl = np.load("imagenet_10.npy"), "imagenet_10_labels.txt"
    elif "cifar" in onnx: data, lbl = np.load("cifar_10.npy"), "cifar_10_labels.txt"
    else: raise Exception("unknown dataset")
    post = lambda arr: np.argmax(np.squeeze(np.stack([i[1] for i in arr]), 1), -1)
    out = post(gati.run(onnx, gml, data))
    res = {"status": "ok", "match": gati.match(lbl, out)}
  except Exception as e:
    res["error"] = str(e)
  if args.out: json.dump(res, open(args.out, "w"))

def fmt(failed, matches):
  out = "=== Test Results ===\n\nFailed:\n"
  out += "".join(f"  - {m}\n" for m, _ in failed) if failed else "  None\n"
  out += "\nMatches:\n"
  out += "  Model".ljust(32) + "Match %\n" + "  " + "-"*45 + "\n"
  out += "".join(f"  {m.ljust(30)}{p:.2f}%\n" for m, p in matches) if matches else "  None\n"
  return out

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('-m', '--models', nargs='+', required=True)
  p.add_argument('-a', '--arch', default='9,4,4')
  p.add_argument('-b', '--bitstream', required=True)
  p.add_argument('-l', '--hostname', required=True)
  p.add_argument('--run-one', action='store_true')
  p.add_argument('--out')
  args = p.parse_args()
  run_one(args) if args.run_one else run_all(args)
