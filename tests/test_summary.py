import gati
import sys
import os

# regressions for testing the parsing/summary feature of gaticc

def get_tbl(l):
    s = ''
    for i in l:
        for j in i:
            s += str(j)
        s += '\n'
    return s

if __name__ == "__main__":
    failed = []
    passed = []
    if len(sys.argv) > 1:
        models_dir = sys.argv[1]
    else: 
        raise SystemExit(f"Insufficient args: Usage: {sys.argv[0]} <path to models dir>")
    print(f"Using onnx models from: {models_dir}")
    gati.set_keep_quiet(True)
    for file in os.listdir(models_dir):
        print(f"File: {file}")
        try:
          gati.summary(os.path.join(models_dir, file))
          ret = 0
        except RuntimeError as e:
          ret = 1

        if ret != 0:
            failed.append(file)
        else:
            passed.append(file)
    resfile_name = os.path.basename(sys.argv[0]).split('.')[0] + ".results.txt"
    with open(f"{resfile_name}", "w") as f:
        f.write(f"Failed Summaries\n")
        f.write(get_tbl(failed))
        f.write(f"Passed Summaries\n")
        f.write(get_tbl(passed))
        f.write(f"Results: {len(passed)}/{len(passed)+len(failed)}")
    print(f"Results written to: {resfile_name}\n")
