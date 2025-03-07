import gati
import sys
import os
# regressions for testing the compile feature of gaticc
arch_list = [
        { "ramsize": 512, "sa-arch": "9,4,4", "vasize": 32, "accbuf-size": 4096, "fcbuf-size": 32768 },
        { "ramsize": 512, "sa-arch": "9,8,8", "vasize": 32, "accbuf-size": 4096, "fcbuf-size": 32768 },
        ]

def get_tbl(l):
    s = ''
    for i in l:
        for j in i:
            s += str(j) + ' '
        s += '\n'
    return s

if __name__ == "__main__":
    failed_compilations = []
    passed_compilations = []
    if len(sys.argv) > 1:
        models_dir = sys.argv[1]
    else: 
        raise SystemExit(f"Insufficient args: Usage: {sys.argv[0]} <path to models dir>")
    print(f"Using onnx models from: {models_dir}")
    gati.set_keep_quiet(True)
    for arch in arch_list:
        gati.set_arch(arch)
        print(f"Testing for arch: {arch}")
        for file in os.listdir(models_dir):
            print(f"File: {file}")
            ret = gati.compile(os.path.join(models_dir, file), "/tmp/model.gml")
            if ret != 0:
                failed_compilations.append((arch, file))
            else:
                passed_compilations.append((arch, file))
    resfile_name = os.path.basename(sys.argv[0]).split('.')[0] + ".results.txt"
    with open(f"{resfile_name}", "w") as f:
        f.write(f"Failed Compilations\n")
        f.write(get_tbl(failed_compilations))
        f.write(f"Passed Compilation\n")
        f.write(get_tbl(passed_compilations))
        f.write(f"Results: {len(passed_compilations)}/{len(passed_compilations)+len(failed_compilations)}")
    print(f"Results written to: {resfile_name}\n")
