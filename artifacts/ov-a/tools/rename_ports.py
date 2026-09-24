import glob, os, pathlib
from safetensors.numpy import load_file, save_file
src = "/workflow/c2t64"
dst = "/workflow/c2t64-fixed-samples"
pathlib.Path(dst).mkdir(exist_ok=True)
RENAME = {"2630": "2665", "4265": "4300"}
n = 0
for p in glob.glob(src + "/*.safetensors"):
    v = load_file(p)
    out = {RENAME.get(k, k): x for k, x in v.items()}
    save_file(out, os.path.join(dst, os.path.basename(p)))
    n += 1
print("rewrote", n, "samples with the fixed export port names")
