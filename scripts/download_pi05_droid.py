import concurrent.futures, json, pathlib, urllib.parse, urllib.request
bucket="openpi-assets"; prefix="checkpoints/pi05_droid/"; dest=pathlib.Path.home()/"robo_run/checkpoints/pi05_droid"
data=json.load(urllib.request.urlopen(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={prefix}&maxResults=1000"))
items=data.get("items",[])
print(f"{len(items)} objects, {sum(int(x["size"])/2**30 for x in items):.3f} GiB",flush=True)
def fetch(item):
    rel=item["name"][len(prefix):]; out=dest/rel; out.parent.mkdir(parents=True,exist_ok=True)
    size=int(item["size"])
    if out.exists() and out.stat().st_size==size: return rel+" SKIP"
    url="https://storage.googleapis.com/"+bucket+"/"+urllib.parse.quote(item["name"],safe="/")
    tmp=out.with_suffix(out.suffix+".part")
    urllib.request.urlretrieve(url,tmp); tmp.replace(out)
    return rel+" DONE"
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for result in pool.map(fetch,items): print(result,flush=True)
print("DONE",flush=True)
