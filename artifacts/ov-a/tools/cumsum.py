import collections
import xml.etree.ElementTree as ET

XML = "/root/w8a8/ir/pi05_droid_dynamic_w8a8.xml"
root = ET.parse(XML).getroot()
layers = {l.get("id"): l for l in root.find("layers")}
prod = collections.defaultdict(list)
cons = collections.defaultdict(list)
for e in root.find("edges"):
    prod[e.get("to-layer")].append(e.get("from-layer"))
    cons[e.get("from-layer")].append(e.get("to-layer"))


def desc(i, depth=0):
    l = layers[i]
    out = l.find("output/port")
    shape = "x".join(d.text for d in out.findall("dim")) if out is not None else "?"
    prec = out.get("precision") if out is not None else "?"
    return "%s%s [%s %s] %s" % ("  " * depth, l.get("type"), prec, shape, l.get("name")[-70:])


cs = [i for i, l in layers.items() if l.get("type") == "CumSum"]
print("CumSum count", len(cs))
for i in cs:
    print(desc(i))
    for p in prod[i]:
        print(desc(p, 1))
        for pp in prod[p]:
            print(desc(pp, 2))
            for ppp in prod[pp]:
                print(desc(ppp, 3))
    print("   consumers:")
    for c in cons[i]:
        print(desc(c, 2))
    print()
