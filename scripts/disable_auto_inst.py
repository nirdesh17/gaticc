#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as ET

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} project.xml")
    sys.exit(1)

xml_path = sys.argv[1]

# Efinix namespace
NS = {
    "efx": "http://www.efinixinc.com/enf_proj"
}

ET.register_namespace("efx", NS["efx"])

tree = ET.parse(xml_path)
root = tree.getroot()

changed = False

# Find debugger -> param(name="auto_instantiation")
for param in root.findall(".//efx:debugger/efx:param", NS):
    if param.attrib.get("name") == "auto_instantiation":
        val = param.attrib.get("value")
        print(f"current auto_instantiation = {val}")

        if val == "on":
            param.set("value", "off")
            changed = True
            print("auto_instantiation turned OFF")
        else:
            print("already OFF")

if changed:
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    print("project file updated.")
else:
    print("no change needed.")
