import urllib.request
import re

html = urllib.request.urlopen('http://localhost:8000/').read().decode('utf-8')
scripts = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)
print(f'Found {len(scripts)} inline scripts in HTML')

dom_ids = set(re.findall(r'id=["\'](.*?)["\']', html))
print(f'Found {len(dom_ids)} unique DOM IDs in HTML')

used_ids = set()
for s in scripts:
    for match in re.findall(r'document\.getElementById\(["\'](.*?)["\']\)', s):
        used_ids.add(match)

print(f'Scripts reference {len(used_ids)} element IDs')
missing = [i for i in used_ids if i not in dom_ids]
print('Missing DOM IDs referenced by JS:', missing)

# Check all event handlers in HTML
handlers = re.findall(r'on\w+=["\'](.*?)\(.*?\)', html)
print(f'Found {len(handlers)} inline event handlers: {set(handlers)}')
