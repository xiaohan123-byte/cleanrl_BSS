import json
from pathlib import Path

path = Path(r'C:\Users\wo\.codex\sessions\2026\09\18\rollout-2026-09-18T11-13-49-01a0b281-89b9-7981-a40c-0e0e42d86a45.jsonl')
out_path = Path(r'D:\Users\wo\文档\vscode files\cleanrl_BSS') / 'session_0918_transcript_readable.txt'

def item_text(content):
    parts = []
    for c in content or []:
        if isinstance(c, dict):
            t = c.get('text')
            if t:
                parts.append(t)
    return '\n'.join(parts)

out = []
for line in path.open('r', encoding='utf-8'):
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if obj.get('type') != 'response_item':
        continue
    p = obj.get('payload', {})
    pt = p.get('type')
    ts = obj.get('timestamp', '')
    if pt == 'message':
        role = p.get('role')
        if role == 'developer':
            continue
        txt = item_text(p.get('content'))
        if not txt.strip():
            continue
        out.append(('===== {} [{}] ====='.format(role.upper(), ts), txt))
    elif pt == 'function_call':
        name = p.get('name')
        args = (p.get('arguments') or '')[:500]
        out.append(('----- tool_call {} [{}] -----'.format(name, ts), args))
    elif pt == 'function_call_output':
        output = str(p.get('output') or '')[:500]
        out.append(('----- tool_output [{}] -----'.format(ts), output))

with out_path.open('w', encoding='utf-8') as f:
    for head, body in out:
        f.write(head + '\n')
        f.write(body + '\n\n')

print(str(out_path))
print('entries:', len(out))
