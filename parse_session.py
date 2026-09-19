import json

path = r'C:\Users\wo\.codex\sessions\2026\09\18\rollout-2026-09-18T13-54-42-01a0b314-d656-73c3-aa3a-28a051eec850.jsonl'

def item_text(content):
    parts = []
    for c in content or []:
        if isinstance(c, dict):
            t = c.get('text')
            if t:
                parts.append(t)
    return '\n'.join(parts)

out = []
for line in open(path, encoding='utf-8'):
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if obj.get('type') != 'response_item':
        continue
    p = obj.get('payload', {})
    pt = p.get('type')
    ts = obj.get('timestamp', '')[11:19]
    if pt == 'message':
        role = p.get('role')
        if role == 'developer':
            continue
        txt = item_text(p.get('content'))
        if not txt.strip():
            continue
        out.append(('===== %s [%s] =====' % (role.upper(), ts), txt))
    elif pt == 'function_call':
        name = p.get('name')
        args = (p.get('arguments') or '')[:300]
        out.append(('----- tool_call %s [%s] -----' % (name, ts), args))

with open(r'D:\Users\wo\文档\vscode files\cleanrl_BSS\session_0918_transcript.txt', 'w', encoding='utf-8') as f:
    for head, body in out:
        f.write(head + '\n')
        f.write(body + '\n\n')

print('entries:', len(out))
for head, body in out:
    first = body[:100].replace('\n', ' ')
    print(head, '|', first)
