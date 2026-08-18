import json,time,uuid,urllib.request,websocket
S="/tmp/claude-1000/-teamspace-studios-this-studio/84db674b-b682-4753-b5c4-14443a648426/scratchpad"
P=json.load(open(f"{S}/ltx_api.json"))
for v in P.values():
    if v["class_type"]=="RandomNoise": v["inputs"]["noise_seed"]=31337
cid=str(uuid.uuid4()); ws=websocket.WebSocket(); ws.connect(f"ws://127.0.0.1:8188/ws?clientId={cid}")
d=json.dumps({"prompt":P,"client_id":cid}).encode()
pid=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8188/prompt",d,{"Content-Type":"application/json"})))["prompt_id"]
t0=time.time(); last=t0; cur=None; times={}
while True:
    m=ws.recv()
    if not isinstance(m,str): continue
    m=json.loads(m)
    if m["type"]=="executing" and m["data"].get("prompt_id")==pid:
        now=time.time()
        if cur is not None: times[cur]=times.get(cur,0)+now-last
        last=now; cur=m["data"]["node"]
        if cur is None: break
tot=time.time()-t0
print(f"TOTAL {tot:.1f}s")
for k,v in sorted(times.items(), key=lambda x:-x[1]):
    if v>1: print(f"  {v:7.1f}s  {P[k]['class_type']} (node {k})")
