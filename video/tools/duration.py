import json,time,uuid,urllib.request,websocket,copy
S="/tmp/claude-1000/-teamspace-studios-this-studio/84db674b-b682-4753-b5c4-14443a648426/scratchpad"
BASE=json.load(open(f"{S}/ltx_api.json"))
for secs in [5,10]:
    P=copy.deepcopy(BASE)
    P["362"]["inputs"]["value"]=secs          # duration primitive
    P["376"]["inputs"]["value"]=f"The camera slowly drifts as light moves across the scene"
    for v in P.values():
        if v["class_type"]=="RandomNoise": v["inputs"]["noise_seed"]=2200+secs
    cid=str(uuid.uuid4()); ws=websocket.WebSocket(); ws.connect(f"ws://127.0.0.1:8188/ws?clientId={cid}")
    d=json.dumps({"prompt":P,"client_id":cid}).encode()
    try:
        pid=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8188/prompt",d,{"Content-Type":"application/json"})))["prompt_id"]
    except Exception as e: print(f"{secs}s: queue failed {e}"); continue
    t0=time.time(); last=t0; cur=None; times={}
    try:
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
        print(f"{secs}s clip: TOTAL {tot:6.1f}s -> {tot/secs:5.1f}s per second of video | pass1 {times.get('344',0):.1f} pass2 {times.get('368',0):.1f} decode {times.get('374',0):.1f}")
    except Exception as e: print(f"{secs}s: failed {type(e).__name__} after {time.time()-t0:.1f}s")
    ws.close()
