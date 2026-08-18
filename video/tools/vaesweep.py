import json,time,uuid,urllib.request,websocket,copy
S="/tmp/claude-1000/-teamspace-studios-this-studio/84db674b-b682-4753-b5c4-14443a648426/scratchpad"
BASE=json.load(open(f"{S}/ltx_api.json"))
cfgs=[("baseline 512/64/64/16",dict(tile_size=512,overlap=64,temporal_size=64,temporal_overlap=16)),
      ("big tiles 1024/64/128/8",dict(tile_size=1024,overlap=64,temporal_size=128,temporal_overlap=8)),
      ("huge 2048/32/256/8",dict(tile_size=2048,overlap=32,temporal_size=256,temporal_overlap=8))]
for name,kw in cfgs:
    P=copy.deepcopy(BASE)
    P["374"]["inputs"].update(kw)
    for v in P.values():
        if v["class_type"]=="RandomNoise": v["inputs"]["noise_seed"]=999
    cid=str(uuid.uuid4()); ws=websocket.WebSocket(); ws.connect(f"ws://127.0.0.1:8188/ws?clientId={cid}")
    d=json.dumps({"prompt":P,"client_id":cid}).encode()
    try:
        pid=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8188/prompt",d,{"Content-Type":"application/json"})))["prompt_id"]
    except Exception as e:
        print(f"{name}: QUEUE FAILED {e}"); continue
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
        print(f"{name}: TOTAL {time.time()-t0:.1f}s | VAEDecodeTiled {times.get('374',0):.1f}s")
    except Exception as e:
        print(f"{name}: FAILED mid-run ({type(e).__name__}) after {time.time()-t0:.1f}s")
    ws.close()
