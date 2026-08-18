import json,time,uuid,urllib.request,websocket,copy
S="/tmp/claude-1000/-teamspace-studios-this-studio/84db674b-b682-4753-b5c4-14443a648426/scratchpad"
BASE=json.load(open(f"{S}/ltx_api.json"))
models=[("int8-convrot 21.5GB","ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
        ("mix4x8 17GB","LTX25-distilled-DiT-comfy-mix4x8-17GB.safetensors")]
for name,fn in models:
    P=copy.deepcopy(BASE)
    P["384"]["inputs"]["unet_name"]=fn
    P["376"]["inputs"]["value"]="The camera slowly drifts as light moves across the scene"
    for v in P.values():
        if v["class_type"]=="RandomNoise": v["inputs"]["noise_seed"]=8888
    cid=str(uuid.uuid4()); ws=websocket.WebSocket(); ws.connect(f"ws://127.0.0.1:8188/ws?clientId={cid}")
    d=json.dumps({"prompt":P,"client_id":cid}).encode()
    try:
        pid=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8188/prompt",d,{"Content-Type":"application/json"})))["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"{name}: REJECTED {json.dumps(json.load(e).get('node_errors'))[:200]}"); continue
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
        h=json.load(urllib.request.urlopen("http://127.0.0.1:8188/history/"+pid,timeout=10))
        st=h.get(pid,{}).get("status",{}); ok=st.get("status_str")=="success"
        outs=json.dumps(h.get(pid,{}).get("outputs",{}))[:120]
        print(f"{name:<24} {'OK ' if ok else 'FAILED'} TOTAL {tot:6.1f}s | pass1 {times.get('344',0):5.1f}s | pass2 {times.get('368',0):5.1f}s | decode {times.get('374',0):5.1f}s | {outs if ok else st.get('status_str')}")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__} after {time.time()-t0:.1f}s")
    ws.close()
