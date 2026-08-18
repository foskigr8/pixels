import json,time,uuid,urllib.request,websocket,copy
S="/tmp/claude-1000/-teamspace-studios-this-studio/84db674b-b682-4753-b5c4-14443a648426/scratchpad"
BASE=json.load(open(f"{S}/ltx_api.json"))
prompts=["The camera slowly pushes in as light shifts across the scene",
         "Wind moves through the frame as the camera drifts left"]
for i,txt in enumerate(prompts):
    P=copy.deepcopy(BASE); P["376"]["inputs"]["value"]=txt
    for v in P.values():
        if v["class_type"]=="RandomNoise": v["inputs"]["noise_seed"]=5150+i
    for attempt in range(5):
        try:
            cid=str(uuid.uuid4()); ws=websocket.WebSocket(); ws.connect(f"ws://127.0.0.1:8188/ws?clientId={cid}")
            d=json.dumps({"prompt":P,"client_id":cid}).encode()
            pid=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8188/prompt",d,{"Content-Type":"application/json"}),timeout=20))["prompt_id"]
            break
        except Exception: time.sleep(15)
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
    h=json.load(urllib.request.urlopen("http://127.0.0.1:8188/history/"+pid,timeout=10))
    st=h.get(pid,{}).get("status",{}); ok=st.get("status_str")=="success"
    f=json.dumps(h.get(pid,{}).get("outputs",{}))[:90] if ok else st.get("status_str")
    print(f"clip {i+1}: {'OK ' if ok else 'FAILED'} TOTAL {tot:6.1f}s | pass1 {times.get('344',0):5.1f} pass2 {times.get('368',0):5.1f} decode {times.get('374',0):5.1f} save {times.get('9002',0):4.1f} | {f}")
    ws.close()
