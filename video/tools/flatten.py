import json,urllib.request,sys
OI=json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info"))
d=json.load(open("/teamspace/studios/this_studio/ComfyUI/user/default/workflows/video_ltx2_5_i2v.json"))
sg=d["definitions"]["subgraphs"][0]
nodes={n["id"]:n for n in sg["nodes"]}
# link map: (target_id,target_slot) -> [origin_id, origin_slot]
lm={(l["target_id"],l["target_slot"]):(l["origin_id"],l["origin_slot"]) for l in sg["links"]}

def widget_names(t, connected):
    """input names that take widget values, in definition order, minus connected ones"""
    spec=OI[t]["input"]; out=[]
    for grp in ("required","optional"):
        for name,v in spec.get(grp,{}).items():
            if name in connected: continue
            typ=v[0]
            if isinstance(typ,list) or typ in ("INT","FLOAT","STRING","BOOLEAN","COMBO"):
                out.append((name,typ))
    return out

api={}
for nid,n in nodes.items():
    t=n["type"]
    if t in ("Note","MarkdownNote","PreviewAny"): continue
    if t not in OI: print("SKIP unknown",t); continue
    slots=n.get("inputs",[]) or []
    connected={}
    for i,s in enumerate(slots):
        if (nid,i) in lm:
            o,os_=lm[(nid,i)]
            if o!=-10: connected[s["name"]]=[str(o),os_]
    inputs=dict(connected)
    wv=list(n.get("widgets_values") or [])
    for name,typ in widget_names(t,set(connected)):
        if not wv: break
        inputs[name]=wv.pop(0)
        # seed-style nodes carry an extra control value
        if name in ("noise_seed","seed") and wv and wv[0] in ("fixed","randomize","increment","decrement"):
            wv.pop(0)
        if typ=="INT" and wv and wv[0] in ("fixed","randomize","increment","decrement"):
            wv.pop(0)
    api[str(nid)]={"class_type":t,"inputs":inputs}

# ---- boundary wiring (subgraph inputs) ----
api["9001"]={"class_type":"LoadImage","inputs":{"image":sys.argv[1] if len(sys.argv)>1 else "neon_cyborg_portrait.png"}}
api["351"]["inputs"]["input"]=["9001",0]          # first_frame -> resize node
api["383"]["inputs"]["value"]=False                # prompt_enhance OFF (skips 10GB gemma-e2b)
api["372"]["inputs"]["value"]=1280                 # width
api["360"]["inputs"]["value"]=736                  # height (16:9, multiple of 32)
api["362"]["inputs"]["value"]=5                    # duration (s)
api["361"]["inputs"]["value"]=24                   # fps
# video out
vid=[k for k,v in api.items() if v["class_type"]=="CreateVideo"][0]
api["9002"]={"class_type":"SaveVideo","inputs":{"video":[vid,0],"filename_prefix":"video/ltx_bench","format":"auto","codec":"auto"}}
# drop the enhancer branch's dangling preview
api.pop([k for k,v in api.items() if v["class_type"]=="PreviewAny"][0],None) if any(v["class_type"]=="PreviewAny" for v in api.values()) else None
json.dump(api,open("/tmp/claude-1000/-teamspace-studios-this-studio/84db674b-b682-4753-b5c4-14443a648426/scratchpad/ltx_api.json","w"),indent=1)
print("nodes in api prompt:",len(api))
