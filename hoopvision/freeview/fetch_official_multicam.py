from __future__ import annotations

import argparse, hashlib, html as htmlmod, json, re, subprocess, urllib.parse, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import cv2, numpy as np

UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/153 Safari/537.36'
HEADERS={'User-Agent':UA,'Referer':'https://clips.nba.com/','Accept':'*/*'}


def get(url,timeout=45):
    req=urllib.request.Request(url,headers=HEADERS)
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()

def signed_join(base,child):
    joined=urllib.parse.urljoin(base,child); bp=urllib.parse.urlsplit(base); cp=urllib.parse.urlsplit(joined)
    if bp.query and not cp.query: joined=urllib.parse.urlunsplit(cp._replace(query=bp.query))
    return joined

def resolve_options(game,event):
    page=f'https://clips.nba.com/?gameNo={game}&eventNum={event}&source=grs'; txt=get(page).decode('utf-8','replace');out={}
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',txt,flags=re.I|re.S):
        url=htmlmod.unescape(m.group(1).strip());label=re.sub(r'<[^>]+>','',htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in url.lower() and 'lrmedia.nba.com' in url.lower():out[label]=url
    return out

def parse_media(url,depth=0):
    if depth>3:raise RuntimeError('playlist nesting exceeded')
    text=get(url).decode('utf-8','replace'); lines=[x.strip() for x in text.splitlines() if x.strip()];variants=[]
    for i,line in enumerate(lines):
        if not line.startswith('#EXT-X-STREAM-INF:'):continue
        bw=int(re.search(r'BANDWIDTH=(\d+)',line).group(1)) if re.search(r'BANDWIDTH=(\d+)',line) else 0
        for child in lines[i+1:]:
            if not child.startswith('#'):variants.append((bw,signed_join(url,child)));break
    if variants:return parse_media(max(variants,key=lambda x:x[0])[1],depth+1)
    init=None;segs=[]
    for line in lines:
        if line.startswith('#EXT-X-MAP:'):
            m=re.search(r'URI="([^"]+)"',line)
            if m:init=signed_join(url,m.group(1))
        elif not line.startswith('#'):segs.append(signed_join(url,line))
    if not segs:raise RuntimeError('no media segments')
    return init,segs

def download_hls(url,out):
    init,segs=parse_media(url); urls=([init] if init else [])+segs;parts=[None]*len(urls)
    def one(i,u):return i,get(u)
    with ThreadPoolExecutor(max_workers=min(8,len(urls))) as pool:
        fut=[pool.submit(one,i,u) for i,u in enumerate(urls)]
        for f in as_completed(fut):i,b=f.result();parts[i]=b
    raw=out.with_suffix('.hls.part')
    with raw.open('wb') as h:
        for b in parts:
            if b is None:raise RuntimeError('missing HLS segment')
            h.write(b)
    subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-i',str(raw),'-map','0:v:0','-map','0:a:0?','-c','copy','-movflags','+faststart',str(out)],check=True);raw.unlink(missing_ok=True)

def probe(path):
    q=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate:format=duration','-of','json',str(path)],text=True));s=q['streams'][0];fr=s.get('avg_frame_rate','0/1');n,d=fr.split('/');fps=float(n)/float(d) if float(d) else 0
    return {'width':int(s['width']),'height':int(s['height']),'fps':fps,'duration_s':float(q['format']['duration'])}

def sha256(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def frame_metrics(path):
    im=cv2.imread(str(path));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var());edges=cv2.Canny(gray,65,155);lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=65,minLineLength=50,maxLineGap=9)
    lengths=[];floor=0
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            ln=math.hypot(float(x2-x1),float(y2-y1));lengths.append(ln);floor+=int(max(y1,y2)>=285 and ln>=70)
    upper=gray[:390];_,bw=cv2.threshold(upper,178,255,cv2.THRESH_BINARY);bw=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));cnts,_=cv2.findContours(bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);rect=0
    for c in cnts:
        peri=cv2.arcLength(c,True)
        if peri<30:continue
        a=cv2.approxPolyDP(c,.025*peri,True)
        if len(a)!=4 or not cv2.isContourConvex(a):continue
        x,y,w,h=cv2.boundingRect(a);ar=w/max(h,1);fill=cv2.contourArea(c)/max(float(w*h),1)
        if 24<=w<=230 and 14<=h<=165 and 1.05<=ar<=2.8 and fill>=.15:rect+=1
    long=sum(x>=110 for x in lengths);edge=float(np.mean(edges>0));score=2.2*math.log1p(max(sharp,0))+.09*min(long,40)+.07*min(floor,40)+.22*min(rect,10)+5*min(edge,.16)
    return score

def select_geometry_frame(video,out,n=9):
    """Reproduce the immutable v101/v108 event-540 sample identity.

    The accepted v108 source-native target-line observations were measured on
    v101 sample ``f00``.  v101 sampled nine frames at linspace(.18, .82, 9),
    therefore changing to the currently "best" sharp frame would silently
    attach those measurements to different pixels.  Extract the original f00
    deterministically and score the remaining samples for diagnostics only.
    """
    q=probe(video);dur=q['duration_s'];rows=[]
    for i,frac in enumerate(np.linspace(.18,.82,n)):
        t=max(.05,min(dur-.05,dur*float(frac)));p=out/f'rs540_f{i:02d}.png'
        subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-ss',f'{t:.5f}','-i',str(video),'-frames:v','1',str(p)],check=True)
        rows.append({'sample':i,'fraction':float(frac),'time_s':t,'score':float(frame_metrics(p)),'path':p})
    chosen=rows[0]
    dst=out/'Right_Slash_event540.png';dst.write_bytes(chosen['path'].read_bytes())
    ranked=sorted(rows,key=lambda x:x['score'],reverse=True)
    return dst,{
        'selection_policy':'locked_v101_f00_fraction_0.18_to_match_v108_source_native_observations',
        'source_sample':chosen['path'].name,'fraction':chosen['fraction'],'time_s':chosen['time_s'],
        'score':chosen['score'],'diagnostic_best_sample':ranked[0]['path'].name,
        'diagnostic_best_score':ranked[0]['score'],
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    labels=['Broadcast','In Arena','Left Slash','Right Slash','Left Above Rim','Right Above Rim'];opts=resolve_options('0022500301',489);rows=[];clips=a.out/'event489';clips.mkdir(exist_ok=True)
    for i,l in enumerate(labels,1):
        if l not in opts:raise RuntimeError(f'missing official feed {l}')
        p=clips/f'{i:02d}_{"_".join(l.split())}.mp4';download_hls(opts[l],p);q=probe(p)
        if (q['width'],q['height'])!=(960,540):raise RuntimeError(f'{l} unexpected resolution {q}')
        rows.append({'label':l,'path':str(p.relative_to(a.out)),**q,'sha256':sha256(p)})
    manifest={'schema_version':1,'game_id':'0022500301','event_num':489,'feeds':rows};(a.out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    o540=resolve_options('0022500301',540);p540=a.out/'right_slash_event540.mp4';download_hls(o540['Right Slash'],p540);selected,diag=select_geometry_frame(p540,a.out)
    (a.out/'event540_selection.json').write_text(json.dumps({'frame':selected.name,**diag},indent=2)+'\n')
    print(json.dumps({'event489_feeds':len(rows),'event540_selected':str(selected),'diag':diag},indent=2))
if __name__=='__main__':main()