from __future__ import annotations

"""Synchronize and render the source-grounded Adams-Jazz freeze-swivel replay.

RGB policy: every rendered pixel is copied from an official NBA source frame.
Learned models may propose pose/depth, but never generate appearance.
"""

import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from ultralytics import YOLO
from moge.model.v2 import MoGeModel

from hoopvision.freeview.camera_solver import RIM_CENTER_CM

W,H=960,540


def read_json(p): return json.loads(Path(p).read_text())

def safe(label): return '_'.join(label.split())


def manifest_clip(manifest_path: Path,label:str)->Path:
    m=read_json(manifest_path); rows=[r for r in m['feeds'] if r['label']==label]
    if len(rows)!=1: raise RuntimeError(f'manifest label {label}: {len(rows)}')
    return manifest_path.parent/rows[0]['path']


def read_frame(video:Path,index:int):
    cap=cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES,int(index)); ok,im=cap.read(); fps=cap.get(cv2.CAP_PROP_FPS) or 29.97; cap.release()
    if not ok or im is None: raise RuntimeError(f'frame {index} missing in {video}')
    return im,float(fps)


def camera(reg,label):
    d=reg['cameras'][label]
    return {'label':label,'K':np.asarray(d['K'],float),'R':np.asarray(d['R_world_to_camera'],float),'C':np.asarray(d['center_cm'],float),'D':np.asarray(d.get('distortion_k1_k2',[0,0]),float)}


def project(cam,X):
    X=np.asarray(X,float).reshape(-1,3); Xc=(cam['R']@(X-cam['C']).T).T; z=Xc[:,2]
    q=(cam['K']@Xc.T).T; uv=q[:,:2]/q[:,2:3]; return uv,z


def undistort_pixels(cam,pts):
    pts=np.asarray(pts,np.float32).reshape(-1,1,2)
    D=np.array([cam['D'][0],cam['D'][1],0,0,0],np.float64)
    return cv2.undistortPoints(pts,cam['K'],D,P=cam['K']).reshape(-1,2)


def fundamental(c1,c2):
    Rrel=c2['R']@c1['R'].T
    trel=c2['R']@(c1['C']-c2['C'])
    tx=np.array([[0,-trel[2],trel[1]],[trel[2],0,-trel[0]],[-trel[1],trel[0],0]],float)
    return np.linalg.inv(c2['K']).T@tx@Rrel@np.linalg.inv(c1['K'])


def symmetric_epi_errors(c1,c2,p1,p2):
    p1=undistort_pixels(c1,p1); p2=undistort_pixels(c2,p2); F=fundamental(c1,c2)
    h1=np.c_[p1,np.ones(len(p1))]; h2=np.c_[p2,np.ones(len(p2))]
    l2=(F@h1.T).T; l1=(F.T@h2.T).T
    d2=np.abs(np.sum(l2*h2,axis=1))/np.maximum(np.hypot(l2[:,0],l2[:,1]),1e-9)
    d1=np.abs(np.sum(l1*h1,axis=1))/np.maximum(np.hypot(l1[:,0],l1[:,1]),1e-9)
    return .5*(d1+d2)


def pose_candidate(model:YOLO,im:np.ndarray,rim_uv:np.ndarray):
    r=model.predict(im,imgsz=960,verbose=False,device='cpu',conf=.15)[0]
    if r.keypoints is None or len(r.keypoints.xy)==0: return None
    xy=r.keypoints.xy.cpu().numpy(); cf=r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
    boxes=r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((len(xy),4))
    best=None
    for i in range(len(xy)):
        valid=cf[i]>.18
        if int(valid.sum())<5: continue
        wrists=[9,10]; wd=[]
        for j in wrists:
            if j<len(xy[i]) and cf[i,j]>.12: wd.append(float(np.linalg.norm(xy[i,j]-rim_uv)))
        wrist=min(wd) if wd else 999.
        b=boxes[i]; center=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2]); cd=float(np.linalg.norm(center-rim_uv))
        # Dunker prior is intentionally weak; cross-view epipolar state decides.
        score=wrist+0.22*cd-5.0*float(np.mean(cf[i,valid]))
        if best is None or score<best[0]: best=(score,xy[i],cf[i],boxes[i])
    if best is None:return None
    return {'xy':best[1],'conf':best[2],'box':best[3],'selection_score':float(best[0])}


def pair_pose_cost(c1,c2,a,b):
    if a is None or b is None:return {'cost':1e6,'median':1e6,'p90':1e6,'n':0}
    n=min(len(a['xy']),len(b['xy'])); keep=(a['conf'][:n]>.22)&(b['conf'][:n]>.22)
    if int(keep.sum())<5:return {'cost':1e5,'median':1e5,'p90':1e5,'n':int(keep.sum())}
    e=symmetric_epi_errors(c1,c2,a['xy'][:n][keep],b['xy'][:n][keep])
    med=float(np.median(e)); p90=float(np.percentile(e,90)); return {'cost':med+.20*p90,'median':med,'p90':p90,'n':int(len(e))}


def synchronize(reg,manifest,legacy_root,out,pose_weights,window=4,min_cameras=4):
    opts=read_json(legacy_root/'freeze_spin/frame_c_camera_chooser/options.json'); nom={x['camera']:int(x['decoded_frame_index']) for x in opts['options']}
    cams={l:camera(reg,l) for l in reg['accepted_camera_labels']}
    model=YOLO(str(pose_weights))
    candidates={}; frame_cache={}
    for label,cam in cams.items():
        rim,_=project(cam,RIM_CENTER_CM[None,:]); rim=rim[0]
        rows=[]
        clip=manifest_clip(manifest,label)
        for off in range(-window,window+1):
            idx=nom[label]+off; im,fps=read_frame(clip,idx); frame_cache[(label,off)]=(im,fps)
            rows.append({'offset':off,'index':idx,'pose':pose_candidate(model,im,rim)})
        candidates[label]=rows

    labels=list(cams); pair_tables={}
    for i,a in enumerate(labels):
        for b in labels[i+1:]:
            tab={}
            for ia,ra in enumerate(candidates[a]):
                for ib,rb in enumerate(candidates[b]): tab[(ia,ib)]=pair_pose_cost(cams[a],cams[b],ra['pose'],rb['pose'])
            pair_tables[(a,b)]=tab

    # Exhaustive search is bounded: <=9^6 = 531,441 combinations for six views.
    best=(float('inf'),None)
    ranges=[range(len(candidates[l])) for l in labels]
    for combo in itertools.product(*ranges):
        cost=.30*sum(abs(candidates[l][j]['offset']) for l,j in zip(labels,combo))
        bad=False
        for i,a in enumerate(labels):
            for k in range(i+1,len(labels)):
                b=labels[k]; v=pair_tables[(a,b)][(combo[i],combo[k])]
                if v['n']<5: cost+=120.
                cost+=v['cost']
                if cost>=best[0]: bad=True; break
            if bad: break
        if cost<best[0]: best=(cost,combo)
    if best[1] is None: raise RuntimeError('pose synchronization search produced no state')

    chosen={l:candidates[l][j] for l,j in zip(labels,best[1])}
    pairqa={}
    for i,a in enumerate(labels):
        for k in range(i+1,len(labels)):
            b=labels[k]; pairqa[f'{a}__{b}']=pair_tables[(a,b)][(best[1][i],best[1][k])]

    # Find largest mutually compatible set, preferring Broadcast and more cameras.
    good_subset=None
    for n in range(len(labels),2,-1):
        for subset in itertools.combinations(labels,n):
            if 'Broadcast' not in subset: continue
            ok=True
            for a,b in itertools.combinations(subset,2):
                key=f'{a}__{b}' if f'{a}__{b}' in pairqa else f'{b}__{a}'
                q=pairqa[key]
                if q['n']<5 or q['median']>18.0 or q['p90']>35.0: ok=False; break
            if ok: good_subset=list(subset); break
        if good_subset: break
    report={'objective':best[0],'chosen':{k:{'offset':v['offset'],'frame_index':v['index'],'pose_selection_score':None if v['pose'] is None else v['pose']['selection_score']} for k,v in chosen.items()},'pair_qa':pairqa,'usable_camera_subset':good_subset or [],'gates':{'median_epipolar_px_max':18,'p90_epipolar_px_max':35,'min_joint_count':5,'minimum_synchronized_camera_count':int(min_cameras)}}
    (out/'sync_report.json').write_text(json.dumps(report,indent=2))
    if not good_subset or len(good_subset)<int(min_cameras):
        raise RuntimeError(f'no >={int(min_cameras)}-camera exact-state subset passes epipolar gates')
    return chosen,good_subset,frame_cache,report


def moge_infer(model,image,tokens):
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); tensor=torch.from_numpy(rgb).float().permute(2,0,1)/255.0
    with torch.no_grad(): pred=model.infer(tensor,num_tokens=int(tokens))
    depth=pred['depth'].detach().cpu().numpy().astype(np.float32); mask=pred.get('mask')
    if mask is None: valid=np.isfinite(depth)&(depth>0)
    else: valid=mask.detach().cpu().numpy().astype(bool)&np.isfinite(depth)&(depth>0)
    return depth,valid


def floor_grid():
    xs=np.linspace(-4*30.48,36*30.48,55); ys=np.linspace(-25*30.48,25*30.48,65); gx,gy=np.meshgrid(xs,ys)
    return np.column_stack([gx.ravel(),gy.ravel(),np.zeros(gx.size)])


def metric_depth_map(depth,valid,cam):
    P=floor_grid(); uv,zs=project(cam,P); sign=1.0 if np.median(zs[np.isfinite(zs)])>=0 else -1.0; z=np.abs(zs)
    x=np.rint(uv[:,0]).astype(int); y=np.rint(uv[:,1]).astype(int)
    ok=np.isfinite(uv).all(1)&(x>=2)&(x<W-2)&(y>=2)&(y<H-2)&(z>20)
    d=depth[y[ok],x[ok]].astype(float); zz=z[ok]; vv=valid[y[ok],x[ok]]&np.isfinite(d)&(d>.02); d=d[vv]; zz=zz[vv]
    if len(d)<30: raise RuntimeError(f"{cam['label']}: only {len(d)} metric floor-depth anchors")
    hold=np.arange(len(d))%7==0; train=~hold; s0=float(np.median(zz[train]/np.maximum(d[train],1e-6)))
    fit=least_squares(lambda p:p[0]*d[train]+p[1]-zz[train],[s0,0],loss='soft_l1',f_scale=40,max_nfev=2000)
    pred=fit.x[0]*d+fit.x[1]; err=np.abs(pred-zz); held=err[hold]
    mapped=fit.x[0]*depth.astype(float)+fit.x[1]
    return mapped,sign,{'anchors':int(len(d)),'heldout_median_cm':float(np.median(held)),'heldout_p95_cm':float(np.percentile(held,95)),'scale':float(fit.x[0]),'offset':float(fit.x[1]),'z_sign':int(sign)}


def make_cloud(image,depth_abs,valid,sign,cam,stride=2):
    yy,xx=np.indices((H,W)); ok=(xx%stride==0)&(yy%stride==0)&valid&np.isfinite(depth_abs)&(depth_abs>20)&(depth_abs<15000)
    ys,xs=np.where(ok); za=depth_abs[ys,xs]; z=sign*za
    K=cam['K']; xn=(xs-K[0,2])/K[0,0]; yn=(ys-K[1,2])/K[1,1]; Xc=np.c_[xn*z,yn*z,z]; Xw=(cam['R'].T@Xc.T).T+cam['C']
    return Xw.astype(np.float32),image[ys,xs].copy()


def orbit_pose(C0,R0,pivot,deg):
    t=math.radians(float(deg)); Q=np.array([[math.cos(t),-math.sin(t),0],[math.sin(t),math.cos(t),0],[0,0,1]],float)
    return R0@Q.T,pivot+Q@(C0-pivot)


def raster(cloud,K,R,C):
    X,col=cloud; Xc=(R@(X.astype(float)-C).T).T; z=Xc[:,2]; q=(K@Xc.T).T; uv=q[:,:2]/q[:,2:3]; u=np.rint(uv[:,0]).astype(int);v=np.rint(uv[:,1]).astype(int)
    ok=np.isfinite(uv).all(1)&(z>20)&(u>=0)&(u<W)&(v>=0)&(v<H); ids=np.where(ok)[0]
    im=np.zeros((H,W,3),np.uint8); mask=np.zeros((H,W),bool); zbuf=np.full(H*W,np.inf,np.float32)
    if len(ids):
        pix=v[ids]*W+u[ids]; np.minimum.at(zbuf,pix,z[ids].astype(np.float32)); win=ids[z[ids]<=zbuf[pix]+1e-4]; im[v[win],u[win]]=col[win];mask[v[win],u[win]]=True
    # one-pixel deterministic hole dilation, source pixels only
    for _ in range(1):
        base=im.copy(); bm=mask.copy()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            si=np.roll(np.roll(base,dy,0),dx,1); sm=np.roll(np.roll(bm,dy,0),dx,1); take=(~mask)&sm;im[take]=si[take];mask[take]=True
    return im,mask


def angular_distance(Ca,Cb,pivot):
    a=Ca-pivot;b=Cb-pivot;a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);return math.degrees(math.acos(float(np.clip(np.dot(a,b),-1,1))))


def encode_video(frames_dir,out_mp4,fps=30):
    subprocess.run(['ffmpeg','-y','-v','error','-framerate',str(fps),'-i',str(frames_dir/'orbit_%03d.png'),'-c:v','libx264','-preset','medium','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(out_mp4)],check=True)


def compose_replay(broadcast,freeze_index,source_fps,orbit,out):
    freeze_t=freeze_index/source_fps; start=max(0.,freeze_t-2.2); end=freeze_t+1.8
    cmd=['ffmpeg','-y','-v','error','-i',str(broadcast),'-i',str(orbit),'-filter_complex',
         f"[0:v]trim=start={start:.6f}:end={freeze_t:.6f},setpts=PTS-STARTPTS,fps=30,scale=960:540:flags=lanczos[pre];"
         f"[1:v]setpts=PTS-STARTPTS[orb];"
         f"[0:v]trim=start={freeze_t+1/source_fps:.6f}:end={end:.6f},setpts=PTS-STARTPTS,fps=30,scale=960:540:flags=lanczos[post];"
         "[pre][orb][post]concat=n=3:v=1:a=0[outv]",'-map','[outv]','-c:v','libx264','-preset','medium','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    subprocess.run(cmd,check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--registry',type=Path,required=True);ap.add_argument('--legacy-root',type=Path,required=True);ap.add_argument('--pose-model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tokens',type=int,default=1000);ap.add_argument('--max-degree',type=float,default=18.0);ap.add_argument('--orbit-frames',type=int,default=61);ap.add_argument('--min-sync-cameras',type=int,default=4);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    reg=read_json(a.registry)
    chosen,subset,cache,sync=synchronize(reg,a.manifest,a.legacy_root,a.out,a.pose_model,window=4,min_cameras=a.min_sync_cameras)
    cams={l:camera(reg,l) for l in subset}
    torch.set_num_threads(max(1,min(4,torch.get_num_threads()))); model=MoGeModel.from_pretrained('Ruicheng/moge-2-vits-normal').eval()
    clouds={}; depthqa={}; selected_images={}
    for l in subset:
        im,fps=cache[(l,chosen[l]['offset'])]; selected_images[l]=im
        depth,valid=moge_infer(model,im,a.tokens); metric,sgn,q=metric_depth_map(depth,valid,cams[l]); clouds[l]=make_cloud(im,metric,valid,sgn,cams[l],stride=2);q['cloud_points']=int(len(clouds[l][0]));depthqa[l]=q
    anchor=cams['Broadcast']; K0=anchor['K'];C0=anchor['C'];R0=anchor['R'];pivot=RIM_CENTER_CM.copy()
    fd=a.out/'orbit_frames';fd.mkdir(exist_ok=True); key=[]
    for i in range(a.orbit_frames):
        phase=i/max(a.orbit_frames-1,1);deg=float(a.max_degree*math.sin(math.pi*phase))
        if abs(deg)<1e-8:
            out=selected_images['Broadcast'].copy();mask=np.ones((H,W),bool);order=['Broadcast']
        else:
            Rt,Ct=orbit_pose(C0,R0,pivot,deg); order=sorted(subset,key=lambda l:angular_distance(cams[l]['C'],Ct,pivot));out=np.zeros((H,W,3),np.uint8);mask=np.zeros((H,W),bool)
            for l in order:
                ri,rm=raster(clouds[l],K0,Rt,Ct);take=(~mask)&rm;out[take]=ri[take];mask|=take
        cv2.imwrite(str(fd/f'orbit_{i:03d}.png'),out)
        if i in {0,a.orbit_frames//4,a.orbit_frames//2,3*a.orbit_frames//4,a.orbit_frames-1}: key.append({'frame':i,'degree':deg,'resolved_fraction':float(mask.mean()),'source_order':order})
    orbit=a.out/'adams_jazz_v13_freeze_swivel_orbit_native.mp4';encode_video(fd,orbit,30)
    bclip=manifest_clip(a.manifest,'Broadcast'); _,sfps=read_frame(bclip,chosen['Broadcast']['frame_index']); final_native=a.out/'adams_jazz_v13_freeze_swivel_native.mp4';compose_replay(bclip,chosen['Broadcast']['frame_index'],sfps,orbit,final_native)
    uhd=a.out/'adams_jazz_v13_freeze_swivel_UHD.mp4';subprocess.run(['ffmpeg','-y','-v','error','-i',str(final_native),'-vf','hqdn3d=0.6:0.6:2.0:2.0,scale=3840:2160:flags=lanczos,cas=0.22,fps=30','-c:v','libx264','-profile:v','high','-level','5.1','-preset','medium','-crf','16','-maxrate','36M','-bufsize','72M','-pix_fmt','yuv420p','-movflags','+faststart',str(uhd)],check=True)
    report={'schema':'hoopvision.freeview.render.v13.1','status':'RENDERED','game_id':'0022500301','event_num':489,'camera_registry_status':reg['status'],'calibrated_camera_count':reg['accepted_camera_count'],'synchronized_render_subset':subset,'sync':sync,'depth_qa':depthqa,'orbit':{'max_degree':a.max_degree,'frames':a.orbit_frames,'key_frames':key},'appearance_policy':'official NBA source pixels only; MoGe depth/pose are geometry evidence, never RGB generation','outputs':{'native':final_native.name,'uhd':uhd.name,'orbit_native':orbit.name}}
    (a.out/'render_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':'RENDERED','subset':subset,'native':str(final_native),'uhd':str(uhd)},indent=2))

if __name__=='__main__':main()