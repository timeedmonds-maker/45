from __future__ import annotations

"""Solve six candidate physical camera views for the Adams-Jazz freeze swivel.

This is a downstream HoopVision v1.3 tool. It consumes the frozen v1.3 court
landmark model/metric court definition, the immutable official event-489 feeds,
and previously validated static-geometry evidence for the three legacy camera
anchors. It never changes v1.3 tracking or identity behaviour.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from hoopvision.core.court_standards import get_court_standard
from hoopvision.freeview.camera_solver import (
    CameraObservations, CameraState, CurveObservation, LineObservation,
    PointObservation, acceptance_report, multistart_stability,
    perturbation_stability, rim_world_points, solve_camera,
    solve_shared_center_states, target_inner_corners, target_inner_lines, target_stripe_center_lines,
    project_points,
)

CANDIDATE_LABELS = (
    "Broadcast",
    "Left Above Rim",
    "Right Above Rim",
    "Left Slash",
    "In Arena",
    "Right Slash",
)


def read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def safe_name(label: str) -> str:
    return "_".join(label.replace("/", " ").split())


def manifest_clip(manifest_path: Path, label: str) -> Path:
    m = read_json(manifest_path)
    rows = [x for x in m.get("feeds", []) if x.get("label") == label]
    if len(rows) != 1:
        raise RuntimeError(f"manifest expected one {label!r}, got {len(rows)}")
    return manifest_path.parent / rows[0]["path"]


def extract_frame(video: Path, frame_index: int, out: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read(); cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode frame {frame_index} from {video}")
    if frame.shape[:2] != (540, 960):
        raise RuntimeError(f"unexpected source resolution {frame.shape[:2]} for {video}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame):
        raise RuntimeError(f"could not write {out}")


def extract_selected_event489_frames(manifest: Path, legacy_root: Path, out_dir: Path) -> dict[str, Path]:
    opts = read_json(legacy_root / "freeze_spin/frame_c_camera_chooser/options.json")
    by_label = {x["camera"]: x for x in opts["options"]}
    paths = {}
    for label in CANDIDATE_LABELS:
        row = by_label[label]
        out = out_dir / row["file"]
        extract_frame(manifest_clip(manifest, label), int(row["decoded_frame_index"]), out)
        paths[label] = out
    return paths


def v13_court_points(model: YOLO, image_path: Path, conf_min: float = 0.35):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"missing frame {image_path}")
    best = None
    for imgsz in (640, 960, 1280):
        result = model.predict(image, imgsz=imgsz, verbose=False, device="cpu")[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            continue
        xy = result.keypoints.xy[0].cpu().numpy().astype(float)
        if result.keypoints.conf is None:
            cf = np.ones(len(xy), dtype=float)
        else:
            cf = result.keypoints.conf[0].cpu().numpy().astype(float)
        score = int(np.sum(cf >= conf_min))
        cand = (score, imgsz, xy, cf)
        if best is None or cand[0] > best[0]:
            best = cand
        if score >= 10:
            break
    if best is None:
        raise RuntimeError(f"no v1.3 court landmarks detected for {image_path}")
    return {"imgsz": int(best[1]), "xy": best[2], "conf": best[3]}


def court_obs(label: str, det: dict, min_conf: float = 0.35) -> CameraObservations:
    standard = get_court_standard("NBA")
    # Frozen v1.3 vertices use full-court centimetres. Convert to the legacy/
    # replay basket-local frame: board face X=0, centre of board Y=0.
    world = np.asarray(standard.vertices, float).copy()
    world[:, 0] -= 4.0 * 30.48
    world[:, 1] -= standard.width_cm / 2.0
    o = CameraObservations(label=label, width=960, height=540, pp_prior_sigma_px=180.0)
    for i, (xy, cf) in enumerate(zip(det["xy"], det["conf"])):
        if i >= len(world) or not np.isfinite(xy).all() or float(cf) < min_conf:
            continue
        o.points.append(PointObservation(world[i], np.asarray(xy, float), weight=max(0.35, math.sqrt(float(cf))), family="v13_court"))
    return o


def clone_obs(base: CameraObservations, label: str | None = None) -> CameraObservations:
    out = CameraObservations(
        label=label or base.label, width=base.width, height=base.height,
        center_prior_cm=None if base.center_prior_cm is None else np.asarray(base.center_prior_cm,float).copy(),
        center_prior_sigma_cm=base.center_prior_sigma_cm,
        focal_prior_px=base.focal_prior_px,
        focal_prior_sigma_log=base.focal_prior_sigma_log,
        pp_prior_px=None if base.pp_prior_px is None else np.asarray(base.pp_prior_px,float).copy(),
        pp_prior_sigma_px=base.pp_prior_sigma_px,
    )
    out.points=[PointObservation(x.world.copy(),x.image.copy(),x.weight,x.family) for x in base.points]
    out.lines=[LineObservation(x.world_segment.copy(),x.image_points.copy(),x.weight,x.family) for x in base.lines]
    out.curves=[CurveObservation(x.world_curve.copy(),x.image_points.copy(),x.weight,x.family) for x in base.curves]
    return out


def add_target_corners(obs: CameraObservations, spec: dict, mirror: bool):
    W = target_inner_corners(mirror)
    for name, uv in spec.items():
        if name in W:
            obs.points.append(PointObservation(W[name], np.asarray(uv,float), 2.0, "target"))


def add_target_lines(obs: CameraObservations, spec: dict, mirror: bool):
    W = target_stripe_center_lines(mirror)
    for name, uv in spec.items():
        if name in W and len(uv):
            obs.lines.append(LineObservation(W[name], np.asarray(uv,float), 1.7, "target"))


def add_rim_curve(obs: CameraObservations, uv):
    if uv:
        obs.curves.append(CurveObservation(rim_world_points(360), np.asarray(uv,float), 1.8, "rim"))


def solve_mirrors(base: CameraObservations, enrich, label: str):
    candidates=[]
    for mirror in (False,True):
        o=clone_obs(base,label)
        enrich(o,mirror)
        if len(o.points)<4:
            continue
        try:
            s=solve_camera(o)
            candidates.append((s.cost,s,o,mirror))
        except Exception as e:
            candidates.append((float("inf"),None,o,mirror))
    good=[x for x in candidates if x[1] is not None]
    if not good:
        raise RuntimeError(f"{label}: both basket-mirror hypotheses failed")
    return min(good,key=lambda x:x[0])


def qa_camera(state: CameraState, obs: CameraObservations, seed: int):
    p=perturbation_stability(obs,state,trials=10,seed=seed)
    m=multistart_stability(obs,state,trials=8,seed=seed+1)
    return acceptance_report(state,p,m)


def legacy_exact_left_above_rim(legacy_root: Path) -> CameraState:
    reg=read_json(legacy_root/"freeze_spin/adams_jazz_game_camera_registry_v3.json")
    cam=reg["cameras"]["Left Above Rim"]
    row=cam["event_cameras"]["489"]
    C=np.asarray(cam["physical_camera_center_prior_cm"],float)
    f=float(row["focal_px"]); pp=np.asarray(row["principal_point_px"],float); R=cv2.Rodrigues(np.asarray(row["rvec"],float).reshape(3,1))[0]
    return CameraState(label="Left Above Rim",K=np.array([[f,0,pp[0]],[0,f,pp[1]],[0,0,1]],float),R=R,C=C,distortion=np.zeros(2),cost=0.0,metrics={"legacy_certified":True,"source":"v37/v41/v42 accepted state"})

def certified_centers(legacy_root: Path):
    reg=read_json(legacy_root/"freeze_spin/adams_jazz_game_camera_registry_v6.json")
    rar=np.asarray(reg["accepted_cameras"]["Right Above Rim"]["physical_camera_center_cm"],float)
    bft=np.asarray(reg["accepted_cameras"]["Broadcast"]["shared_camera_center_ft"],float)
    return rar,bft*30.48

def solve_center_locked(base: CameraObservations, center_cm, enrich, label: str):
    candidates=[]
    for mirror in (False,True):
        o=clone_obs(base,label);o.center_prior_cm=np.asarray(center_cm,float);o.center_prior_sigma_cm=.25
        enrich(o,mirror)
        try:
            st=solve_camera(o);candidates.append((st.cost,st,o,mirror))
        except Exception:pass
    if not candidates:raise RuntimeError(f"{label}: center-locked solve failed")
    return min(candidates,key=lambda x:x[0])


def overlay(image_path: Path, state: CameraState, obs: CameraObservations, out: Path):
    im=cv2.imread(str(image_path)); assert im is not None
    rv,_=cv2.Rodrigues(state.R)
    logf=math.log(float(state.K[0,0])); cx,cy=state.K[0,2],state.K[1,2]
    k1,k2=state.distortion
    # Observed court landmarks + predicted points.
    for o in obs.points:
        uv,_=project_points(o.world[None,:],rv.reshape(-1),state.C,logf,cx,cy,k1,k2)
        a=tuple(np.rint(o.image).astype(int)); b=tuple(np.rint(uv[0]).astype(int))
        cv2.circle(im,a,3,(0,255,255),-1,cv2.LINE_AA); cv2.circle(im,b,3,(255,255,255),1,cv2.LINE_AA)
        cv2.line(im,a,b,(255,255,255),1,cv2.LINE_AA)
    # Full rim and target opening are always visualized, even if held out.
    rim=rim_world_points(240); uv,_=project_points(rim,rv.reshape(-1),state.C,logf,cx,cy,k1,k2)
    pts=np.rint(uv).astype(np.int32).reshape(-1,1,2); cv2.polylines(im,[pts],True,(255,0,255),2,cv2.LINE_AA)
    c=target_inner_corners(False); seq=[c['target_inner_top_left'],c['target_inner_top_right'],c['target_inner_bottom_right'],c['target_inner_bottom_left']]
    tuv,_=project_points(np.asarray(seq,float),rv.reshape(-1),state.C,logf,cx,cy,k1,k2)
    cv2.polylines(im,[np.rint(tuv).astype(np.int32).reshape(-1,1,2)],True,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(im,f"{state.label} f={state.K[0,0]:.0f}px",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(str(out),im)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--legacy-root',type=Path,required=True)
    ap.add_argument('--court-model',type=Path,required=True)
    ap.add_argument('--right-slash-event540-frame',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    frames=a.out/'frames'; frames.mkdir(exist_ok=True)
    frame_paths=extract_selected_event489_frames(a.manifest,a.legacy_root,frames)
    rs540=frames/'Right_Slash_event540.png'; shutil.copy2(a.right_slash_event540_frame,rs540)

    model=YOLO(str(a.court_model))
    detections={}
    for label,p in {**frame_paths,'Right Slash event540':rs540}.items():
        detections[label]=v13_court_points(model,p)
    (a.out/'v13_court_landmarks.json').write_text(json.dumps({k:{'imgsz':v['imgsz'],'xy':v['xy'].tolist(),'conf':v['conf'].tolist()} for k,v in detections.items()},indent=2))

    states={};reports={};observations={}
    # Left Above Rim retains the already hard-gated exact event state.
    lar=legacy_exact_left_above_rim(a.legacy_root);states['Left Above Rim']=lar
    reports['Left Above Rim']={'accepted':True,'source':'legacy_hard_gated_exact_event_state','metrics':lar.metrics}
    rar_center,broadcast_center=certified_centers(a.legacy_root)

    # Re-solve the operated state of the certified Right Above Rim physical mount
    # using the v1.3 court landmarks plus the full native rim conic.
    rar_base=court_obs('Right Above Rim',detections['Right Above Rim'])
    rar_spec=read_json(a.legacy_root/'freeze_spin/adams_jazz_frame_c_right_above_rim_fixed_geometry_v46.json')
    def enrich_rar(o,mirror):add_rim_curve(o,rar_spec['rim_inner_edge_samples_px'])
    _,rar,rar_obs,rar_mirror=solve_center_locked(rar_base,rar_center,enrich_rar,'Right Above Rim')
    rar_qa=qa_camera(rar,rar_obs,130473);rar_qa.update({'source':'certified_fixed_mount_center_plus_v13_court_plus_full_rim','target_mirror_y':rar_mirror})
    observations['Right Above Rim']=rar_obs;reports['Right Above Rim']=rar_qa
    if rar_qa['accepted']:states['Right Above Rim']=rar

    # Broadcast uses its independently certified shared optical centre, v1.3
    # court landmarks, white target-stripe centre lines and full rim contour.
    b_base=court_obs('Broadcast',detections['Broadcast'])
    b_target=read_json(a.legacy_root/'freeze_spin/adams_jazz_broadcast_target_lines_v87.json')
    b_rim=read_json(a.legacy_root/'freeze_spin/adams_jazz_broadcast_rim_v88.json')
    def enrich_b(o,mirror):
        add_target_lines(o,b_target['observed_line_samples_px'],mirror);add_rim_curve(o,b_rim['rim_contour_samples_px'])
    _,bcam,b_obs,b_mirror=solve_center_locked(b_base,broadcast_center,enrich_b,'Broadcast')
    b_qa=qa_camera(bcam,b_obs,130490);b_qa.update({'source':'certified_shared_center_plus_v13_court_plus_target_lines_plus_rim','target_mirror_y':b_mirror})
    observations['Broadcast']=b_obs;reports['Broadcast']=b_qa
    if b_qa['accepted']:states['Broadcast']=bcam

    # Left Slash: v1.3 floor landmarks + independent full target/rim native-pixel evidence.
    ls_base=court_obs('Left Slash',detections['Left Slash'])
    ls_spec=read_json(a.legacy_root/'freeze_spin/adams_jazz_left_slash_frame_c_metric_v59.json')['views'][0]
    def enrich_ls(o,mirror):
        add_target_corners(o,ls_spec['landmarks'],mirror); add_rim_curve(o,ls_spec['rim_curve_samples_px'])
    _,ls,ls_obs,ls_mirror=solve_mirrors(ls_base,enrich_ls,'Left Slash')
    ls_qa=qa_camera(ls,ls_obs,130590)
    observations['Left Slash']=ls_obs
    reports['Left Slash']={**ls_qa,'target_mirror_y':ls_mirror,'source':'v13_court_plus_full_target_rim'}
    if ls_qa['accepted']: states['Left Slash']=ls

    # In Arena: v1.3 floor landmarks + three visible noncoplanar target corners.
    ia_base=court_obs('In Arena',detections['In Arena'])
    ia_spec=read_json(a.legacy_root/'freeze_spin/adams_jazz_in_arena_target_v76.json')
    def enrich_ia(o,mirror): add_target_corners(o,ia_spec['target_inner_corners_px'],mirror)
    _,ia,ia_obs,ia_mirror=solve_mirrors(ia_base,enrich_ia,'In Arena')
    ia_qa=qa_camera(ia,ia_obs,130760)
    observations['In Arena']=ia_obs
    reports['In Arena']={**ia_qa,'target_mirror_y':ia_mirror,'source':'v13_court_plus_visible_target'}
    if ia_qa['accepted']: states['In Arena']=ia

    # Right Slash: establish the same-game physical centre from the geometry-rich
    # event-540 state, then lock that centre for the event-489 PTZ state.
    # This avoids asking a planar target frame to identify distance by itself.
    rs489=court_obs('Right Slash',detections['Right Slash'])
    rs540base=court_obs('Right Slash event540',detections['Right Slash event540'])
    rs540spec=read_json(a.legacy_root/'freeze_spin/right_slash_event540_metric_support_v108.json')
    rs_trials=[]
    for mirror in (False,True):
        o540=clone_obs(rs540base,'Right Slash event540')
        add_target_lines(o540,rs540spec['candidate_source_native_observations_px']['target_line_samples_px'],mirror)
        try:
            s540=solve_camera(o540)
            qa540=qa_camera(s540,o540,130540+(1 if mirror else 0))
            target=clone_obs(rs489,'Right Slash');target.center_prior_cm=s540.C.copy();target.center_prior_sigma_cm=.25
            seed=np.r_[cv2.Rodrigues(s540.R)[0].reshape(-1),s540.C,math.log(s540.K[0,0]),s540.K[0,2],s540.K[1,2],s540.distortion]
            st=solve_camera(target,extra_seeds=[seed])
            qat=qa_camera(st,target,130108+(1 if mirror else 0))
            score=float(s540.cost+st.cost)
            rs_trials.append((score,st,target,mirror,s540,o540,qa540,qat))
        except Exception:
            pass
    if rs_trials:
        _,rs,rs_obs,rs_mirror,s540,o540,qa540,rs_qa=min(rs_trials,key=lambda x:x[0])
        rs_qa['gates']['event540_camera_accepted']=bool(qa540['accepted'])
        rs_qa['accepted']=bool(all(rs_qa['gates'].values()))
        rs_qa.update({'target_mirror_y':rs_mirror,'event540_metrics':s540.metrics,'event540_qa':qa540,'shared_center_cm':s540.C.tolist(),'source':'v13_event540_target_lines_establish_center_then_center_locked_event489'})
        reports['Right Slash']=rs_qa; observations['Right Slash']=rs_obs
        if rs_qa['accepted']: states['Right Slash']=rs
    else:
        reports['Right Slash']={'accepted':False,'source':'event540_or_target_solve_failed'}

    # Anchor overlays use detected v1.3 court points only for diagnostic display.
    observations['Left Above Rim']=court_obs('Left Above Rim',detections['Left Above Rim'])

    for label,state in states.items():
        overlay(frame_paths[label],state,observations[label],a.out/f"overlay_{safe_name(label)}.png")

    distinct_new=[x for x in ('Left Slash','In Arena','Right Slash') if x in states]
    status='PASS_SIX_CAMERA_SOLVE' if len(states)>=6 else ('PASS_FOUR_PLUS_CAMERA_SOLVE' if len(states)>=4 else 'FAIL_INSUFFICIENT_CAMERA_SOLVE')
    out={
        'schema':'hoopvision.freeview.camera-registry.v13.1',
        'status':status,
        'upstream_engine':{'name':'Solved Engine 1.3','branch':'release/v1.3','immutable':True},
        'game_id':'0022500301','event_num':489,
        'candidate_labels':list(CANDIDATE_LABELS),
        'accepted_camera_count':len(states),
        'accepted_camera_labels':list(states.keys()),
        'newly_solved_camera_labels':distinct_new,
        'cameras':{k:v.as_json() for k,v in states.items()},
        'qa':reports,
        'policy':{
            'moving_players_or_ball_never_move_camera':True,
            'v13_tracking_identity_not_modified':True,
            'static_nba_geometry_only_for_calibration':True,
            'fail_closed_on_camera_instability':True,
        },
    }
    (a.out/'camera_registry_v13.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'status':status,'accepted':list(states),'new':distinct_new},indent=2))
    if len(states)<4:
        raise SystemExit(2)

if __name__=='__main__': main()