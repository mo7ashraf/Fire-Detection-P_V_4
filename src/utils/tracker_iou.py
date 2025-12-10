from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
@dataclass
class Track:
    id:int; cls:int; score:float; bbox:Tuple[float,float,float,float]
    age:int=0; hits:int=0; time_since_update:int=0; areas:List[float]=field(default_factory=list)
def iou(b1,b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1]); x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    w=max(0.0,x2-x1); h=max(0.0,y2-y1); inter=w*h
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1]); return inter/max(1e-6,a1+a2-inter)
class IOUTracker:
    def __init__(self,iou_thresh=0.3,max_age=30,min_hits=3):
        self.iou_thresh=iou_thresh; self.max_age=max_age; self.min_hits=min_hits; self.next_id=1; self.tracks:List[Track]=[]
    def update(self,dets:List[Tuple[int,float,Tuple[float,float,float,float]]]):
        for tr in self.tracks: tr.age+=1; tr.time_since_update+=1
        for cls,score,bb in dets:
            best_i, best=None, None
            for tr in self.tracks:
                if tr.cls!=cls: continue
                i=iou(tr.bbox,bb); 
                if best is None or i>best_i: best_i, best=i, tr
            if best is not None and best_i>=self.iou_thresh:
                best.bbox=bb; best.score=max(best.score,score); best.hits+=1; best.time_since_update=0; best.areas.append((bb[2]-bb[0])*(bb[3]-bb[1]))
            else:
                tr=Track(self.next_id,cls,score,bb,hits=1,areas=[(bb[2]-bb[0])*(bb[3]-bb[1])]); self.next_id+=1; self.tracks.append(tr)
        self.tracks=[tr for tr in self.tracks if tr.time_since_update<=self.max_age]; return self.tracks
