from .utils.tracker_iou import Track
def should_alert(tr:Track,min_persistence=8,min_growth_ratio=0.15):
    if tr.hits<min_persistence: return False
    if len(tr.areas)>=min_persistence:
        base=sum(tr.areas[:max(1,len(tr.areas)//3)])/max(1,len(tr.areas[:max(1,len(tr.areas)//3)]))
        growth=(tr.areas[-1]-base)/max(1e-6,base)
        return growth>=min_growth_ratio
    return False
