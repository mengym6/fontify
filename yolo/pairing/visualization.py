"""Task-separated paired TensorBoard previews, rendered only in memory."""
import colorsys
import numpy as np
from PIL import Image, ImageDraw


def paired_preview(images, targets, output, index):
    """BF/JT rows; target A/B and prediction A/B columns. No raw-only columns."""
    size, header = 256, 30
    canvas = Image.new('RGB', (size * 4, (size + header) * 2 + 24), 'white')
    target = targets[index]
    pred_masks = output['masks'][index].detach().sigmoid().cpu().numpy() >= .5
    pred_presence = output['presence'][index].detach().sigmoid().cpu().numpy()
    pred_groups = output['groups'].detach().cpu().numpy()
    gt_masks = target['masks'].cpu().numpy() >= .5
    gt_presence = target['present'].cpu().numpy()
    gt_groups = target['groups'].cpu().numpy()
    gt_known = target.get('mask_supervised', target['present'] == 1).cpu().numpy()
    for task, name in enumerate(('BF', 'JT')):
        top = task * (size + header)
        for column, (side, predicted) in enumerate([(0,False),(1,False),(0,True),(1,True)]):
            pixels = (images[index,side].detach().cpu().permute(1,2,0).numpy()*255).clip(0,255).astype('uint8')
            im = Image.fromarray(pixels).resize((size,size)).convert('RGBA')
            masks = pred_masks[:,side] if predicted else gt_masks[:,side]
            presence = pred_presence if predicted else gt_presence
            groups = pred_groups if predicted else gt_groups
            keep = pred_presence.max(1)>=.5 if predicted else gt_known[:,side]
            marks = []
            for q in range(len(masks)):
                if groups[q]!=task or not keep[q] or not masks[q].any():
                    continue
                extra = presence[q,0]<.5 and presence[q,1]>=.5
                missing = presence[q,0]>=.5 and presence[q,1]<.5
                if side==0 and extra:
                    continue
                rgb = (225,0,0) if extra else (220,125,0) if side==1 and missing else tuple(round(c*255) for c in colorsys.hsv_to_rgb(q*.618%1,.85,.8))
                mask = Image.fromarray((masks[q]*255).astype('uint8')).resize((size,size),Image.Resampling.NEAREST)
                array = np.asarray(mask)>0
                # A one-pixel binary boundary prevents translucent fills obscuring small BF regions.
                inner = array.copy()
                inner[1:,:] &= array[:-1,:]; inner[:-1,:] &= array[1:,:]
                inner[:,1:] &= array[:,:-1]; inner[:,:-1] &= array[:,1:]
                alpha = np.where(array & ~inner,220,np.where(array,35,0)).astype('uint8')
                layer=Image.new('RGBA',im.size,rgb+(0,));layer.putalpha(Image.fromarray(alpha));im=Image.alpha_composite(im,layer)
                ys,xs=np.nonzero(array)
                label=('ERR' if extra else 'MISS' if side==1 and missing else 'Q' if predicted else 'T')+'-'+str(q)
                marks.append((int(xs.mean()),int(ys.mean()),label,rgb))
            draw=ImageDraw.Draw(im)
            for x,y,label,rgb in marks:
                draw.text((x,y),label,fill=rgb+(255,),stroke_width=1,stroke_fill=(255,255,255,220))
            canvas.paste(im.convert('RGB'),(column*size,top+header))
            draw=ImageDraw.Draw(canvas)
            title=name+' | '+('GT pred' if side==0 else 'Gen pred') if predicted else name+' | '+('GT target' if side==0 else 'Gen target')
            unknown=not predicted and not bool(target['complete'][side,task]) and not any((gt_groups==task)&gt_known[:,side])
            draw.text((column*size+5,top+8),title+(' | UNANNOTATED' if unknown else f' | n={len(marks)}'),fill='black')
    ImageDraw.Draw(canvas).text((5,canvas.height-17),'Q IDs match across prediction A/B only; T IDs are targets. ERR=extra; MISS=expected missing area.',fill='black')
    return np.asarray(canvas)
