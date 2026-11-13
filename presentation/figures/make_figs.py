# Crops SYNTHETIC evalboard demo screenshots and burns in a visible label.
import os
from PIL import Image, ImageDraw, ImageFont
# Run from the presentation folder. SHOTS_DIR points at the evalboard demo screenshot folder
# (default: ../../evalboard-demo, relative to this deck).
BASE = os.environ.get('SHOTS_DIR', os.path.join('..', '..', 'evalboard-demo'))
S2 = os.path.join(BASE, 'shots2') + os.sep; S3 = os.path.join(BASE, 'shots3') + os.sep
crops = {  # name: (src, (left, top, right, bottom))
 'syn-scoreboard-forget.png': (S2+'02-scoreboard.png', (430,2000,2140,2690)),
 'syn-scoreboard-judge.png':  (S2+'02-scoreboard.png', (430,1300,2140,2070)),
 'syn-token-layer-map.png':   (S3+'light-03-interp-token-layer-map.png', (440,1660,2090,2680)),
 'syn-circuit-graph.png':     (S3+'light-06-interp-circuit-graph.png', (480,1300,2040,1960)),
 'syn-act-diff-hist.png':     (S2+'11-interp-activation-diff-distribution.png', (440,1280,2030,2040)),
 'syn-pca-scatter.png':       (S2+'12-interp-projection-scatter.png', (430,1100,2080,1900)),
 'syn-head-ablation.png':     (S2+'10-interp-attention-heads.png', (450,1160,2150,2020)),
}
try: font = ImageFont.truetype('DejaVuSans-Bold.ttf', 44)
except Exception: font = ImageFont.load_default()
for name,(src,box) in crops.items():
    crop = Image.open(src).convert('RGB').crop(box)
    h = 64  # banner strip added ABOVE the crop so no content is hidden
    im = Image.new('RGB', (crop.width, crop.height+h), 'white'); im.paste(crop, (0,h))
    d = ImageDraw.Draw(im, 'RGBA')
    txt = 'SYNTHETIC DEMO DATA, not real results'
    w = d.textlength(txt, font=font)
    d.rectangle([0,0,im.width,h], fill=(200,30,30,215))
    d.text(((im.width-w)/2, 8), txt, fill='white', font=font)
    im.save('figures/'+name)
    print(name, im.size)
