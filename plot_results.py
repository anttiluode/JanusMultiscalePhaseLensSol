"""Optional visual audit; requires matplotlib. Generates fixed held-out examples."""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from benchmark import SIZE,frames,TRAIN_SEEDS,train_world,noisy_image,wiener_prior,wiener,infer_world
from phase_world import ComplexDirectionalFrame,generate_scene
import matplotlib.pyplot as plt

frame=ComplexDirectionalFrame(SIZE,scales=4,orientations=4)
training=frames(TRAIN_SEEDS)
vocabulary=train_world(frame,training)
prior=wiener_prior(training)
fig,axes=plt.subplots(2,5,figsize=(11,5.3),constrained_layout=True)
for row,(seed,domain) in enumerate(((9005,'familiar'),(19005,'unrelated'))):
    clean=generate_scene(seed,SIZE,domain)
    sigma=.22
    obs=noisy_image(clean,sigma,500000+seed+int(sigma*1000))
    phase,_=infer_world(frame,vocabulary,obs,blend=.75,phase=True)
    no_phase,_=infer_world(frame,vocabulary,obs,blend=.75,phase=False)
    images=[clean,obs,gaussian_filter(obs,1.3),phase,no_phase]
    for col,(title,img) in enumerate(zip(['Clean','Noisy','Gaussian','Phase vocabulary','No-phase vocabulary'],images)):
        axes[row,col].imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        axes[row,col].set_title(title,fontsize=9)
        axes[row,col].set_xlabel(f'{domain}; MSE={np.mean((img-clean)**2):.4f}',fontsize=8)
        axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
Path('results').mkdir(exist_ok=True)
fig.savefig('results/heldout_examples.png',dpi=140)
plt.close(fig)
print('Wrote results/heldout_examples.png')
