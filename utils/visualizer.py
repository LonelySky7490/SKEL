import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def check_space_distribution(embeddings, outputs, epoch, n_iter, save_dir):
    """
    Visualize spatial distribution of Visual, Audio, Prompts (excluding Raw Text).
    """
##  1.
    
## A. Visual Features (Post-Transformer) ->
## [B, 10, 1024] -> Mean -> [B, 1024]
    visual_feats = embeddings['vision'].mean(dim=1).detach().cpu().numpy()

## B. Audio Features (Post-Transformer) ->
## [B, 10, 1024] -> Mean -> [B, 1024]
    if 'audio' in embeddings:
        audio_feats = embeddings['audio'].mean(dim=1).detach().cpu().numpy()
    else:
        print("Warning: No audio embeddings found!")
        audio_feats = np.zeros_like(visual_feats)
    
## C. Raw Text Prompt
    
## D. Learned Prompts (Post-Encoder) ->
## [B, N, 1024] -> [N, B, 1024]
    if 'prompts' in outputs and outputs['prompts'] is not None:
        prompts_tensor = outputs['prompts'].detach().cpu().permute(1, 0, 2)
        num_prompts = prompts_tensor.shape[0]
    else:
## Baseline prompt
        num_prompts = 0
    
    prompt_data_list = []
    for i in range(num_prompts):
        p_i = prompts_tensor[i].numpy()
        prompt_data_list.append(p_i)
        
##  2. t-SNE
    
## : [Visual] + [Audio] + [Prompts...]
    data_blocks = [visual_feats, audio_feats] + prompt_data_list
    
    if len(data_blocks) == 0:
        return

    X = np.concatenate(data_blocks, axis=0)
    
## Labels
## 0: Visual, 1: Audio, 2: P0, 3: P1...
    labels = []
    labels.extend([0] * len(visual_feats))
    labels.extend([1] * len(audio_feats))
    
    for i in range(num_prompts):
        labels.extend([2 + i] * len(prompt_data_list[i])) 
    
    labels = np.array(labels)
    
    print(f"Running t-SNE check... Total points: {len(X)}")
    try:
## perplexity
        perp = min(30, len(X) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, init='pca', learning_rate='auto', random_state=42)
        X_embedded = tsne.fit_transform(X)
    except Exception as e:
        print(f"t-SNE skipped due to error (maybe too few points): {e}")
        return
    
##  3. ()
    plt.figure(figsize=(12, 10))
    
## 1. Visual Features ( - Label 0)
## XY labels==0
    plt.scatter(
        X_embedded[labels==0, 0], X_embedded[labels==0, 1], 
        c='red', label='Visual Feats', marker='x', s=60, zorder=1
    )

## 2. Audio Features ( - Label 1)
## XY labels==1
    plt.scatter(
        X_embedded[labels==1, 0], X_embedded[labels==1, 1], 
        c='blue', label='Audio Feats', marker='P', s=60, zorder=1
    )
    
## 3. Prompts ( - Label 2+)
    colors = ['purple', 'orange', 'cyan', 'brown', 'lime', 'pink']
## Prompt 0 Visual()Prompt 1 Audio()
    prompt_names = ['Prompt 0 (Aim:Visual)', 'Prompt 1 (Aim:Audio)', 'Prompt 2', 'Prompt 3']
    
    for i in range(num_prompts):
        idx = 2 + i
        color = colors[i % len(colors)]
        name = prompt_names[i] if i < len(prompt_names) else f'Prompt {i}'
        
## XY labels==idx
        plt.scatter(
            X_embedded[labels==idx, 0], X_embedded[labels==idx, 1], 
            c=color, label=name, s=80, edgecolors='white', alpha=0.9, zorder=2
        )
    
    plt.title(f"Focus Check (No Text Noise) Epoch-{epoch} Iter-{n_iter}\nExpect: Purple->Red, Orange->Blue")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
## prompt focus check baseline check
    file_prefix = 'tsne_focus' if num_prompts > 0 else 'tsne_baseline'
    save_path = os.path.join(save_dir, f'{file_prefix}_E{epoch}_I{n_iter}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE saved to {save_path}")