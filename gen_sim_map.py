import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from models import build_insid3

"""
Select multiple query points in reference image,
display the highlighted patches in target image.
"""


def select_query_pt(image_path):
    """
    Opens the image and allows the user to click a point to get the (y, x) ratios.
    """
    img = Image.open(image_path)
    w, h = img.size
    
    # Ensure interactive mode is possible
    plt.figure("Query Point Selection")
    plt.imshow(img)
    plt.axis('off')
    plt.title("Click to select points (Left click) | Remove last (Right click) | Finish (Enter/Middle click)")
    print(f"Selecting query points for: {image_path}")
    print("Select points on the reference object, then press Enter to finish.")
    
    pts = plt.ginput(0, timeout=0) # 0: any num of clicks; 0: wait forever
    plt.close()
    
    if not pts:
        print("No point selected, defaulting to center (0.5, 0.5)")
        return [(0.5, 0.5)]
        
    ratios = [(y / h, x / w) for x, y in pts]
    for i, ratio in enumerate(ratios):
        print(f"Selected point: {i+1}: pixel({int(pts[i][0])}, {int(pts[i][1])}) -> Ratio(y={ratio[0]:.3f}, x={ratio[1]:.3f})")
    return ratios

def plot_debiased_similarity(ref_path, tgt_path, query_pt_ratio=[(0.5, 0.5)], device='cuda'):
    """
    Reproduces the similarity map visualization from the INSID3 paper.
    
    Args:
        ref_path: Path to reference image.
        tgt_path: Path to target image.
        query_pt_ratio: List of (y, x) ratios in [0, 1] for the query point in the reference image.
        device: 'cuda' or 'cpu'.
    """
    if not isinstance(query_pt_ratio, list):
        query_pt_ratio = list(query_pt_ratio)
    model = build_insid3()
    model.eval()
    ref_img_pil = Image.open(ref_path).convert('RGB')
    tgt_img_pil = Image.open(tgt_path).convert('RGB')
    ref_w, ref_h = ref_img_pil.size
    tgt_w, tgt_h = tgt_img_pil.size
    model.set_reference(ref_img_pil, torch.zeros([ref_w, ref_h]))
    model.set_target(tgt_img_pil)
    
    sim_map, sim_map_deb = model.get_sim_maps() # list, list
    sim_bias = sim_map[0][0].detach() # -> (ref_patch_h, ref_patch_w, tgt_patch_h, tgt_patch_w)
    sim_deb = sim_map_deb[0][0].detach() # -> (ref_patch_h, ref_patch_w, tgt_patch_h, tgt_patch_w)
    ref_patch_h, ref_patch_w, _, _ = sim_deb.shape
    query_heatmaps_deb = []
    query_heatmaps_bias = []
    for ry_ratio, rx_ratio in query_pt_ratio:
        ry, rx = int(ry_ratio * ref_patch_h), int(rx_ratio * ref_patch_w)
        query_heatmaps_bias.append(sim_bias[ry, rx]) # shape (tgt_patch_h, tgt_patch_w)
        query_heatmaps_deb.append(sim_deb[ry, rx])   # shape (tgt_patch_h, tgt_patch_w)
    # sim map mean
    query_heatmaps_bias = torch.stack(query_heatmaps_bias).mean(dim=0)
    query_heatmaps_deb = torch.stack(query_heatmaps_deb).mean(dim=0)
    query_heatmaps_bias = F.interpolate(
        query_heatmaps_bias.unsqueeze(0).unsqueeze(0), 
        size=(tgt_h, tgt_w), 
        mode='bilinear'
    ).squeeze().cpu().numpy()
    query_heatmaps_deb = F.interpolate(
        query_heatmaps_deb.unsqueeze(0).unsqueeze(0), 
        size=(tgt_h, tgt_w), 
        mode='bilinear'
    ).squeeze().cpu().numpy()
  
    # plotting
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    
    # Panel 1: Reference image with query point
    axes[0,0].imshow(ref_img_pil)
    for ry_ratio, rx_ratio in query_pt_ratio:
        axes[0,0].scatter([rx_ratio * ref_w], [ry_ratio * ref_h], c='lime', marker='x', s=200, linewidth=3)
    axes[0,0].set_title(f"Reference Image ({len(query_pt_ratio)} pts)")
    axes[0,0].axis('off')
    
    # Panel 2: Target image
    axes[0,1].imshow(tgt_img_pil)
    axes[0,1].set_title("Target Image")
    axes[0,1].axis('off')
    
    # Panel 3: Original Similarity Map
    im3 = axes[1,0].imshow(query_heatmaps_bias, cmap='magma')
    axes[1,0].set_title("DINOv3 Similarity (Original)")
    axes[1,0].axis('off')
    fig.colorbar(im3, ax=axes[1,0], shrink=0.6)
    
    # Panel 4: Debiased Similarity Map
    im4 = axes[1,1].imshow(query_heatmaps_deb, cmap='magma')
    axes[1,1].set_title("INSID3 Similarity (Debiased)")
    axes[1,1].axis('off')
    fig.colorbar(im4, ax=axes[1,1], shrink=0.6)
    
    plt.savefig("debiased_similarity_repro.png", bbox_inches='tight', dpi=150)
    print("Saved reproduction plot to debiased_similarity_repro.png")
    plt.show()

if __name__ == "__main__":
    # Example using teapots from test_fig
    # ref = "assets/ref_cat_image.jpg"
    # tgt = "assets/target_cat_image.jpg"
    ref = "./test_fig/teapot_ref.png"
    tgt = "./test_fig/teapot_diff1.png"

    # Interactively select the point
    ratio = select_query_pt(ref)
    
    # Run visualization
    plot_debiased_similarity(ref, tgt, query_pt_ratio=ratio)
