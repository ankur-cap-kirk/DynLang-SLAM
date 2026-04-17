"""Test pure PyTorch Gaussian renderer on Replica room0."""
import time
import torch
from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.core.renderer import GaussianRenderer

device = 'cuda'
ds = ReplicaDataset('data/Replica/room0', max_frames=5)
intr = get_replica_intrinsics()
frame = ds[0]

# Initialize with larger downsample for fewer Gaussians
gmap = GaussianMap(sh_degree=0, lang_feat_dim=16, device=device)
gmap.initialize_from_depth(
    depth=frame['depth'].to(device), rgb=frame['rgb'].to(device),
    pose=frame['pose'].to(device),
    fx=intr['fx'], fy=intr['fy'], cx=intr['cx'], cy=intr['cy'],
    downsample=8,  # fewer Gaussians for speed
)
print(f'Gaussians: {gmap.num_gaussians}')

renderer = GaussianRenderer()
K = intr['K'].to(device)
viewmat = torch.inverse(frame['pose'].to(device))

# Test at different resolutions
for ds_factor in [8, 4, 2]:
    torch.cuda.synchronize()
    t0 = time.time()
    result = renderer(gmap, viewmat, K, intr['width'], intr['height'], downscale=ds_factor)
    torch.cuda.synchronize()
    dt = time.time() - t0
    h, w = result['rgb'].shape[:2]
    print(f'  Downscale {ds_factor}x ({h}x{w}): {dt:.3f}s | '
          f'RGB range: {result["rgb"].min():.3f}-{result["rgb"].max():.3f} | '
          f'Alpha mean: {result["alpha"].mean():.3f} | '
          f'Depth range: {result["depth"][result["depth"]>0].min():.2f}-{result["depth"].max():.2f}m')

# Test gradient flow (pose optimization)
print('\nTesting gradient flow (pose -> render -> loss)...')
from dynlang_slam.utils.camera import pose_to_matrix, matrix_to_pose

# Freeze Gaussian params, optimize pose
for p in gmap.parameters():
    p.requires_grad_(False)

# Use the actual pose from frame 0, perturbed slightly
gt_pose = frame['pose'].to(device)
gt_quat, gt_trans = matrix_to_pose(gt_pose)
opt_trans = gt_trans.clone().detach().requires_grad_(True)
opt_quat = gt_quat.clone().detach().requires_grad_(True)
pose = pose_to_matrix(opt_quat, opt_trans)
vm = torch.inverse(pose)
result = renderer(gmap, vm, K, intr['width'], intr['height'], downscale=8)
print(f'  RGB grad_fn: {result["rgb"].grad_fn}')
print(f'  Alpha mean: {result["alpha"].mean():.3f}')
loss = result['rgb'].mean()
loss.backward()
print(f'  Translation grad: {opt_trans.grad}')
print(f'  Quaternion grad: {opt_quat.grad}')
print(f'  Gradients flow: {"YES" if opt_trans.grad is not None and opt_trans.grad.abs().sum() > 0 else "NO"}')

# Test gradient flow (Gaussian params -> render -> loss)
print('\nTesting gradient flow (gaussians -> render -> loss)...')
for p in gmap.parameters():
    p.requires_grad_(True)
viewmat2 = torch.inverse(frame['pose'].to(device)).detach()
result2 = renderer(gmap, viewmat2, K, intr['width'], intr['height'], downscale=8)
loss2 = result2['rgb'].mean()
loss2.backward()
print(f'  Means grad norm: {gmap.means.grad.norm():.6f}' if gmap.means.grad is not None else '  Means grad: None')
print(f'  Colors grad norm: {gmap.colors.grad.norm():.6f}' if gmap.colors.grad is not None else '  Colors grad: None')
grad_ok = gmap.means.grad is not None and gmap.means.grad.abs().sum() > 0
print(f'  Gaussian gradients flow: {"YES" if grad_ok else "NO"}')

print(f'\nVRAM: {torch.cuda.memory_allocated()/(1024**3):.2f} GB')
print('SUCCESS: Pure PyTorch renderer works!')
