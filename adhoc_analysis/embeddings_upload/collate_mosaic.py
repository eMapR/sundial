import os
import sys
import glob
import torch
import argparse
import torchvision.transforms.v2 as v2
import torch.nn.functional as F


def collate(embed, output_size, kernel_size, stride, count=None):
    G, G, D, T, H, W = embed.shape
    Hk, Wk = kernel_size
    Ho, Wo = output_size
    
    embed = embed.reshape(G*G, D*T, H, W)
    embed = torch.nn.functional.interpolate(embed, size=kernel_size).reshape(G*G, D*T, Hk*Wk)
    embed = embed.permute(2, 1, 0).reshape(D*Hk*Wk, G*G)
    embed = F.fold(embed, output_size=output_size, kernel_size=kernel_size, stride=stride)
    embed = embed.reshape(D, Ho, Wo)
    if count is None:
        return embed
    else:
        return v2.functional.center_crop(embed/count, kernel_size)


def main(base_path, out_path, pattern, output_size=232, kernel_size=224, stride=1, nodes=1, devices=1, rank=1):
    output_size = (output_size, output_size)
    kernel_size = (kernel_size, kernel_size)
    stride = (stride, stride)
    
    pattern = os.path.join(base_path, pattern)
    matching_files = sorted(glob.glob(pattern))
    num_files = len(matching_files)
    
    files_per_node = (num_files + nodes - 1) // nodes
    start_idx = rank * files_per_node
    end_idx = min(start_idx + files_per_node, num_files)
    
    for i in range(start_idx, end_idx):
        file = matching_files[i]
        file_name = os.path.basename(file)
        print(f'Stitching {file_name} on node {rank}')
        embed = torch.load(file, map_location="cuda:0")
        if i == start_idx:
            count = torch.ones_like(embed)
            count = collate(embed, output_size, kernel_size, stride)
        embed = collate(embed, output_size, kernel_size, stride, count=count)
        torch.save(embed, os.path.join(out_path, file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed stitching script")
    parser.add_argument("base_path", type=str, help="Path to input files")
    parser.add_argument("out_path", type=str, help="Path to output directory")
    parser.add_argument("pattern", type=str, help="File pattern to match")
    parser.add_argument("output_size", type=int)
    parser.add_argument("kernel_size", type=int)
    parser.add_argument("stride", type=int)
    parser.add_argument("nodes", type=int, help="Total number of nodes")
    parser.add_argument("devices", type=int, help="Total number of devices")
    parser.add_argument("rank", type=int, help="Current node rank")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_path, exist_ok=True)
    main(args.base_path,
         args.out_path,
         args.pattern,
         args.output_size,
         args.kernel_size,
         args.stride,
         args.nodes,
         args.devices,
         args.rank)