import math
import argparse

input_dims = [5, 3, 5]
sa_arch = [9, 8, 8]
dram_width = 32

def ceil_mod(a, b):
    return int(math.ceil(a / b) * b)

def align_dims(input_dims, sa_arch, dk):
    new_dims = [ceil_mod(input_dims[0], sa_arch[2]), input_dims[1], input_dims[2]]
    return new_dims

def viz_sa_input(input_dims, sa_arch, dram_width):
    dk = dram_width // sa_arch[2]
    new_dims = align_dims(input_dims, sa_arch, dk)
    print(f"input_dims: {input_dims}, sa_arch: {sa_arch}, dram_width: {dram_width} new_dims: {new_dims}")
    table_data = []
    for c in range(new_dims[0] // sa_arch[2]):
        for e in range(ceil_mod(new_dims[1] * new_dims[2], dk) // dk):
            row = []
            for ci in range(sa_arch[2]):
                for ei in range(dk):
                    chan = c * sa_arch[2] + ci
                    elem = e * dk + ei
                    if chan >= input_dims[0] or elem >= input_dims[1] * input_dims[2]:
                        row.append("0")
                    else:
                        row.append(f"c{chan}e{elem}")
            table_data.append(row)
    return table_data

def main():
    parser = argparse.ArgumentParser(description="Visualize aligned dimensions for a given input")
    parser.add_argument('--input_dims', type=int, nargs=3, help="Input dimensions [depth, height, width]")
    parser.add_argument('--sa_arch', type=int, nargs=3, help="Systolic array arch dimensions [rows, cols, N]")
    parser.add_argument('--dram_width', type=int, help="DRAM width")
    args = parser.parse_args()
    table_data = viz_sa_input(args.input_dims, args.sa_arch, args.dram_width)
    for i in table_data:
        for j in i:
            print(f"{j}\t", end='')
        print()

if __name__ == "__main__":
    main()
