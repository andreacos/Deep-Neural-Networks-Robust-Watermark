import torch
import json
import sys
import numpy as np
from tabulate import tabulate
np.set_printoptions(precision=2)


def lorem_ipsum(length=256, bit_encode=8):

    # Repeating last block of text to reach 16384 bit length for larger embedding setups
    lorem = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore " \
            "et dolore magna aliqua Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut " \
            "aliquip ex ea commodo consequat Duis aute irure dolor in reprehenderit in voluptate velit esse " \
            "cillum dolore eu fugiat nulla pariatur Excepteur sint occaecat cupidatat non proident sunt in culpa " \
            "qui officia deserunt mollit anim id est laborum"\
            "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae " \
            "Nulla nec lacus eu tellus porta lobortis Aliquam faucibus maximus arcu nec pellentesque " \
            "Cras mattis dolor ut vestibulum consequat turpis quam elementum ligula ut vulputate dui " \
            "sem id lectus Sed convallis iaculis mi ac ultricies sapien efficitur quis Quisque tempor " \
            "mollis neque eu rhoncus Praesent quis nunc commodo viverra leo vel aliquam magna"\
            "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae " \
            "Nulla nec lacus eu tellus porta lobortis Aliquam faucibus maximus arcu nec pellentesque " \
            "Cras mattis dolor ut vestibulum consequat turpis quam elementum ligula ut vulputate dui " \
            "sem id lectus Sed convallis iaculis mi ac ultricies sapien efficitur quis Quisque tempor " \
            "mollis neque eu rhoncus Praesent quis nunc commodo viverra leo vel aliquam magna"\
            "Nulla nec lacus eu tellus porta lobortis Aliquam faucibus maximus arcu nec pellentesque " \
            "Cras mattis dolor ut vestibulum consequat turpis quam elementum ligula ut vulputate dui " \
            "sem id lectus Sed convallis iaculis mi ac ultricies sapien efficitur quis Quisque tempor " \
            "mollis neque eu rhoncus Praesent quis nunc commodo viverra leo vel aliquam magna" \
            "mollis neque eu rhoncus Praesent quis nunc commodo viverra leo vel aliquam magna" \
            "Nulla nec lacus eu tellus porta lobortis Aliquam faucibus maximus arcu nec pellentesque " \
            "Cras mattis dolor ut vestibulum consequat turpis quam elementum ligula ut vulputate dui " \
            "sem id lectus Sed convallis iaculis mi ac ultricies sapien efficitur quis Quisque tempor " \
            "mollis neque eu rhoncus Praesent quis nunc commodo viverra leo vel aliquam magna"

    return lorem[:int(length/bit_encode)]


def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def frombits(bits):
    chars = []
    for b in range(int(len(bits) / 8)):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


def create_watermark_configuration(model, out_json='config.json'):
    data = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            layer_weights = param.view(-1)
            data[name] = {"std": torch.sqrt(torch.var(layer_weights)).item(),
                          "var": torch.var(layer_weights).item(),
                          "idx": i}

    with open(out_json, "w") as outfile:
        json.dump(data, outfile)

    return


def print_redirect(file, *args):
    temp = sys.stdout # assign console output to a variable
    print(' '.join([str(arg) for arg in args]) )
    sys.stdout = file
    print(' '.join([str(arg) for arg in args]))
    sys.stdout = temp # set stdout back to console output


def top_n_accuracy(output, labels, n=5):
    topN = 0
    for i in range(len(output)):
        i_pred = output[i]
        top_values = (-i_pred).argsort()[:n]
        if labels[i] in top_values:
            topN += 1.0
    return topN


def weights_summary(model):
    layer_names, layer_shapes, layer_numels, layer_variance, layer_stddev, indices = [], [], [], [], [], []
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            weights = param.view(-1).cpu().detach().numpy()
            layer_names.append(name)
            layer_shapes.append(str(param.size()))
            layer_numels.append(param.numel())
            layer_variance.append(np.var(weights))
            layer_stddev.append(np.sqrt(np.var(weights)))
            indices.append(i)

    # Gather data for output
    headers = ["L_index", "L_Name", "L_shape", "L_count", "L_variance", "L_std_dev"]

    table = zip(indices, layer_names, layer_shapes, layer_numels, layer_variance, layer_stddev)

    # Print data
    return tabulate(table, headers=headers)
