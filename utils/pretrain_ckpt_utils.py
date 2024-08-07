import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputckpt', type=str, required=True)
    parser.add_argument('--outputckpt', type=str, required=True)
    parser.add_argument('--pop_name_list',
                        type=str,
                        help='split by ,',
                        default='none')
    parser.add_argument('--template',
                        type=str,
                        help='template for ckpt pop list',
                        default='none')
    parser.add_argument('--template_args',
                        action='store_true',
                        help='if true, copy template args to ckpt')

    args = parser.parse_args()

    if args.pop_name_list == 'none':
        pop_list = [
            'encoder.lm_head_transform_weight.weight',
            'encoder.lm_head_transform_weight.bias',
            'encoder.layer_norm1.weight', 'encoder.layer_norm1.bias'
        ]


# 'encoder.layer_norm.weight', 'encoder.layer_norm.bias'
    ckpt = torch.load(args.inputckpt)

    temp_args = False
    if args.template != 'none':
        template = torch.load(args.template)
        if args.template_args:
            temp_args = True
        pop_list = []
        for k in ckpt['model'].keys():
            if k not in template['model'].keys():
                pop_list.append(k)
            # elif ckpt['model'][k].shape != template['model'][k].shape:
            #     print(k, ckpt['model'][k].shape, template['model'][k].shape)
        for k in template['model'].keys():
            if k not in ckpt['model'].keys():
                print(f'warning!! {k} not in ckpt')

    ckpt['model']['encoder.embed_tokens.weight'] = ckpt['model'][
        'encoder.embed_tokens.weight'][:-1, :]
    ckpt['model']['decoder.embed_tokens.weight'] = ckpt['model'][
        'decoder.embed_tokens.weight'][:-1, :]
    ckpt['model']['decoder.output_projection.weight'] = ckpt['model'][
        'decoder.output_projection.weight'][:-1, :]

    for k in pop_list:
        if k in ckpt['model']:
            ckpt['model'].pop(k)
        else:
            print(f'warning!!! {k} not in ckpt')
    if temp_args:
        template['model'] = ckpt['model']
        torch.save(template, args.outputckpt)
    else:
        torch.save(ckpt, args.outputckpt)

if __name__ == '__main__':
    main()
