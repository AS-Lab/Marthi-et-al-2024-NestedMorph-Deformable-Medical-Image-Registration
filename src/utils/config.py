import torch

if torch.cuda.is_available():
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPUs:', GPU_num)
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    print('Currently using:', torch.cuda.get_device_name(GPU_iden))
    device = torch.device('cuda:' + str(GPU_iden))
else:
    print('GPU not available, using CPU.')
    device = torch.device('cpu')
