import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from mir_eval.separation import bss_eval_sources

from arguments import ArgParser
from dataset import Music21_dataset
from models import ModelBuilder, activate
from utils import AverageMeter, istft_reconstruction, warpgrid

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame, self.net_motion = nets
        self.crit = crit

    def forward(self, batch_data, args):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['appearance_imags']
        clip_imgs = batch_data['clips_frames']
        gt_masks = batch_data['masks']
        mag_mix = mag_mix + 1e-10
        N = args.num_mix
        weight = torch.ones_like(mag_mix)

        B = mag_mix.size(0)
        T = mag_mix.size(2)
        grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(args.device)
        mag_mix = F.grid_sample(mag_mix.unsqueeze(1), grid_warp, align_corners=False).squeeze()

        mag_mix = torch.log(mag_mix).detach() if args.use_mel else mag_mix

        feat_frames = [None for n in range(N)]
        for n in range(N):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n])
            feat_frames[n] = activate(feat_frames[n], args.img_activation)

        feat_motion = [None for n in range(N)]
        for n in range(N):
            feat_motion[n] = self.net_motion(clip_imgs[n])

        feat_sound = [None for n in range(N)]
        pred_masks = [None for n in range(N)]

        for n in range(N):
            feat_sound[n], _, _ = self.net_sound(mag_mix.to(args.device), feat_motion[n], feat_frames[n])
            pred_masks[n] = activate(feat_sound[n], args.sound_activation)
        for n in range(N):
            grid_warp = torch.from_numpy(warpgrid(B, args.stft_frame // 2 + 1, T, warp=False)).to(args.device)
            pred_masks[n] = F.grid_sample(pred_masks[n].unsqueeze(1), grid_warp, align_corners=False).squeeze()

        err = self.crit(pred_masks, gt_masks, weight).reshape(1)

        return err, \
               {'pred_masks': pred_masks, 'gt_masks': gt_masks,
                'mag_mix': mag_mix, 'mags': mags, 'weight': weight}


def main(args):
    builder = ModelBuilder(args)
    net_sound = builder.build_sound(arch=args.arch_sound, fc_dim=args.num_channels, weights=args.weights_sound)
    net_frame = builder.build_frame(arch=args.arch_frame, fc_dim=args.num_channels, pool_type=args.img_pool,
                                    weights=args.weights_frame)
    net_motion = builder.build_pretrained_3Dresnet_50(args.motion_path)

    nets = (net_sound, net_frame, net_motion)
    crit = builder.build_criterion(arch=args.loss)

    dataset_train = Music21_dataset(args, split='train')
    dataset_val = Music21_dataset(args, split='val')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=int(8),
        drop_last=True,
        pin_memory=True)

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(8),
        drop_last=False,
        pin_memory=True)

    args.epoch_iters = len(dataset_train) // (args.batch_size * args.num_gpus)
    optimizer = _create_optimizer(nets, args)
    netwrapper = NetWrapper(nets, crit)
    netwrapper = nn.parallel.DistributedDataParallel(netwrapper.cuda(), device_ids=[args.local_rank],
                                                     output_device=args.local_rank, find_unused_parameters=True)
    total = sum([param.nelement() if param.requires_grad else 0 for param in netwrapper.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    if args.reuse != 'None':
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        netwrapper.load_state_dict(torch.load(args.reuse, map_location=map_location))

    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}}

    if args.mode == 'eval':
        evaluate(netwrapper, loader_val, history, 0, args)
        print('Evaluation Done!')
        return 0

    for epoch in range(1, args.num_epoch + 1):
        train(netwrapper, loader_train, optimizer, history, epoch, args)
        evaluate(netwrapper, loader_val, history, epoch, args)
        _checkpoint2(netwrapper, history, epoch, args) if args.local_rank == 0 else None
        if epoch in args.lr_steps:
            _adjust_learning_rate(optimizer, args)

    print('Training Done!')


def train(netwrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    netwrapper.train()

    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)
        netwrapper.zero_grad()
        err, out = netwrapper.forward(batch_data, args)
        err = err.sum()
        err.backward()
        optimizer.step()
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          err.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())


def evaluate(netwrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)
    netwrapper.eval()
    loss_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()
    for i, batch_data in enumerate(loader):
        err, outputs = netwrapper.forward(batch_data, args)
        err = err.sum()

        loss_meter.update(err.item())
        print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))
        sdr, sir, sar = _calc_metrics(batch_data, outputs, args)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, '
          ' SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
          .format(epoch, loss_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history, use_mas=args.use_mas)


def _calc_metrics(batch_data, outputs, args):
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']
    pred_masks_ = outputs['pred_masks']
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        pred_masks_linear[n] = pred_masks_[n]

    mag_mix = mag_mix.squeeze().cpu().numpy()
    phase_mix = phase_mix.numpy() if not args.use_mel else phase_mix
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)
    for j in range(B):
        preds_wav = [None for n in range(N)]
        for n in range(N):
            pred_mag = mag_mix[j] * pred_masks_linear[n][j]
            phase = phase_mix[j] if not args.use_mel else None
            preds_wav[n] = istft_reconstruction(pred_mag, phase, use_mel=args.use_mel, hop_length=args.stft_hop,
                                                sr=args.sr, n_fft=args.stft_frame, n_mels=256)  # phase_mix[j, 0]

        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(np.asarray(gts_wav), np.asarray(preds_wav), False)
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())
    return [sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]


def _checkpoint2(net_wrapper, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'
    torch.save(history, '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_wrapper.state_dict(), '{}/net_{}'.format(args.ckpt, suffix_latest))
    current_result = history['val']['sir'][-1] + history['val']['sdr'][-1]
    if current_result > args.best_result:
        print('best: {0}, current: {1}'.format(args.best_result, current_result))
        args.best_result = current_result
        torch.save(net_wrapper.state_dict(), '{}/net_{}'.format(args.ckpt, suffix_best))


def _create_optimizer(nets, args):
    (net_sound, net_frame, net_motion) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound}]
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


def _adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.batch_size_per_gpu
    args.device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    if args.mode == 'train':
        import datetime
        now = datetime.datetime.now()
        time_txt = now.strftime('%m.%d-%H:%M')
        args.id = time_txt if args.discription != 'debug' else args.discription
        args.ckpt = os.path.join(args.ckpt, args.id)
        if args.local_rank == 0:
            makedirs(args.ckpt, remove=True) if not os.path.exists(args.ckpt) else None
            with open(os.path.join(args.ckpt, 'discription.txt'), 'w') as file:
                file.write('Date: {} \nDiscription: {} \n'.format(time_txt, args.discription))
    args.best_result = 0
    main(args)