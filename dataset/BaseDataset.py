import torch
import torch.nn.functional as F
import os
import librosa
import torchaudio
import random
import numpy as np
from torchvision import transforms
from . import video_transforms as vtransforms
from numpy.random import randint
from PIL import Image
import sys

sys.path.append("..")
import utils

warpgrid = utils.warpgrid


class Basedataset(object):
    def __init__(self, arg, split):
        self.path = arg.path
        self.jason_name = ''
        self.split = split
        if self.split == 'train':
            self.jason_name = ''

        else:
            self.jason_name = ''
        self.type_dir = arg.type_dir
        self.fea_length = arg.fea_length
        self.sample_method = arg.sample_method
        self.fixed_interval = arg.fixed_interval
        self.imgSize = arg.imgSize

        self.use_mel = arg.use_mel
        self.split = split
        self.log_freq = arg.log_freq
        self.stft_length = arg.stft_length
        self.stft_hop = arg.stft_hop
        self.sr = arg.sr
        self.wave_length = arg.wave_length
        self.binary_mask = arg.binary_mask

        self.mix_num = arg.num_mix
        self.data_path = ''
        self.classes = [] # classes of your dataset.

        self.device = arg.device
        self.seed = arg.seed

    def __len__(self):
        if self.split == 'train':
            data_len = int(102400)
        else:
            data_len = int(256)

        return data_len

    def sample_method1(self, num_mix, index):
        sample_dict={'class': 'acoustic_guitar', 'video_id': '-4nIqLncdB0', 'clip_id': 1}
        return sample_dict

    def get_appearance_img_path_list(self, sample_dict={'class': 'acoustic_guitar', 'video_id': '-4nIqLncdB0', 'clip_id': 1}):
        subclass = sample_dict['class']
        video_id = sample_dict['video_id']
        clip_id = sample_dict['clip_id']
        app_img_path = os.path.join(self.music_path, subclass, video_id, video_id + '_{}.jpg'.format(clip_id))
        return [app_img_path]

    def get_frames_path_list(self, sample_dict={'class': 'acoustic_guitar', 'video_id': '-4nIqLncdB0', 'clip_id': 1}, appear=False):
        subclass = sample_dict['class']
        video_id = sample_dict['video_id']
        clip_id = sample_dict['clip_id']
        img_path_list = self._get_indices(subclass, video_id, clip_id, appear)
        return img_path_list

    def get_spec_and_mix(self, sample_dicts, sr, wave_length):
        audio_raws = []
        for sample_dict in sample_dicts:
            # load audio file from video
            subclass = sample_dict['class']
            video_id = sample_dict['video_id']
            clip_id = sample_dict['clip_id']
            audio_path = os.path.join(self.music_path, subclass, video_id, video_id + '_{}.mp3'.format(clip_id))
            audio_raw, old_sr = torchaudio.load(audio_path)

            if audio_raw.shape[1] == 2:
                audio_raw = (audio_raw[0, :] + audio_raw[1, :]) / 2
            else:
                audio_raw = audio_raw[0, :]

            audio_raw = torchaudio.transforms.Resample(old_sr, sr)(audio_raw.view(1, -1))

            if audio_raw.shape[1] > wave_length:
                audio_raw = audio_raw[0, 0:wave_length]

            elif audio_raw.shape[1] < wave_length:
                audio_raw = self._suppliment(audio_raw, wave_length)

            if not audio_raw.shape[0] == wave_length:
                assert False

            if self.split == 'train':
                scale = random.random() + 0.5
                audio_raw *= scale
                audio_raw[audio_raw > 1.] = 1.
                audio_raw[audio_raw < -1.] = -1.
            audio_raws.append(audio_raw.squeeze().numpy().astype(np.float32))
        amp_mix, phase_mix, mags_list, phase_list = self._mix_n_and_stft(audio_raws)
        return amp_mix, phase_mix, mags_list, phase_list, audio_raws

    def _suppliment(self, wave, wavelength):
        original_length = wave.shape[1]
        assert original_length < wavelength
        while wave.shape[1] != wavelength:
            wave = torch.cat((wave, wave[:, 0:(wavelength - wave.shape[1])]), dim=1)
        assert wave.shape[1] == wavelength
        return wave.squeeze()

    def load_frames(self, img_path_list, cat='video', image_size=None):
        if image_size is None:
            image_size = self.imgSize
        frames = []
        for img_path in img_path_list:
            img = Image.open(img_path).convert('RGB')
            frames.append(img)
        frames = self._vtransform(frames, image_size, cat)
        return frames

    def load_frame_feature(self, sample_dict):
        fea_path = os.path.join(self.path, self.type_dir, sample_dict['class'], sample_dict['video_id'],
                                sample_dict['video_id'] + '_{}.npy'.format(sample_dict['clip_id']))
        return np.load(fea_path)

    def get_masks(self, mags, mag_mix, mask_thred=1):
        masks = [None for n in range(len(mags))]
        for n in range(len(mags)):
            if self.binary_mask:
                masks[n] = (mags[n] > mask_thred * mag_mix).float().squeeze()
            else:
                masks[n] = (mags[n] / mag_mix).squeeze()
                masks[n].clamp_(0., 5.)
        return masks

    def _mix_n_and_stft(self, audios):
        N = len(audios)
        mags = [None for _ in range(N)]
        phases = [None for _ in range(N)]

        for n in range(N):
            audios[n] /= N
        audio_mix = np.asarray(audios).sum(axis=0)

        amp_mix, phase_mix = self._mel_spec(audio_mix) if self.use_mel else self._stft(audio_mix)
        phase_mix = phase_mix if not self.use_mel else []

        for n in range(N):
            if self.use_mel:
                ampN, phaseN = self._mel_spec(audios[n])
                phases[n] = []
                mags[n] = ampN
            else:
                ampN, phaseN = self._stft(audios[n])
                mags[n] = ampN
                phases[n] = phaseN

        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])
        return amp_mix, phase_mix, mags, phases

    def _stft(self, audio):
        spec = librosa.stft(audio, n_fft=self.stft_length, hop_length=self.stft_hop)  # 返回为(F,T)频谱
        amp = np.abs(spec)
        phase = np.angle(spec)
        amp = torch.from_numpy(amp)
        phase = torch.from_numpy(phase)
        return amp, phase

    def _vtransform(self, x, image_size, cat='video'):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train' and cat != 'video':
            transform_list.append(vtransforms.Resize(int(image_size * 1.1), Image.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(image_size))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(image_size, Image.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(image_size))
        transform_list.append(vtransforms.ToTensor())

        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        tr = transforms.Compose(transform_list)
        x = tr(x)
        return x

    def _get_indices(self, subclass, video, clip_id, appear=False):
        clip_path = os.path.join(self.music_path, subclass, video, video + '_' + str(clip_id))  # URMP
        _, num_img = _count_file_dict(clip_path)
        clip_length = num_img
        t_length = self.fea_length if not appear else 3
        if self.sample_method == 'uniform':
            indices = self._uniform_indices(clip_length, fea_length=t_length)
        else:
            indices = self._get_dense_indices(clip_length, fea_length=t_length)

        imgs_list = [os.path.join(clip_path, 'frame' + str(i) + '.jpg') for i in indices]
        return imgs_list

    def _get_dense_indices(self, clip_length, step=4, fea_length=None):
        if fea_length is None:
            fea_length = self.fea_length
        expanded_sample_length = fea_length * step
        if clip_length >= expanded_sample_length:
            start_pos = randint(clip_length - expanded_sample_length + 1)
            offsets = range(start_pos, start_pos + expanded_sample_length, step)

        elif clip_length > fea_length * (step // 2):
            start_pos = randint(clip_length - fea_length * (step // 2) + 1)
            offsets = range(start_pos, start_pos + fea_length * (step // 2), (step // 2))
        elif clip_length > fea_length:
            start_pos = randint(clip_length - fea_length + 1)
            offsets = range(start_pos, start_pos + fea_length, 1)
        else:
            offsets = np.sort(randint(clip_length, size=fea_length))

        offsets = np.array([int(v) for v in offsets])
        return offsets  # + 1

    def _uniform_indices(self, clip_length, new_length=0, fea_length=None):
        assert fea_length is not None
        average_duration = (clip_length - new_length) // fea_length
        if average_duration > 0:
            interval = randint(average_duration, size=fea_length) if self.fixed_interval == False else randint(
                average_duration, size=1).tolist() * fea_length
            offsets = np.multiply(list(range(fea_length)), average_duration) + np.array(interval)

        elif clip_length > fea_length:
            offsets = np.sort(randint(clip_length - new_length, size=fea_length))

        else:
            offsets = np.zeros((fea_length,))
        return offsets


def _count_file_dict(path):
    items = os.listdir(path)
    num_file = 0
    num_dict = 0
    for item in items:
        subpath = os.path.join(path, item)
        num_file = num_file + 1 if os.path.isfile(subpath) else num_file
        num_dict = num_dict + 1 if os.path.isdir(subpath) else num_dict
    return num_dict, num_file
