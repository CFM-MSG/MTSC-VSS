import random
import numpy as np
from . import BaseDataset
import os
import pandas as pd


# from .base import he_BaseDataset


class Music21_dataset(he_BaseDataset.Basedataset):
    def __init__(self, arg, split):
        super(Music21_dataset, self).__init__(arg, split)
        self.split = split
        if self.split == 'train':
            # self.jason_name = 'MUSIC21_solo_videos_train.json'  # 'MUSIC21_solo_videos_train.json'
            self.get_samples = self.fair_sample_mixing_pair  # fair_sample_mixing_pair sample_dclasses_pairs

        else:
            # self.jason_name = 'MUSIC21_solo_videos_val.json'  # 'MUSIC21_solo_videos_val.json'
            self.get_samples = self.sample_dclasses_pairs

    def __getitem__(self, index):
        np.random.seed(index)
        # Load clip
        sample_dicts = self.get_samples(self.mix_num, index)
        # assert len(sample_dicts) == self.mix_num
        # Load appearance img
        appearance_imags = []
        clips_frames = []
        appearance_feas = []
        for sample_dict in sample_dicts:
            # load clip frames:
            clip_fpaths = self.get_frames_path_list(sample_dict)
            clip_frames = self.load_frames(clip_fpaths, cat='video', image_size=112)
            clips_frames.append(clip_frames)

            # appearance_path = self.get_appearance_img_path_list(sample_dict)
            appearance_path = self.get_frames_path_list(sample_dict, appear=True)
            appearance_img = self.load_frames(appearance_path, cat='img')
            appearance_imags.append(appearance_img)
            # appearance_feas.append(self.load_frame_feature(sample_dict))

        # Load spectrum
        amp_mix, phase_mix, mags_list, phase_list, audios = self.get_spec_and_mix(sample_dicts, self.sr,
                                                                                  self.wave_length)

        # Load mask
        masks = self.get_masks(mags_list, amp_mix, mask_thred=1 / 2)
        masks2 = []
        masks3 = []
        if self.multi_mask:
            masks2 = self.get_masks(mags_list, amp_mix, mask_thred=0.25)
            masks3 = self.get_masks(mags_list, amp_mix, mask_thred=0.15)

        # Packaging
        if self.split == 'train':
            ret_dict = {'mags': mags_list, 'mag_mix': amp_mix,
                        'phases': phase_list,  # audio data
                        'masks': masks,
                        'clips_frames': clips_frames,  # clip data
                        'appearance_imags': appearance_imags, 'masks2': masks2, 'masks3': masks3}  # appearance data
        else:
            ret_dict = {'audios': audios, 'mags': mags_list, 'mag_mix': amp_mix, 'phase_mix': phase_mix,
                        'phases': phase_list,  # audio data
                        'masks': masks,
                        'clips_frames': clips_frames,  # clip data
                        'appearance_imags': appearance_imags, 'sample_dicts': sample_dicts, 'masks2': masks2,
                        'masks3': masks3}  # appearance data
        return ret_dict
