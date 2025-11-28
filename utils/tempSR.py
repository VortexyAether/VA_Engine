"""
Temporal Super Resolution Dataset Library

사용법:
    from temporal_dataset import create_temporal_dataset, TemporalDataset
    
    # 간단 사용
    train_loader, val_loader, test_loader = create_temporal_dataset("data.npy")
    
    # 커스터마이징
    datasets = TemporalDataset.from_file("data.npy", train_ratio=0.7)
    train_loader = datasets.get_train_loader(batch_size=16)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

class TemporalSuperResolutionDataset(Dataset):
    """
    Temporal Super Resolution을 위한 PyTorch Dataset
    temporal_stride에 따라:
    - stride=1: t1, t6 → t2, t3, t4, t5
    - stride=2: t1, t11 → t3, t5, t7, t9
    - stride=3: t1, t16 → t4, t7, t10, t13
    """
    
    def __init__(self, data, sequence_length=6, temporal_stride=1, normalize=True, split_stats=None):
        """
        Parameters:
        data: numpy array (timestep, 1, grid_x, grid_y) 또는 (timestep, grid_x, grid_y)
        sequence_length: 출력할 프레임 수 (기본값: 6, 입력 2개 + 타겟 4개)
        temporal_stride: 프레임 간 간격 (기본값: 1)
        normalize: 정규화 여부
        split_stats: {'mean': float, 'std': float} 전체 데이터의 통계
        """
        self.data = self._validate_data(data)
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride
        self.normalize = normalize
        
        # 실제 필요한 프레임 범위
        self.actual_sequence_length = (sequence_length - 1) * temporal_stride + 1
        
        # 유효한 시작 인덱스 계산
        self.valid_indices = list(range(len(self.data) - self.actual_sequence_length + 1))
        
        # 정규화 통계
        if self.normalize:
            if split_stats is not None:
                self.mean = split_stats['mean']
                self.std = split_stats['std']
            else:
                self.mean = np.mean(self.data)
                self.std = np.std(self.data)
    
    def _validate_data(self, data):
        """데이터 형태 검증 및 변환"""
        if len(data.shape) == 3:
            # (timestep, grid_x, grid_y) -> (timestep, 1, grid_x, grid_y)
            data = data[:, np.newaxis, :, :]
        elif len(data.shape) == 4:
            # (timestep, C, grid_x, grid_y) - C는 임의의 채널 수 가능 (u, v, p 등)
            pass  # 이미 올바른 형태
        else:
            raise ValueError(f"데이터는 3D 또는 4D여야 합니다. 현재: {data.shape}")

        return data
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        # stride를 고려한 시퀀스 추출
        indices = [start_idx + i * self.temporal_stride for i in range(self.sequence_length)]
        sequence = self.data[indices]  # (seq_len, C, H, W)

        # 정규화
        if self.normalize:
            sequence = (sequence - self.mean) / self.std

        # Input/Target 분리
        input_frames = np.stack([sequence[0], sequence[-1]], axis=0)  # (2, C, H, W)
        target_frames = sequence[1:-1]  # (num_target_frames, C, H, W)

        # Input: 2개 프레임의 채널을 concat (2, C, H, W) -> (2*C, H, W)
        input_concat = input_frames.reshape(-1, input_frames.shape[2], input_frames.shape[3])

        # Target: 모든 프레임의 채널을 flatten (num_target_frames, C, H, W) -> (num_target_frames*C, H, W)
        target_concat = target_frames.reshape(-1, target_frames.shape[2], target_frames.shape[3])

        return {
            'input': torch.FloatTensor(input_concat),  # (2*C, H, W)
            'target': torch.FloatTensor(target_concat),  # (num_target_frames*C, H, W)
            'sequence_idx': start_idx,
            'stride': self.temporal_stride,
            'num_frames': target_frames.shape[0],  # 나중에 unflatten하기 위해
            'channels_per_frame': target_frames.shape[1]  # 나중에 unflatten하기 위해
        }
    
    def denormalize(self, normalized_data):
        """정규화 해제"""
        if self.normalize:
            return normalized_data * self.std + self.mean
        return normalized_data
    
    def get_info(self):
        """데이터셋 정보 반환"""
        if len(self) > 0:
            sample = self[0]
            input_shape = sample['input'].shape
            target_shape = sample['target'].shape
        else:
            input_shape = (2, 1, 0, 0)
            target_shape = (4, 1, 0, 0)
        
        return {
            'sequences': len(self),
            'timesteps': self.data.shape[0],
            'grid_size': (self.data.shape[2], self.data.shape[3]),
            'input_shape': input_shape,
            'target_shape': target_shape,
            'normalized': self.normalize,
            'temporal_stride': self.temporal_stride,
            'actual_sequence_length': self.actual_sequence_length
        }

class TemporalDataset:
    """
    Temporal Dataset 관리 클래스
    """
    
    def __init__(self, train_dataset, val_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset  
        self.test_dataset = test_dataset
    

    @classmethod
    def from_file(cls, data, sequence_length=6, temporal_stride=1, normalize=True,
                  train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """
        파일에서 데이터셋 생성

        Parameters:
        data: 데이터 배열
        sequence_length: 시퀀스 길이
        temporal_stride: 프레임 간격 (추가됨)
        normalize: 정규화 여부
        train_ratio, val_ratio, test_ratio: 분할 비율

        Returns:
        TemporalDataset 객체
        """

        # 시간순 분할
        total_timesteps = data.shape[0]
        train_end = int(total_timesteps * train_ratio)
        val_end = int(total_timesteps * (train_ratio + val_ratio))

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        print(f"📊 데이터 분할:")
        print(f"   Train: {train_data.shape[0]} frames ({train_ratio*100:.1f}%)")
        print(f"   Val:   {val_data.shape[0]} frames ({val_ratio*100:.1f}%)")
        print(f"   Test:  {test_data.shape[0]} frames ({test_ratio*100:.1f}%)")

        # 전체 통계 계산
        split_stats = None
        if normalize:
            split_stats = {
                'mean': np.mean(data),
                'std': np.std(data)
            }

        # 데이터셋 생성 - temporal_stride 전달
        train_dataset = TemporalSuperResolutionDataset(
            train_data, sequence_length, temporal_stride, normalize, split_stats
        )
        val_dataset = TemporalSuperResolutionDataset(
            val_data, sequence_length, temporal_stride, normalize, split_stats
        )
        test_dataset = TemporalSuperResolutionDataset(
            test_data, sequence_length, temporal_stride, normalize, split_stats
        )

        print(f"🎯 시퀀스 생성 (stride={temporal_stride}):")
        print(f"   Train: {len(train_dataset)} sequences")
        print(f"   Val:   {len(val_dataset)} sequences")
        print(f"   Test:  {len(test_dataset)} sequences")

        # stride 정보 출력
        actual_length = (sequence_length - 1) * temporal_stride + 1
        print(f"   각 시퀀스는 {actual_length} 프레임 범위를 커버")

        return cls(train_dataset, val_dataset, test_dataset)
   
    
    def get_train_loader(self, batch_size=8, shuffle=True, num_workers=0, **kwargs):
        """Train DataLoader 반환"""
        return DataLoader(self.train_dataset, batch_size=batch_size, 
                         shuffle=shuffle, num_workers=num_workers, **kwargs)
    
    def get_val_loader(self, batch_size=8, shuffle=False, num_workers=0, **kwargs):
        """Validation DataLoader 반환"""
        return DataLoader(self.val_dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers, **kwargs)
    
    def get_test_loader(self, batch_size=8, shuffle=False, num_workers=0, **kwargs):
        """Test DataLoader 반환"""
        return DataLoader(self.test_dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers, **kwargs)
    
    def get_all_loaders(self, batch_size=8, num_workers=0, **kwargs):
        """모든 DataLoader 반환"""
        train_loader = self.get_train_loader(batch_size, True, num_workers, **kwargs)
        val_loader = self.get_val_loader(batch_size, False, num_workers, **kwargs)
        test_loader = self.get_test_loader(batch_size, False, num_workers, **kwargs)
        return train_loader, val_loader, test_loader
    
    def visualize_sample(self, split='train', sample_idx=0, figsize=(15, 8)):
        """샘플 시각화"""
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'val':
            dataset = self.val_dataset
        elif split == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError("split은 'train', 'val', 'test' 중 하나여야 합니다.")
        
        if len(dataset) == 0:
            print(f"⚠️  {split} 데이터셋이 비어있습니다.")
            return
        
        if sample_idx >= len(dataset):
            sample_idx = 0
            warnings.warn(f"sample_idx가 범위를 벗어나 0으로 설정했습니다.")
        
        sample = dataset[sample_idx]
        input_frames = sample['input'].numpy()
        target_frames = sample['target'].numpy()
        
        # 정규화 해제
        if dataset.normalize:
            input_frames = dataset.denormalize(input_frames)
            target_frames = dataset.denormalize(target_frames)
        
        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'{split.capitalize()} Sample {sample_idx}')
        
        # Input frames
        im1 = axes[0, 0].imshow(input_frames[0, 0], cmap='RdBu', origin='lower')
        axes[0, 0].set_title('Input: t1')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(input_frames[1, 0], cmap='RdBu', origin='lower')
        axes[0, 1].set_title('Input: t6')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Difference
        diff = input_frames[1, 0] - input_frames[0, 0]
        im3 = axes[0, 2].imshow(diff, cmap='RdBu', origin='lower')
        axes[0, 2].set_title('Difference (t6-t1)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Target frames (일부만)
        target_indices = [0, 1, 3]  # t2, t3, t5
        target_labels = ['t2', 't3', 't5']
        
        for i, (idx, label) in enumerate(zip(target_indices, target_labels)):
            im = axes[1, i].imshow(target_frames[idx, 0], cmap='RdBu', origin='lower')
            axes[1, i].set_title(f'Target: {label}')
            plt.colorbar(im, ax=axes[1, i])
        
        plt.tight_layout()
        plt.show()
        
        # 통계 출력
        print(f"\n📊 {split.capitalize()} Sample {sample_idx} 통계:")
        print(f"   Input range:  {input_frames.min():.4f} ~ {input_frames.max():.4f}")
        print(f"   Target range: {target_frames.min():.4f} ~ {target_frames.max():.4f}")
    
    def get_info(self):
        """전체 데이터셋 정보"""
        train_info = self.train_dataset.get_info()
        val_info = self.val_dataset.get_info()
        test_info = self.test_dataset.get_info()
        
        return {
            'train': train_info,
            'val': val_info,
            'test': test_info,
            'total_sequences': train_info['sequences'] + val_info['sequences'] + test_info['sequences']
        }
    
    def print_info(self):
        """정보 출력"""
        info = self.get_info()
        
        print("="*50)
        print("📋 Temporal Super Resolution Dataset Info")
        print("="*50)
        print(f"🎯 Task: t1, t6 → t2, t3, t4, t5")
        print(f"📊 Total sequences: {info['total_sequences']}")
        
        for split in ['train', 'val', 'test']:
            split_info = info[split]
            print(f"\n{split.capitalize()}:")
            print(f"   Sequences: {split_info['sequences']}")
            print(f"   Timesteps: {split_info['timesteps']}")
            print(f"   Grid size: {split_info['grid_size']}")
            if split_info['sequences'] > 0:
                print(f"   Input shape: {split_info['input_shape']}")
                print(f"   Target shape: {split_info['target_shape']}")

# 편의 함수들

def create_temporal_dataset(data_file, batch_size=8, sequence_length=6, 
                           temporal_stride=1, normalize=True,
                           train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                           num_workers=0, **kwargs):
    """
    Parameters 추가:
    temporal_stride: 프레임 간격 (1, 2, 3, ...)
        - stride=1: 연속 프레임 (t1, t2, t3, ...)
        - stride=2: 한 칸 띄워서 (t1, t3, t5, ...)
        - stride=3: 두 칸 띄워서 (t1, t4, t7, ...)
    """
    print(f"🚀 Temporal Super Resolution 데이터셋 생성 (stride={temporal_stride})")
    
    if temporal_stride == 1:
        print(f"   Input: t1, t{sequence_length}")
        print(f"   Target: t2 ~ t{sequence_length-1}")
    else:
        end_frame = 1 + (sequence_length - 1) * temporal_stride
        target_frames = [1 + (i+1) * temporal_stride for i in range(sequence_length - 2)]
        print(f"   Input: t1, t{end_frame}")
        print(f"   Target: t{target_frames}")
    
    # 데이터셋 생성 - from_file 메서드도 수정 필요
    datasets = TemporalDataset.from_file(
        data_file, sequence_length, temporal_stride, normalize,
        train_ratio, val_ratio, test_ratio
    )
    
    # DataLoader 생성
    train_loader, val_loader, test_loader = datasets.get_all_loaders(
        batch_size=batch_size, num_workers=num_workers, **kwargs
    )
    
    print(f"✅ DataLoader 생성 완료 (batch_size={batch_size})")
    return train_loader, val_loader, test_loader


def quick_visualize(data_file, split='train', sample_idx=0, **dataset_kwargs):
    """
    빠른 시각화를 위한 편의 함수
    
    Parameters:
    data_file: 데이터 파일 경로
    split: 'train', 'val', 'test'
    sample_idx: 샘플 인덱스
    **dataset_kwargs: TemporalDataset.from_file에 전달할 인자
    """
    datasets = TemporalDataset.from_file(data_file, **dataset_kwargs)
    datasets.visualize_sample(split, sample_idx)
    return datasets

# 버전 정보
def get_version():
    """라이브러리 버전 반환"""
    return __version__

def print_usage():
    """사용법 출력"""
    print("""
🔥 Temporal Dataset Library 사용법

1️⃣ 간단 사용:
   from temporal_dataset import create_temporal_dataset
   
   train_loader, val_loader, test_loader = create_temporal_dataset("data.npy")

2️⃣ 커스터마이징:
   from temporal_dataset import TemporalDataset
   
   datasets = TemporalDataset.from_file("data.npy", train_ratio=0.7)
   train_loader = datasets.get_train_loader(batch_size=16)
   datasets.visualize_sample('train', 0)
   datasets.print_info()

3️⃣ 빠른 시각화:
   from temporal_dataset import quick_visualize
   
   quick_visualize("data.npy", split='train')

📚 주요 기능:
   - 6:2:2 시간순 데이터 분할
   - t1,t6 → t2,t3,t4,t5 temporal super resolution
   - 자동 정규화 (전체 데이터 통계 사용)
   - 시각화 및 정보 출력
   - PyTorch DataLoader 자동 생성
""")

if __name__ == "__main__":
    print_usage()
    print(f"\n📦 Temporal Dataset Library v{__version__}")
    print(f"👨‍💻 Created by {__author__}")
