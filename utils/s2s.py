import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os

class SnapshotDataset(Dataset):
    """
    Snapshot to snapshot prediction dataset
    디렉토리 구조: 
    - data_dir/
        - X.npy (input snapshots)
        - Y.npy (target snapshots)
    """
    
    def __init__(self, data_dir, normalize=True, transform=None, condition='X.npy', target='Y.npy'):
        """
        Parameters:
        data_dir: 데이터 디렉토리 경로
        normalize: 정규화 여부
        transform: 추가 변환 함수
        condition: 조건 데이터 파일명 (None이면 unconditional)
        target: 타겟 데이터 파일명
        """
        self.data_dir = Path(data_dir)
        self.has_condition = condition is not None

        # 파일 로드
        y_path = self.data_dir / target

        if not y_path.exists():
            raise FileNotFoundError(f"{target} not found in {data_dir}")

        self.Y = np.load(y_path).astype(np.float32)

        # Condition이 있는 경우에만 X 로드
        if self.has_condition:
            x_path = self.data_dir / condition
            if not x_path.exists():
                raise FileNotFoundError(f"{condition} not found in {data_dir}")
            self.X = np.load(x_path).astype(np.float32)

            # 데이터 형태 검증
            assert self.X.shape[0] == self.Y.shape[0], \
                f"Sample count mismatch: X={self.X.shape[0]}, Y={self.Y.shape[0]}"

            # 3D -> 4D 변환 (채널 차원 추가)
            if len(self.X.shape) == 3:
                self.X = self.X[:, np.newaxis, :, :]
        else:
            self.X = None

        # Y의 3D -> 4D 변환
        if len(self.Y.shape) == 3:
            self.Y = self.Y[:, np.newaxis, :, :]

        self.normalize = normalize
        self.transform = transform

        # 정규화 통계
        if self.normalize:
            if self.has_condition:
                self.x_mean = np.mean(self.X)
                self.x_std = np.std(self.X)
                # 정규화 적용
                self.X = (self.X - self.x_mean) / (self.x_std + 1e-8)

            self.y_mean = np.mean(self.Y)
            self.y_std = np.std(self.Y)
            self.Y = (self.Y - self.y_mean) / (self.y_std + 1e-8)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        y = self.Y[idx]

        if self.has_condition:
            x = self.X[idx]
            if self.transform:
                x = self.transform(x)
        else:
            # Unconditional: return empty placeholder
            x = np.zeros_like(y)

        if self.transform:
            y = self.transform(y)

        return {
            'input': torch.FloatTensor(x),
            'target': torch.FloatTensor(y),
            'index': idx
        }
    
    def denormalize(self, x, y=None):
        """정규화 해제"""
        if not self.normalize:
            return x, y
            
        x_denorm = x * self.x_std + self.x_mean
        y_denorm = y * self.y_std + self.y_mean if y is not None else None
        
        return x_denorm, y_denorm

def create_snapshot_dataloaders(data_dir, batch_size=32,
                               train_ratio=0.7, val_ratio=0.15,
                               data_ratio=1.0,
                               normalize=True, num_workers=0, condition='X.npy', target='Y.npy'):
    """
    Train/Val/Test 데이터로더 생성 (Random split with data subsampling)

    Parameters:
    data_dir: 데이터 디렉토리
    batch_size: 배치 크기
    train_ratio: 학습 데이터 비율 (선택된 데이터 중)
    val_ratio: 검증 데이터 비율 (선택된 데이터 중)
    data_ratio: 전체 데이터 중 사용할 비율 (0.0-1.0)
    normalize: 정규화 여부
    num_workers: 데이터로더 워커 수

    Returns:
    train_loader, val_loader, test_loader, full_dataset
    """
    # 전체 데이터셋 로드
    full_dataset = SnapshotDataset(data_dir, normalize=normalize, condition=condition, target=target)

    # Step 1: 전체 데이터셋에서 data_ratio만큼 랜덤 선택
    n_total_samples = len(full_dataset)
    n_selected = int(n_total_samples * data_ratio)

    # Random indices 생성 (고정된 seed로 재현 가능)
    import random
    random.seed(42)
    np.random.seed(42)

    all_indices = list(range(n_total_samples))
    random.shuffle(all_indices)
    selected_indices = all_indices[:n_selected]

    print(f"Dataset subsampling: {n_selected}/{n_total_samples} samples ({data_ratio*100:.1f}%)")

    # Step 2: 선택된 데이터를 train/val/test로 랜덤 분할
    n_train = int(n_selected * train_ratio)
    n_val = int(n_selected * val_ratio)
    n_test = n_selected - n_train - n_val

    # Random split
    random.shuffle(selected_indices)
    train_indices = selected_indices[:n_train]
    val_indices = selected_indices[n_train:n_train + n_val]
    test_indices = selected_indices[n_train + n_val:]
    
    # Subset 생성
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded from {data_dir}")
    print(f"Total samples: {n_total_samples}")
    print(f"Selected samples: {n_selected} ({data_ratio*100:.1f}%)")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    if full_dataset.has_condition:
        print(f"Input shape: {full_dataset.X.shape}")
    else:
        print(f"Mode: Unconditional (no input)")
    print(f"Target shape: {full_dataset.Y.shape}")
    
    return train_loader, val_loader, test_loader, full_dataset

# 사용 예시
if __name__ == "__main__":
    data_dir = "/path/to/your/data"
    train_loader, val_loader, test_loader, dataset = create_snapshot_dataloaders(
        data_dir,
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # 샘플 확인
    batch = next(iter(train_loader))
    print(f"Batch input shape: {batch['input'].shape}")
    print(f"Batch target shape: {batch['target'].shape}")
