import configargparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
import numpy as np

def load_args():
    parser = configargparse.ArgumentParser(description="Main")

    # Device configuration
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for computation ('cuda' or 'cpu')")

    # Model configuration
    # parser.add_argument('--ckpt_path', default=None, help="Path to the model checkpoint for loading")
    parser.add_argument('--save_model_path', type=str, help="Path to save the trained model")
    # parser.add_argument('--model_path', type=str, help="Path to the trained model for inference")
    parser.add_argument('--version', type=str, choices=['V1', 'V2'], default='V1', help="Model version to use: 'V1' or 'V2'")

    # Data configuration
    # parser.add_argument('--feat_dim', type=int, default=512, help="Video features dimension (used if loading features directly)")
    parser.add_argument('--videos_root', type=str, help="Root directory of the video dataset")
    # parser.add_argument('--audio_root', type=str, help="Root directory of the audio dataset")
    # parser.add_argument('--file_list', type=str, help="(List of) video file paths relative to videos_root, can also be regex")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker threads for loading data")
    # parser.add_argument('--whisper_model', type=str, default="base", help="Size of the whisper model to use for transcription")

    # Transformer configuration
    # parser.add_argument('--num_blocks', type=int, default=6, help="Number of transformer blocks")
    # parser.add_argument('--hidden_units', type=int, default=512, help="Transformer model size (hidden units)")
    # parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads")
    # parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout probability in the transformer")

    # Preprocessing
    # parser.add_argument('--frame_size', type=int, default=224, help="Resize the input video frames to this resolution")
    # parser.add_argument('--videos_output', type=str, help="Output directory for the preporcessed video dataset")

    # Training and inference parameters
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--sample_size', type=float, default=1, help="Percentage of the dataset to use for training")
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'WER'], help="Mode to run the script: 'train', 'infer' or 'WER'")
    # parser.add_argument('--infer_video_path', type=str, help="Path to the video file for inference")
    # parser.add_argument('--num_classes', type=int, help="Number of classes for the classifier")

    args = parser.parse_args()

    return args

def get_vocab():
    phoneme_vocab = {
        '<pad>': 0, # Blank token for CTC loss       
        'AA': 1,  # Example: odd -> AA D
        'AE': 2,  # Example: at -> AE T
        'AH': 3,  # Example: hut -> HH AH T
        'AO': 4,  # Example: ought -> AO T
        'AW': 5,  # Example: cow -> K AW
        'AY': 6, # Example: hide -> HH AY D
        'B': 7,  # Example: be -> B IY
        'CH': 8, # Example: cheese -> CH IY Z
        'D': 9,  # Example: dee -> D IY
        'DH': 10, # Example: thee -> DH IY
        'EH': 11, # Example: Ed -> EH D
        'ER': 12, # Example: hurt -> HH ER T
        'EY': 13, # Example: ate -> EY T
        'F': 14,  # Example: fee -> F IY
        'G': 15,  # Example: green -> G R IY N
        'HH': 16, # Example: he -> HH IY
        'IH': 17, # Example: it -> IH T
        'IY': 18, # Example: eat -> IY T
        'JH': 19, # Example: gee -> JH IY
        'K': 20,  # Example: key -> K IY
        'L': 21,  # Example: lee -> L IY
        'M': 22,  # Example: me -> M IY
        'N': 23,  # Example: knee -> N IY
        'NG': 24, # Example: ping -> P IH NG
        'OW': 25, # Example: oat -> OW T
        'OY': 26, # Example: toy -> T OY
        'P': 27,  # Example: pee -> P IY
        'R': 28,  # Example: read -> R IY D
        'S': 29,  # Example: sea -> S IY
        'SH': 30, # Example: she -> SH IY
        'T': 31,  # Example: tea -> T IY
        'TH': 32, # Example: theta -> TH EY T AH
        'UH': 33, # Example: hood -> HH UH D
        'UW': 34, # Example: two -> T UW
        'V': 35,  # Example: vee -> V IY
        'W': 36,  # Example: we -> W IY
        'Y': 37,  # Example: yield -> Y IY L D
        'Z': 38,  # Example: zee -> Z IY
        'ZH': 39  # Example: seizure -> S IY ZH ER
    }
    return phoneme_vocab

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.current_step = 0
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, loss=None):
        self.current_step += 1
        
        # During warmup
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (self.current_step / self.warmup_steps)
            self._set_lr(lr)
        else:
            # After warmup, defer to the after_scheduler (if any)
            if self.after_scheduler is not None:
                if loss is not None:
                    self.after_scheduler.step(loss)
                else:
                    self.after_scheduler.step()
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self._last_lr = [lr for _ in self.optimizer.param_groups]  # Update last_lr

    def get_last_lr(self):
        return self._last_lr
    
def log_ctc_inputs_to_csv(logits, labels, input_lengths, target_lengths, filepath="ctc_loss_inputs.csv"):
    """
    Log the values of logits, labels, input lengths, and target lengths to a CSV file.
    """
    # Flatten the logits and labels for easier CSV logging
    logits_flat = logits.detach().cpu().numpy().flatten()
    labels_flat = labels.detach().cpu().numpy().flatten()
    input_lengths_flat = input_lengths.detach().cpu().numpy().flatten()
    target_lengths_flat = target_lengths.detach().cpu().numpy().flatten()

    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Logits"] + list(logits_flat))
        writer.writerow(["Labels"] + list(labels_flat))
        writer.writerow(["Input Lengths"] + list(input_lengths_flat))
        writer.writerow(["Target Lengths"] + list(target_lengths_flat))
        writer.writerow([])  # Add a blank row for separation

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """Generate sampled frame indices for video."""
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices
