from MrML import Vocab, ModelInfo
import torch

nums = list("0123456789")
lowercase = list("abcdefghijklmnopqrstuvwxyz")
uppercase = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
symbols = list(",./;'[]\`-=<>?:\"{\}|~_+!@#$%^&*() ")
vocab = Vocab(nums + lowercase + uppercase + symbols)

BATCH_SIZE = 64
SEQ_LEN = 64
D_MODEL = 8
STRIDE = SEQ_LEN // 2
N_LAYERS = 1
N_HEADS = 1

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_float32_matmul_precision('medium')
print(f"Using Device: {device}")

info = ModelInfo(BATCH_SIZE, SEQ_LEN, D_MODEL, STRIDE, vocab, device)
