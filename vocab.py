nums = list("0123456789")
lowercase = list("abcdefghijklmnopqrstuvwxyz")
uppercase = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
symbols = list(",./;'[]\`-=<>?:\"{}|~_+!@#$%^&*() ")
special = ["<PAD>", "<SOS>", "<EOS>"]

# All of the tokens in our vocabulary
vocab = nums + lowercase + uppercase + symbols + special

vocab_len = len(vocab)

# The int values of the special tokens
PAD = vocab_len - 3
SOS = vocab_len - 2
EOS = vocab_len - 1