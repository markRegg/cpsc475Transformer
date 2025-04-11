from MrML import Vocab

nums = list("0123456789")
lowercase = list("abcdefghijklmnopqrstuvwxyz")
uppercase = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
symbols = list(",./;'[]\`-=<>?:\"{\}|~_+!@#$%^&*() ")
vocab = Vocab(nums + lowercase + uppercase + symbols)