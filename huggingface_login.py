import sys
from huggingface_hub import login

token = sys.argv[1] if len(sys.argv) > 1 else input("Enter HF token: ")
login(token=token)
