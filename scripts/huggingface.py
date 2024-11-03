from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B", use_auth_token='hhf_lkGDdLpJfzorSNmxuQRHAcHapEzCoqvsPg')
result = pipe("What is the meaning of life?")
