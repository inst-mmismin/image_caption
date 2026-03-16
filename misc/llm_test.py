import os
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = os.path.abspath("./checkpoints/llm")

device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True) 
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

messages = [{"role": "user", "content": "What is gravity?"}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)
print('*'*100)

inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))

print('-'*100)

'''
LlamaForCausalLM(

  (model): LlamaModel(

    (embed_tokens): Embedding(49152, 576, padding_idx=2)

    (layers): ModuleList(

      (0-29): 30 x LlamaDecoderLayer(

        (self_attn): LlamaAttention(

          (q_proj): Linear(in_features=576, out_features=576, bias=False)

          (k_proj): Linear(in_features=576, out_features=192, bias=False)

          (v_proj): Linear(in_features=576, out_features=192, bias=False)

          (o_proj): Linear(in_features=576, out_features=576, bias=False)

        )

        (mlp): LlamaMLP(

          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)

          (up_proj): Linear(in_features=576, out_features=1536, bias=False)

          (down_proj): Linear(in_features=1536, out_features=576, bias=False)

          (act_fn): SiLUActivation()

        )

        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)

        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)

      )

    )

    (norm): LlamaRMSNorm((576,), eps=1e-05)

    (rotary_emb): LlamaRotaryEmbedding()

  )

  (lm_head): Linear(in_features=576, out_features=49152, bias=False)

)
'''