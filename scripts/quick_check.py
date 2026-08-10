# quick_check.py
from src.utils import load_config, build_conversations
from unsloth import FastVisionModel

config = load_config("configs/config.yaml")
model, tokenizer = FastVisionModel.from_pretrained(config["base_model"], load_in_4bit=True)
from unsloth.trainer import UnslothVisionDataCollator
collator = UnslothVisionDataCollator(
    model,
    tokenizer,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
convos = build_conversations(config)
batch = collator([convos[0]])
labels = batch["labels"][0]
unmasked = labels[labels != -100]
print(tokenizer.decode(unmasked))