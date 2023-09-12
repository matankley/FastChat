import os

from fastchat.model.apply_lora import apply_lora
import transformers
from fastchat.train.train import ModelArguments, TrainingArguments


def cli(model_path):
    from fastchat.serve.cli import chat_loop, SimpleChatIO, GptqConfig

    chat_loop(
        model_path,
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        load_8bit=None,
        cpu_offloading=None,
        conv_template=None,
        temperature=0.7,
        repetition_penalty=1.0,
        max_new_tokens=512,
        chatio=SimpleChatIO(),
        gptq_config=GptqConfig(
            ckpt=None,
            wbits=16,
            groupsize=-1,
            act_order=None,
        ),
        revision="main",
        judge_sent_end=False,
        debug=None,
    )


def inference():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    apply_lora(model_args.model_name_or_path, os.path.join(training_args.output_dir, "merged"),
               training_args.output_dir)

    # generate
    print("==== start chatting ===")
    cli(model_path)
