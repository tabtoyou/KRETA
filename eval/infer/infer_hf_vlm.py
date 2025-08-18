import base64
import io
import json
import os
import re
import sys
from typing import Dict, Tuple

from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    LlavaOnevisionForConditionalGeneration,
)
from tqdm import tqdm


# Usage:
#   python eval/infer/infer_hf_vlm.py <MODEL_ID> <SETTING>
# Examples:
#   python eval/infer/infer_hf_vlm.py naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B default
#   python eval/infer/infer_hf_vlm.py kakaocorp/kanana-1.5-v-3b-instruct default
#   python eval/infer/infer_hf_vlm.py NCSOFT/VARCO-VISION-14B default
#   python eval/infer/infer_hf_vlm.py skt/A.X-4.0-VL-Light default


if len(sys.argv) == 3:
    MODEL_ID = sys.argv[1]
    SETTING = sys.argv[2]
else:
    print(
        "Usage: python eval/infer/infer_hf_vlm.py <MODEL_ID> <SETTING>\n",
        "Example: python eval/infer/infer_hf_vlm.py naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B default",
    )
    MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
    SETTING = "default"


# Set number of workers: for GPU VLMs, prefer single-threaded generation
WORKERS = 1

prompt_config = {
    "default": "Please select the correct answer from the options above. The last line of your response should be of the following format: 'Answer: LETTER' (without quotes) where LETTER is one of options.",
    "direct": "Answer directly with the option letter from the given choices.",
}


def parse_options(doc: Dict) -> str:
    option_letters = ["A", "B", "C", "D"]
    options = [doc[letter] for letter in option_letters]
    return "\n".join([f"{l}. {o}" for l, o in zip(option_letters, options)])


def construct_prompt(doc: Dict) -> str:
    question = doc["question"]
    parsed_options = parse_options(doc)
    return f"Question: {question}\nOptions:\n{parsed_options}\n{prompt_config[SETTING]}"


def b64_to_pil_image(b64_str: str) -> Image.Image:
    # Accept PIL.Image directly for robustness
    if isinstance(b64_str, Image.Image):
        return b64_str.convert("RGB")
    # Some datasets include pure base64 without header
    if b64_str.startswith("data:image"):
        b64_data = b64_str.split(",", 1)[1]
    else:
        b64_data = b64_str
    image_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def select_strategy(model_id: str) -> str:
    lower_id = model_id.lower()
    if "hyperclovax" in lower_id:
        return "hyperclovax"
    if "kanana" in lower_id:
        return "kanana"
    if "varco-vision-2.0" in lower_id or ("varco-vision" in lower_id and "2.0" in lower_id):
        return "varco2_llavaov"
    if "a.x-4.0-vl-light" in lower_id:
        return "ax_vl_light"
    # default strategy should work for many LLaVA/Qwen-VL like models
    return "default"


class HFVLMInference:
    def __init__(self, model_id: str, device: str = None, dtype: torch.dtype = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)

        self.strategy = select_strategy(model_id)

        if self.strategy == "kanana":
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
        elif self.strategy == "varco2_llavaov":
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else self.dtype,
                attn_implementation="sdpa",
                device_map="auto" if self.device == "cuda" else None,
            )
        elif self.strategy == "ax_vl_light":
            # A.X 4.0 VL Light uses AutoModelForCausalLM with processor(conversations=...)
            dtype = torch.bfloat16 if self.device == "cuda" else self.dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            # Some processors bundle tokenizer
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        if self.tokenizer is not None and getattr(self.tokenizer, "pad_token", None) is None:
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos is not None:
                self.tokenizer.pad_token = eos


    def _to_device(self, obj, dtype: torch.dtype = None):
        # Backward-compat: move to self.device, cast dtype for floats only
        target_device = self.device
        def move(o):
            if torch.is_tensor(o):
                if dtype is not None and o.is_floating_point():
                    return o.to(target_device, dtype=dtype)
                return o.to(target_device)
            if isinstance(o, dict):
                return {k: move(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                seq = [move(v) for v in o]
                return type(o)(seq)
            return o
        return move(obj)

    def _to_model_device(self, obj, dtype: torch.dtype = None):
        # Move recursively to the primary device of the model
        target_device = self._get_primary_device()
        def move(o):
            if torch.is_tensor(o):
                if dtype is not None and o.is_floating_point():
                    return o.to(target_device, dtype=dtype)
                return o.to(target_device)
            if isinstance(o, dict):
                return {k: move(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                seq = [move(v) for v in o]
                return type(o)(seq)
            return o
        return move(obj)

    def _get_primary_device(self) -> torch.device:
        # Prefer input embedding weight's device; otherwise first parameter; fallback to self.device
        try:
            if hasattr(self.model, "get_input_embeddings") and self.model.get_input_embeddings() is not None:
                emb = self.model.get_input_embeddings()
                if hasattr(emb, "weight") and emb.weight is not None:
                    return emb.weight.device
        except Exception:
            pass
        try:
            first_param = next(p for p in self.model.parameters() if p is not None)
            return first_param.device
        except Exception:
            return torch.device(self.device)

    def _ensure_same_device(self, inputs_dict, dtype: torch.dtype = None):
        # Convenience: move all tensors found in dict/list/tuple to model primary device
        return self._to_model_device(inputs_dict, dtype=dtype)

    @torch.inference_mode()
    def generate(self, text: str, image_b64: str, max_new_tokens: int = 300) -> str:
        if self.strategy == "ax_vl_light":
            image = b64_to_pil_image(image_b64)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            inputs = self.processor(
                images=[image],
                conversations=[messages],
                padding=True,
                return_tensors="pt",
            )
            inputs = self._to_model_device(inputs)

            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
            }
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
            input_ids = inputs.get("input_ids")
            if input_ids is not None:
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
                response_texts = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                return response_texts[0]
            response_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return response_texts[0]

        if self.strategy == "varco2_llavaov":
            # Follow LLaVA-OneVision conversation style
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_b64, "mime_type": "image/jpeg"},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            # Cast to model device and dtype per VARCO guidance (fp16 on GPU)
            if self.device == "cuda":
                inputs = self._to_model_device(inputs, dtype=torch.float16)
            else:
                inputs = self._to_model_device(inputs)

            generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            # Trim prompt tokens like the model card example
            input_ids = inputs.get("input_ids")
            if input_ids is not None:
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generate_ids)]
                output = self.processor.decode(trimmed[0], skip_special_tokens=True)
            else:
                output = self.processor.decode(generate_ids[0], skip_special_tokens=True)
            return output

        if self.strategy == "kanana":
            # Build a Kanana-style batch with single image and conversation containing <image>
            image = b64_to_pil_image(image_b64)
            batch = [{
                "image": [image],
                "conv": [
                    {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
                    {"role": "user", "content": "<image>"},
                    {"role": "user", "content": text},
                ],
            }]
            inputs = self.processor.batch_encode_collate(
                batch,
                padding_side="left",
                add_generation_prompt=True,
                max_length=8192,
            )
            inputs = self._to_model_device(inputs)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                do_sample=False,
            )
            # Kanana processor exposes tokenizer under processor.tokenizer
            tokenizer = getattr(self.processor, "tokenizer", self.tokenizer)
            return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        if self.strategy == "hyperclovax":
            # HyperCLOVAX uses chat+preprocess path
            vlm_chat = [
                {"role": "system", "content": {"type": "text", "text": "You are a helpful assistant."}},
                {"role": "user", "content": {"type": "text", "text": text}},
                {"role": "user", "content": {"type": "image", "filename": "image.jpg", "image": image_b64}},
            ]

            if hasattr(self.processor, "load_images_videos"):
                new_chat, all_images, is_video_list = self.processor.load_images_videos(vlm_chat)

                # Ensure processor receives lists, even for single image/video
                if not isinstance(all_images, (list, tuple)):
                    all_images = [all_images]
                if not isinstance(is_video_list, (list, tuple)):
                    is_video_list = [is_video_list]

                preprocessed = self.processor(images=all_images, is_video_list=is_video_list)

                # Prefer processor.apply_chat_template (newer API)
                try:
                    model_inputs = self.processor.apply_chat_template(
                        new_chat,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                    )
                    model_inputs = self._to_model_device(model_inputs)
                    preprocessed = self._to_model_device(preprocessed)
                    output_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        **preprocessed,
                    )
                    return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                except Exception:
                    # Fallback to tokenizer.apply_chat_template (older API)
                    if self.tokenizer is None:
                        raise RuntimeError("Tokenizer is required when processor.apply_chat_template is unavailable.")
                    input_ids_cpu = self.tokenizer.apply_chat_template(
                        new_chat,
                        return_tensors="pt",
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    input_ids = self._to_model_device(input_ids_cpu)
                    preprocessed = self._to_model_device(preprocessed)
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        **preprocessed,
                    )
                    return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            # Fallback to default if processor API not found

        # Default LLaVA-like path: ensure images is a list
        image = b64_to_pil_image(image_b64)
        inputs = self.processor(images=[image], text=text, return_tensors="pt")
        inputs = self._to_model_device(inputs)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        if self.tokenizer is not None:
            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]


def parse_model_name_from_id(model_id: str) -> str:
    # Convert HF repo id to a readable short model name for filename
    # e.g., naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B -> HYPERCLOVAX-SEED-VISION-INSTRUCT-3B
    name = model_id.split("/")[-1]
    # Strip revision-like suffix if any (not common in HF ids), and uppercase
    name = re.split(r"-\d{4}-\d{2}-\d{2}", name)[0].upper()
    return name


def save_results_to_file(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as outfile:
        for output, data in results:
            data["response"] = output
            # Exclude image fields but keep topic_difficulty and image_type
            data = {k: v for k, v in data.items() if not (k.startswith("image") and k != "image_type")}
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write("\n")


def main():
    dataset = load_dataset("tabtoyou/KoTextVQA", split="test")

    model_name_for_file = parse_model_name_from_id(MODEL_ID)
    output_path = f"./output/{model_name_for_file}_{SETTING}.jsonl"

    runner = HFVLMInference(MODEL_ID)

    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as infile:
            for line in infile:
                data = json.loads(line)
                results.append((data["response"], data))
        print(f"Loaded existing results from {output_path}")
    else:
        def process_one(doc) -> Tuple[str, Dict]:
            try:
                prompt = construct_prompt(doc)
                image_b64 = doc["image"]
                output = runner.generate(prompt, image_b64, max_new_tokens=300)
                return output, doc
            except Exception as e:
                print(f"Error processing item {doc.get('id', 'N/A')}: {e}")
                return f"Error: {e}", doc

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(process_one, data) for data in dataset]
            for future in tqdm(futures, desc=f"Processing {model_name_for_file} ({SETTING})"):
                results.append(future.result())

        save_results_to_file(results, output_path)

    # Cleanup resources
    try:
        del runner.model
        del runner.processor
        if hasattr(runner, "tokenizer") and runner.tokenizer is not None:
            del runner.tokenizer
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Inference finished and resources cleaned up.")


if __name__ == "__main__":
    main()


