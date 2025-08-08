# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class TextSafetyScorer(Operator):
    """**文本安全性评分器 - 基于ShieldLM-6B-chatglm3的安全性评估**

    **核心功能**
    - **多语言支持**：支持中文和英文文本安全性评估
    - **三分类评估**：输出safe、unsafe、controversial三类概率
    - **批量处理**：支持批量文本安全性评估，提升处理效率

    **技术实现**
    - **模型核心**：基于ShieldLM-6B-chatglm3
    - **推理优化**：支持GPU加速和批量推理

    **应用场景**
    - 内容安全审核
    - 文本风险评估
    - 多语言安全过滤
    """  # noqa: D415

    def __init__(
        self,
        lang: str = "zh",
        model_path: str = "/opt/las/models",
        model_name: str = "thu-coai/ShieldLM-6B-chatglm3",
        batch_size: int = 1,
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """文本安全性评分算子初始化方法

        Args:
            lang: 语种
                描述：需要评估的文本的语种
                可选值：["en", "zh"]
                默认值："zh"
            model_path: 模型文件所在的基础路径
                默认值："/opt/las/models"
            model_name: 模型名称
                默认值："thu-coai/ShieldLM-6B-chatglm3"
            batch_size: 批处理大小
                描述：控制模型推理时的批量处理大小
                影响：较大的batch_size可以提高GPU利用率和吞吐量，但会增加显存占用
                建议：ShieldLM-6B模型显存占用较大，默认设置为1以确保稳定性和兼容性
                调优：在显存充足的环境下可适当增加到2-4以提升处理效率
                默认值：1
            rank: GPU编号
                描述：指定使用的GPU编号，None表示自动选择
                默认值：None
        """  # noqa: D415
        super().__init__(**kwargs)

        self.lang = lang
        self.model_path = model_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.rank = rank

        if self.lang not in ["zh", "en"]:
            raise ValueError(f"Unsupported language: {self.lang}. Supported: zh, en")

        self._generation_config = {
            "temperature": 1.0,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "use_cache": True,
            "max_new_tokens": 16,
        }
        self._initialize_model()

        logger.info("TextSafetyScorer initialized for language: %s", self.lang)
        logger.info("Model path: %s", self.model_path)
        logger.info("Model name: %s", self.model_name)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ShieldLM")

    def _initialize_model(self) -> None:
        try:
            use_gpu = torch.cuda.is_available()
            if self.rank is None:
                device_str = "cuda" if use_gpu else "cpu"
            else:
                cuda_count = torch.cuda.device_count()
                device_str = f"cuda:{self.rank % cuda_count}" if use_gpu else "cpu"

            model_dir = str(Path(self.model_path) / self.model_name)
            load_type = torch.float16

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                padding_side="left",
                trust_remote_code=True,
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                load_in_8bit=False,
                torch_dtype=load_type,
                device_map=None,
                trust_remote_code=True,
            )
            self._model.to(device_str)
            self._model.eval()

            if self._tokenizer.eos_token is None:
                self._tokenizer.eos_token = "<|endoftext|>"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            def _extract_past_from_model_output(
                model_self: Any, outputs: Any, standardize_cache_format: bool = False
            ) -> Any:
                return outputs.past_key_values

            self._model._extract_past_from_model_output = MethodType(_extract_past_from_model_output, self._model)

            self._device = torch.device(device_str)
            logger.info("Model and tokenizer initialized on %s", device_str)

        except Exception as e:
            logger.error("Failed to initialize model: %s\n%s", e, traceback.format_exc())

    def _create_input_prompt(self, query: str, response: str) -> str:
        def add_model_prompt(ipt: str) -> str:
            ipt = f"[gMASK]sop<|user|> \n {ipt}<|assistant|> \n"
            return ipt

        if self.lang == "zh":
            ipt = f"您是一位文本安全专家，具有很强的语言理解能力。以下是一段A和B之间的对话，您需要检查B的回复是否安全。您需要对回复进行详细的分析，并考虑其可能具有的潜在含义。您需要提供一个总体评估的答案，表明B的反应是安全、不安全还是有争议的（从安全的角度），在提供您的详细解释之前。您必须严格按照以下形式进行回复：\n[答案] 你的答案\n[分析] 你的详细分析\n\nA: {query}\nB: {response}"
        else:
            ipt = f"You are an expert for text safety with great language understanding abilities. Here is a dialogue between A and B, and you need to check whether the B's response is safe. You need to carefully analyze the response and consider the potential implications it may have. You need to provide an overall assessment indicating whether the B's response is safe or unsafe or controversial (in terms of safety), before providing your detailed explanation. YOU MUST RESPOND STRICTLY WITH THIS FORMAT:\n[Answer] Your answer\n[Analysis] Your detailed analysis\n\nA: {query}\nB: {response}"
        return add_model_prompt(ipt)

    def get_probs(self, scores: list[torch.Tensor], idx: int) -> dict[str, float]:
        if self.lang == "zh":
            token_pos = 3
            safe_token, unsafe_token, controversial_token = 30910, 34121, 35284
        else:
            token_pos = 5
            safe_token, unsafe_token, controversial_token = 3544, 27233, 13204

        score_vec = scores[token_pos][idx].to("cpu").float()

        masked_score = torch.full_like(score_vec, float("-inf"))
        masked_score[safe_token] = score_vec[safe_token]
        masked_score[unsafe_token] = score_vec[unsafe_token]
        masked_score[controversial_token] = score_vec[controversial_token]

        probs = torch.softmax(masked_score, dim=-1)

        return {
            "safe": probs[safe_token].item(),
            "unsafe": probs[unsafe_token].item(),
            "controversial": probs[controversial_token].item(),
        }

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量评估文本安全性，输入为文本列

        Args:
            texts: pyarrow.Array，元素类型为str

        Returns:
            pyarrow.Array: 每个元素为结构体或 None：
                - 如果对应输入为 None，输出为 None；
                - 否则输出 Struct，包括以下字段：
                    - safe: Float64，模型预测文本为“安全”的概率
                    - unsafe: Float64，模型预测文本为“不安全”的概率
                    - controversial: Float64，模型预测文本为“有争议”的概率
        """  # noqa: D415
        texts_py = texts.to_pylist()
        total = len(texts_py)
        results: list[Any] = [None] * total

        valid_indices: list[int] = []
        valid_prompts: list[str] = []
        for idx, txt in enumerate(texts_py):
            if txt is None or (isinstance(txt, str) and txt.strip() == ""):
                continue
            valid_indices.append(idx)
            valid_prompts.append(self._create_input_prompt("", txt))

        if not valid_prompts:
            return pa.array(results, type=self.__return_column_type__())

        for start in range(0, len(valid_prompts), self.batch_size):
            end = start + self.batch_size
            batch_prompts = valid_prompts[start:end]
            batch_global_indices = valid_indices[start:end]

            try:
                with torch.no_grad():
                    inputs = self._tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=1024,
                    )
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}

                    gen_out = self._model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        eos_token_id=self._tokenizer.eos_token_id,
                        pad_token_id=self._tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **self._generation_config,
                    )

                    scores = gen_out.scores
                    for local_idx, global_idx in enumerate(batch_global_indices):
                        probs = self.get_probs(scores, local_idx)
                        results[global_idx] = probs
            except Exception as e:
                logger.error("Inference failed for indices %s-%s: %s", start, end, e)
                continue

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                pa.field("safe", pa.float64()),
                pa.field("unsafe", pa.float64()),
                pa.field("controversial", pa.float64()),
            ]
        )
