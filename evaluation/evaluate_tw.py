import argparse
import json
import os

from tqdm import tqdm
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="model name or path"
    )
    parser.add_argument(
        "--shot", type=int, default=5, help="number of shot for few-shot learning"
    )
    parser.add_argument(
        "--split", type=str, default="val", help="split of dataset to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="ceval_output", help="output directory"
    )
    parser.add_argument(
        "--max_input_length", type=int, default=-1, help="max input length of model"
    )
    return parser.parse_args()


class CEval:
    DATA_PATH = "./ceval-exam"
    TASK2DESC = {
        "high_school_physics": "高中物理",
        "fire_engineer": "註冊消防工程師",
        "computer_network": "計算機網絡",
        "advanced_mathematics": "高等數學",
        "logic": "邏輯學",
        "middle_school_physics": "初中物理",
        "clinical_medicine": "臨床醫學",
        "probability_and_statistics": "概率統計",
        "ideological_and_moral_cultivation": "思想道德修養與法律基礎",
        "operating_system": "操作系統",
        "middle_school_mathematics": "初中數學",
        "chinese_language_and_literature": "中國語言文學",
        "electrical_engineer": "註冊電氣工程師",
        "business_administration": "工商管理",
        "high_school_geography": "高中地理",
        "modern_chinese_history": "近代史綱要",
        "legal_professional": "法律職業資格",
        "middle_school_geography": "初中地理",
        "middle_school_chemistry": "初中化學",
        "high_school_biology": "高中生物",
        "high_school_chemistry": "高中化學",
        "physician": "醫師資格",
        "high_school_chinese": "高中語文",
        "tax_accountant": "稅務師",
        "high_school_history": "高中歷史",
        "mao_zedong_thought": "毛澤東思想和中國特色社會主義理論概論",
        "high_school_mathematics": "高中數學",
        "professional_tour_guide": "導游資格",
        "veterinary_medicine": "獸醫學",
        "environmental_impact_assessment_engineer": "環境影響評價工程師",
        "basic_medicine": "基礎醫學",
        "education_science": "教育學",
        "urban_and_rural_planner": "註冊城鄉規劃師",
        "middle_school_biology": "初中生物",
        "plant_protection": "植物保護",
        "middle_school_history": "初中歷史",
        "high_school_politics": "高中政治",
        "metrology_engineer": "註冊計量師",
        "art_studies": "藝術學",
        "college_economics": "大學經濟學",
        "college_chemistry": "大學化學",
        "law": "法學",
        "sports_science": "體育學",
        "civil_servant": "公務員",
        "college_programming": "大學編程",
        "middle_school_politics": "初中政治",
        "teacher_qualification": "教師資格",
        "computer_architecture": "計算機組成",
        "college_physics": "大學物理",
        "discrete_mathematics": "離散數學",
        "marxism": "馬克思主義基本原理",
        "accountant": "註冊會計師",
    }

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        output_dir: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir

    def run(self, shot: int, split: str):
        results, accs = {}, {}

        # run all task
        for task_name in self.TASK2DESC:
            print("=" * 100)
            print(f"run task: {task_name}")
            result, acc = self.run_single_task(task_name, shot, split)
            results[task_name] = result
            accs[task_name] = acc
            result_path = os.path.join(self.output_dir, f"{task_name}.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"save result to {result_path}")

        # results
        acc_path = os.path.join(self.output_dir, "acc.json")
        with open(acc_path, "w") as f:
            json.dump(accs, f, indent=2, ensure_ascii=False)
        average_acc = sum(accs.values()) / len(accs)
        print(f"average acc: {average_acc}")

    def run_single_task(self, task_name: str, shot: int, split: str):
        dataset = load_dataset(self.DATA_PATH, task_name)
        results = []
        acc = 0
        for data in tqdm(dataset[split]):
            prompt = f"以下是中國關於{self.TASK2DESC[task_name]}考試的單項選擇題，請選出其中的正確答案。\n"
            if shot != 0:
                shuffled = dataset["dev"].shuffle()
                for i in range(min(shot, len(shuffled))):
                    prompt += "\n" + self.build_example(shuffled[i], with_answer=True)
            prompt += "\n" + self.build_example(data, with_answer=False)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True).cuda()
            output = self.model.generate(
                input_ids,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                temperature=0.1,
                top_p=0.5,
                repetition_penalty=1.1,
            )
            scores = output.scores[0][0].to(torch.float32)
            label_score = []
            candidates = ["A", "B", "C", "D"]
            for can in candidates:
                can_id = self.tokenizer.encode(can)[-1]
                label_score.append(scores[can_id].item())
            answer = candidates[np.argmax(label_score)]
            results.append(
                {
                    "prompt": prompt,
                    "correct": answer == data["answer"].strip().upper(),
                    "answer": answer,
                }
            )
            acc += answer == data["answer"].strip().upper()
        acc /= len(dataset[split])
        return results, acc

    def build_example(self, data, with_answer: bool = True):
        question = data["question"]
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],
            ]
        )
        answer = data["answer"].strip().upper() if with_answer else ""
        return f"{question}\n{choice}\n答案：{answer}"


def main():
    args = parse_argument()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=False,
        add_eos_token=False,
        padding_side="left",
    )
    ceval = CEval(model, tokenizer, args.output_dir)
    ceval.run(args.shot, args.split)


if __name__ == "__main__":
    main()