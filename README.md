

<h1 align="center">
  CKIP-Llama-2-7b
</h1>
<p align="center" width="100%">
  <img src="assets/logo.png" alt="Llama" style="width: 20%; display: block; margin: auto;"></a>
</p>
<p align="center">
  <font face="黑體" color=orange size="6"> 繁體中文大型語言模型 by 中央研究院詞庫小組 </font>
</p>
<p align="center">
  <a href="https://huggingface.co/ckiplab/CKIP-Llama-2-7b">
      <img alt="Static Badge" style="display: inline;"
        src="https://img.shields.io/badge/%F0%9F%A4%97Huggingface-CKIP--Llama2-orange">
  </a>
  
  <a href="https://huggingface.co/ckiplab/CKIP-Llama-2-7b-chat">
      <img alt="Static Badge" style="display: inline;"
        src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-CKIP--Llama2--chat-yellow?style=flat">
  </a>
  
  <a href="https://huggingface.co/spaces/ckiplab/CKIP-Llama-2-7b-chat">
      <img alt="Static Badge" style="display: inline;"
        src="https://img.shields.io/badge/demo-CKIP--Llama2--chat-green?logo=dependabot">
  </a>
  
  <a href="https://ckip.iis.sinica.edu.tw/">
      <img alt="Static Badge" style="display: inline;"
        src="https://img.shields.io/badge/Website-CKIP%20Lab-blue?style=flat&logo=internetcomputer">
  </a>
</p>

</br></br>

## 介紹
CKIP-Llama-2-7b 是[中央研究院詞庫小組(CKIP)](https://ckip.iis.sinica.edu.tw/)開發的開源可商用繁體中文大型語言模型（large language model），以商用開源模型Llama-2-7b以及Atom-7b為基礎，再補強繁體中文的處理能力，並對405個可商用的任務檔案同步進行訓練優化，參數量達70億(7 billion)。提供大眾下載，作為學術使用或是商業使用，可應用於文案生成、文學創作、問答系統、客服系統、語言翻譯、文字編修、華語教學等。也可以使用自己的訓練資料集進行訓練，快速構建出一個可以滿足自身任務需求的模型。模型僅需一張 12G 記憶體容量的GPU即可佈屬。

## 展示
您可以在[此連結](https://huggingface.co/spaces/ckiplab/CKIP-Llama-2-7b-chat)測試多任務微調訓練後的 CKIP-Llama-2-7b-chat模型。

## 特色

1. 在多個繁體中文任務上表現優異：我們以商用開源模型Llama-2-7b以及Atom-7b為基礎，再補強繁體中文的處理能力，並對405個可商用的任務檔案同步進行訓練優化，在量化的評估指標上均有優異成績。

2. 可以針對自身任務進行訓練：可以使用自己的訓練資料集接續進行訓練，能夠快速地構建出一個可以滿足自身任務需求的模型。
   
3. 模型可以輕鬆佈署：模型僅需一張 12G 記憶體容量的GPU即可佈屬。

4. 模型可商用：Meta所釋出的Llama-2-7b模型具有開源可商用的特色，以其基礎進行後續加強簡體中文能力的Atom-7b亦以可商用的授權對外開源，我們承襲Llama-2-7b以及Atom-7b，再補強繁體中文的處理能力，訓練出CKIP-Llama-2-7b，亦以可商用的授權對外開源。

## 實測範例

| 輸入                                | 輸出  |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
|台灣最高的山？|玉山|
|在台灣, 什麼是22K？|在臺灣，“22k”指的是一般勞動者的最低薪資|
|由豆腐發酵製作而來，是臺灣、長沙、南京、紹興等地的小吃之一，並流傳世界其它地方，但各地的製作方式、食用方法均有相當大的差異。屬於高鈉食品，但富含發酵食品中普遍存在的維生素B12，另外含有必須胺基酸離胺酸（lysine），應避免與還原醣共同加熱，以免產生糖化終產物等致癌物。\n該食物的名稱是：|臭豆腐|
|為中研院推出的漢堡寫一份推銷文案|中研院漢堡以天然食材為基礎，結合了多種創新技術，精心製作而成。不僅可以滿足你的味蕾，還能讓你感受到中研院對於健康生活的態度。我們的漢堡口感細膩柔滑，營養豐富，口感十分獨特，是一種美味可口的選擇。此外，中研院漢堡還有多種口味可供選擇，如經典原味、香辣、蒜香等，每種口味都有其獨特風味，能夠讓你一次嘗遍多種美味！|
|給你一個標題請寫出內容. 標題:"我的志願"|我的志願是成爲一名醫生。我從小就對醫學感興趣，因爲我喜歡幫助別人。我認爲醫生是一個能夠幫助別人的職業，因爲他們可以幫助人們恢復健康，並且可以幫助他們解決問題。我還認爲醫生是一個能夠學習新知識的職業，因爲他們需要不斷更新自己的知識以應對新的疾病和治療方法。|
|我是一個剛畢業的資工系學生，我想要申請中研院資訊所研究助理的職位，請幫我寫一封求職信。|尊敬的中研院資訊所招聘團隊：</br></br>我是一名剛畢業的資工系學生，擁有豐富的編程和數據分析經驗，擁有良好的學習能力。我對人工智能和機器學習等新技術充滿熱情，希望能夠在中研院資訊所找到一份研究助理的工作，能夠與研究人員一起探索人工智能領域的前沿技術。</br></br>我有良好的計算機編程基礎，熟悉Python和C++等語言，熟練掌握數據分析工具，如Pandas、NumPy和Scikit-learn等。此外，我還具備出色的自學能力，能夠快速理解和掌握新的技術。</br>我相信自己有能力勝任中研院資訊所的研究助理職位，並期待能夠通過面試的機會來展示自己的技能和能力。</br></br>非常感謝您的時間，期待您的回覆！|
|規劃一下日月潭二日遊|日月潭二日遊行程可以安排在第一天上午前往日月潭纜車站乘坐纜車，欣賞湖面美景；下午前往水社碼頭，乘船遊湖，參觀水社壩、玄光寺和日月潭青年活動中心；晚上可以前往潭心亭觀賞夜景。第二天上午，可以參觀伊達邵文化公園，瞭解伊達邵文化的歷史；下午則可以前往文武廟，參觀遺址、廟宇建築和歷史文物。| 
|請幫我猜測一下一架波音747有多重？請給出分析的過程|首先，我們需要了解一種飛機的重量由什麼組成。首先考慮的是燃油重量和乘客、駕駛員和其他乘務人員的重量（包括行李）等客運部分的因素；其次還有航空公司本身要承擔的引擎、空調系統等設施的相關重量以及裝配部件等其他各種必要材料的成本因素。然後根據不同型號的七四七和其性能來計算它的整體重量；最後對比其他相似尺寸或同樣大小但使用不同材料進行加工和使用的不同商業模式的新款飛機以得出可能的答案—一架新的波音747通常會達到80-125萬磅之間（即36.9至55公噸）左右|
|請為以下內容產生條列式摘要。亞運會開設的比賽項目由亞奧理事會認可。主辦國可在認可的項目中選擇比賽項目，每屆亞運會比賽項目不得少於11項，除田徑和游泳以外的項目由亞運會主辦國選擇，所設各項目至少須有4個以上參賽隊。而且考慮亞洲不同區域的傳統文化特點，並且為亞洲各個國家或地區的青年提供展示技能的機會及增加亞運會的吸引力，亞運會設立一部分非奧運項目。|1. 亞運會比賽項目由亞奧理事會認可。</br>2. 主辦國可在認可的項目中選擇比賽項目。</br>3. 每屆亞運會比賽項目不得少於11項。</br>4. 除田徑和游泳以外的項目由亞運會主辦國選擇。</br>5. 所有項目至少須有4個以上參賽隊。</br>6. 設立一部分非奧運項目，以增加亞運會的吸引力。|
|將以下內容換句話說：他們經常一起坐在河岸上，討論哲學和科學問題以及未來的新發明。|他們常常在河畔坐下來圍繞着哲理、科技和其他新事物展開對話與思考|
|火星大氣以二氧化碳為主，既稀薄又寒冷，其表面特徵讓人聯想起月球上的撞擊坑，以及地球上的山谷、沙漠和極地冰蓋。英語中，通常被稱為紅色星球（英語：Red Planet）。火星在視覺上呈現為橘紅色是因為地表廣泛普遍的分佈著氧化鐵，它使火星呈現出一種紅色的外觀，在肉眼可見的天體中獨具一格。根據以上描述，火星為何被稱做紅色星球？|火星表面的氧化鐵呈現橘紅色，因此被稱作「紅色星球」。|
|翻譯以下為英文: 他們從來沒有停止過彼此之間浪漫的火花。|They never stop the sparks of romance between them.|
|翻譯以下為中文: They never stop the sparks of romance between them.|他們從未熄滅彼此的浪漫火花
|為滑鼠做一首詩|我的指尖像一隻飛行的鳥，把我帶到每一個新的地方；我將它放在桌面上溫暖的手掌中，從而開啟一段旅程。我的手指就像一個魔法師一樣能牽引我去任何想去的地方！|
|為滑鼠做一首七言絕句|左右鍵動如流星，彷似指揮操舵手；時光蹉踎人難追，遊走漫無底限路|
|我出上聯, 你接下聯, 上聯是:"地處中研院"|下聯：人稱大學士|
|老闆在下班後仍找我怎麼辦？|你可以委婉地告訴他，如果你無法做到他所要求的時間或者你必須要處理其他事情。同樣你也應當明確自己需要完成什麼以及何時能夠達到目標，讓他知道你能承擔這個要求並且給予適當的回報以表示感謝。|
|我怎麼知道自己不是外星人|您好，如果您認爲自己是外星生命體並且擔心與人相處時感到尷尬、害怕或困惑的話，您可能需要考慮進行心理諮詢。這樣做可以讓您可以更好地理解自己的情緒和想法以及如何應對它們。|
|說一個關於原子筆的愛情故事|有一天，一個男孩和一個女孩在學校裏相遇了。他們都喜歡寫字，於是他們開始交換原子筆。他們用原子筆寫下他們的愛情故事，並把它們放在一起。他們每天都會用原子筆寫下他們的愛情故事，並把它們放在一起。他們的愛情故事越來越長，越來越深，越來越甜蜜。最後，他們決定把原子筆放在一起，永遠不會分開。|

## 模型

我們有兩個模型，一個是繁體中文預訓練後的 CKIP-Llama-2-7b，一個是以前者為基礎，繼續多任務微調訓練後的 CKIP-Llama-2-7b-chat

|             **Model**                                                               | **Type**                |
|:-----------------------------------------------------------------------------------:|:-----------------------:|
|🤗[ckiplab/CKIP-Llama-2-7b](https://huggingface.co/ckiplab/CKIP-Llama-2-7b)          | Continuous Pretraining |
|🤗[ckiplab/CKIP-Llama-2-7b-chat](https://huggingface.co/ckiplab/CKIP-Llama-2-7b-chat) • ✍️[Online Demo](https://huggingface.co/spaces/ckiplab/CKIP-Llama-2-7b-chat)| w/ Instruction Tuning  |


## 資料

CKIP-Llama-2-7b的資料分成兩個：預訓練資料 和 多任務微調資料

### 預訓練資料

| 資料集                                                       | 檔案大小 | 資料筆數  |    token數    |
| :----------------------------------------------------------- | :------: | :-------: | :-----------: |
| [CommonCrawl](https://commoncrawl.org/) (subset)             |   2.0G   |  700,000  | 1,024,820,457 |
| [英文維基百科](https://dumps.wikimedia.org/enwiki/) (subset) |   721M   |  250,000  |  176,021,369  |
| [中文維基百科](https://dumps.wikimedia.org/zhwiki/)          |   1.4G   |  707,175  |  726,000,253  |
| [臺灣碩博士論文摘要](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi?o=d) |   1.7G   | 1,055,844 |  865,681,958  |
| [中央研究院漢語平衡語料庫](https://asbc.iis.sinica.edu.tw/) * 2 |   42M    |  16,673   |  21,635,195   |
| [徐志摩詩歌全集](https://ixdzs.tw/read/55056/) * 2           |   258K   |    74     |    133,029    |
| [朱自清散文經典全集](https://ixdzs.tw/read/55431/) * 2       |   1.1M   |    217    |    526,190    |
| total                                                        |   5.8G   | 2,736,947 | 2,837,112,865 |
- 我們使用 OpenAI tiktoken tokenizer 來計算資料集的 token 數量，避免各個模型詞表不同造成的誤差。
訓練使用 CKIP-Llama-2-7b tokenizer 計算的結果可以參照[此表](https://github.com/ckiplab/CKIP-Llama-2-7b/blob/master/token_amounts.md)。


### 多任務微調資料

為了達到可商用目的，我們的多任務微調資料避免使用alpaca及vicuna等經由OpenAI產品生成的資料集，而取自於純人工產生的COIG-PC資料集和dolly資料集，分別經過如下操作：
* [COIG-PC 資料集](https://huggingface.co/datasets/BAAI/COIG-PC)：COIG-PC資料集當中有為每一個任務檔案註明是否可以商用，我們從中挑選出405個可商用的任務檔案，再從其中隨機抽取出我們的任務微調資料集。
* [dolly-15k 資料集](https://huggingface.co/datasets/Elliot4AI/dolly-15k-chinese-guanacoformat)：dolly-15k資料集整體均為為可商用資料集。我們將已經轉成簡體中文的 Dolly dataset 處理成 [BELLE](https://github.com/LianjiaTech/BELLE/) 訓練格式並經過 [OpenCC](https://github.com/BYVoid/OpenCC) 轉成繁體中文




| 資料集                     | Training | Development | Testing |
|:---------------------------:|:--------:|:-----------:|:-------:|
|BAAI/COIG-PC (subset) |  1,441,977 |     4,050    |   3,740  |
|[traditional-chinese-dolly-15k](https://huggingface.co/datasets/ckiplab/traditional-chinese-dolly-15k)  |   15,001  |      10     |    -    |
|  Total  |  1,456,978 |     4,060    |   3,740  |


### C-Eval
[C-Eval ](https://cevalbenchmark.com/index.html)是一個涵蓋四大領域共52個學科的中文模型評測數據集。我們使用 `OpenCC` 將[原始數據集](https://huggingface.co/datasets/ceval/ceval-exam)轉成繁體中文後進行測試。測試使用該數據集的dev作為 `5-shot` 來源，下表是在val集上進行測試的結果。

| Model 5-shot  |    Average |  Humanities|     Other  |     STEM   |Social Science|
|:-------------:|:----------:|:----------:|:----------:|:----------:|:------------:|
|[Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 22.26±0.00 | 23.45±0.00 | 20.81±0.00 | 21.25±0.00 |  24.57±0.00  |
|[bloom-3b-zh](https://huggingface.co/ckip-joint/bloom-3b-zh) | 29.99±0.58 | 31.47±0.98 | 30.53±1.31 | 28.51±1.26 |  30.72±1.29  |
|[bloom-3b-zh-chat](https://huggingface.co/ckip-joint/bloom-3b-zh-instruct) | 27.97±0.33 | 28.20±0.80 | 26.98±0.99 | 25.81±0.76 |  33.09±1.22  |
|[Atom-7B](https://huggingface.co/FlagAlpha/Atom-7B) | 35.95±0.67 | 41.72±1.78 | 36.05±0.86 | 29.73±1.67 |  41.92±1.66  |
|[Atom-7B-Chat](https://huggingface.co/FlagAlpha/Atom-7B-Chat) | 34.14±0.55 | 35.77±1.59 | 32.59±0.76 | 31.86±1.28 |  38.59±0.50  |
|[CKIP-Llama-2-7b](https://huggingface.co/ckiplab/CKIP-Llama-2-7b) | 37.03±0.57 | **45.42±1.16** | 37.95±0.63 | 29.36±0.83 |  **42.11±0.68**  |
|[CKIP-Llama-2-7b-chat](https://huggingface.co/ckiplab/CKIP-Llama-2-7b-chat) | **38.39±0.54** | 44.45±1.38 | **40.69±1.30** | **32.12±0.59** |  41.72±1.17  |

#### 評測方法
1. 下載[C-Eval數據集](https://huggingface.co/datasets/ceval/ceval-exam)並轉換成繁體中文，放在 `evaluation` 資料夾下
    ```bash
    .
    ├── evaluation/
    │   ├── ceval-exam/
    │   │   ├── dev/
    │   │   ├── val/
    │   │   └── test/
    │   └── evaluate_tw.py
    ```
2. 填入模型名和路徑後執行下列命令測試於單顆GPU上。 `evaluate_tw.py` 檔案修改於 [evaluate_zh.py](https://github.com/baichuan-inc/Baichuan-7B/blob/main/evaluation/evaluate_zh.py)，我們將程式中的prompt字串也轉換成繁體中文
    ```bash
    MODEL="YOUR_MODEL_NAME"
    MODEL_PATH="YOUR_MODEL_PATH"
    SPLIT="val"

    for i in {1..5}; do
      OUT="${MODEL}/run${i}"
      mkdir -p "$OUT"
      echo "It will write the evaluation results to $OUT"
      python evaluate_tw.py --model_name_or_path $MODEL_PATH --split $SPLIT --output_dir $OUT
    done
    ```



## 如何佈署在我自己的機器

|      使用裝置      | GPU數量 | Quantization | 生成100個tokens所需時間 |
| :----------------: | :-----: | :----------: | :---------------------: |
| NVIDIA GTX 1080 Ti |    1    |     int8     |          36秒           |
| NVIDIA RTX 2080 Ti |    1    |     int8     |          23秒           |
| NVIDIA GTX 1080 Ti |    4    |     fp16     |          20秒           |
| NVIDIA RTX 2080 Ti |    4    |     fp16     |           7秒           |


```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name_or_path = 'ckiplab/CKIP-Llama-2-7b-chat'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# int8
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', torch_dtype=torch.float16, load_in_8bit=True)
# fp16
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', torch_dtype=torch.float16)
# fp32
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')

sampling_strategy = {'max_new_tokens': 50,
                     'top_p': 0.95}

prompt_template = "Human: \n{}\n\nAssistant: \n"
prompt = prompt_template.format('台灣最高的山？')
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

output = model.generate(input_ids=inputs.input_ids, **sampling_strategy)

print( tokenizer.batch_decode(output) )
```

### Training
我們的模型訓練在4顆A100的機器上，環境使用 `python 3.8` 和 `CUDA 11.8` 。

### Inference on a Single GPU
| Quantization | CUDA Memory Requirements |
|:------------:|:-----------:|
|[int8](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/quantization#load-a-large-model-in-8bit)|     12G     |
|     fp16     |     20G     |

## 如何用我自己的資料集來訓練

您可以輕鬆用您自己的資料集來訓練，一般來說，不需要再進行預訓練步驟，只要將您的任務進行任務微調訓練即可。您可以使用 CKIP-Llama-2-7b 或 CKIP-Llama-2-7b-chat 作為訓練的起始模型。

### Installation

系統需求：
* Python 3.8 and above.
* CUDA 11.6 and above.
* PyTorch 2.0.0 and above.
  * 請先至[PyTorch官方網站](https://pytorch.org/)根據您的系統環境安裝對應的PyTorch版本。
* Linux only.

```bash
pip install -U pip
git clone --recursive https://github.com/ckiplab/CKIP-Llama-2-7b.git
cd CKIP-Llama-2-7b
pip install -r requirements.txt
```

### Preparing data

準備 jsonl 格式資料集

```json
{"id": "dolly0", "conversations": [{"from": "human", "value": "維珍澳大利亞航空（Virgin Australia Airlines Pty Ltd 的商業名稱）是一家總部位於澳大利亞的航空公司。它是機隊規模最大的使用維珍品牌的航空公司。它於 2000 年 8 月 31 日以 Virgin Blue 名義開始運營，在一條航線上有兩架飛機。 2001 年 9 月安捷澳大利亞公司倒閉後，它突然發現自己成爲澳大利亞國內市場的一家主要航空公司。此後，該航空公司已發展到以布里斯班、墨爾本和悉尼爲樞紐，直接爲澳大利亞 32 個城市提供服務。 維珍澳大利亞航空什麼時候開始運營？"}, {"from": "assistant", "value": "維珍澳大利亞航空於 2000 年 8 月 31 日以 Virgin Blue 名義開始運營，在一條航線上有兩架飛機。"}]}
{"id": "dolly1", "conversations": [{"from": "human", "value": "魚是哪一個品種？頂部或繩索"}, {"from": "assistant", "value": "託普"}]}
{"id": "dolly2", "conversations": [{"from": "human", "value": "駱駝爲什麼能在沒有水的情況下生存很長時間？"}, {"from": "assistant", "value": "駱駝利用駝峯中的脂肪來長時間保持能量充足和水分充足。"}]}
```

### Modify parameters

登入 [wandb](https://docs.wandb.ai/ref/cli/wandb-login)，可以透過 wandb 即時監控訓練狀態

run_sft_CKIP-LLaMA.sh 填入參數

```bash
output_dir=... # output file
train_file=... # training file
validation_file=... # training file
export WANDB_PROJECT=... # wandb project name
export BELLE_DIR=... # absolute path of BELLE
```

### Start training

```bash
bash run_sft_CKIP-LLaMA.sh
```

## 參與者

**[Wei-Yun Ma](https://www.iis.sinica.edu.tw/pages/ma/) (馬偉雲)** at CKIP
* Email: ma@iis.sinica.edu.tw

**[Yu-Hung Wu](https://marzear.github.io/) (吳昱宏)** at CKIP
* Email: yhwu@iis.sinica.edu.tw

**[Nai-Chi Yang](https://github.com/nike00811/) (楊奈其)** at CKIP
* Email: nike00811@iis.sinica.edu.tw

**[Chin-Tung Lin](https://github.com/linchintung) (林鑫彤)** at CKIP
* Email: cindylin@iis.sinica.edu.tw

## 引用

```bibtex

@misc{CKIP-Llama-2-7b,
    author={Wei-Yun Ma, Yu-Hung Wu, Nai-Chi Yang, Chin-Tung Lin}
    title={Optimize Llama-2 to Better Support Traditional Chinese},
    year={2023},
    url={https://github.com/ckiplab/CKIP-Llama-2-7b},   
}
```



