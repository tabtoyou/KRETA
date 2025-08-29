# KRETA Benchmark
 
[🤗 KRETA](https://huggingface.co/datasets/tabtoyou/KRETA) | [📖 Paper](https://arxiv.org/abs/2508.19944) | [🏆 Leaderboard](https://github.com/tabtoyou/KRETA/tree/main?tab=readme-ov-file#leaderboard)

**KRETA: A Benchmark for Korean Reading and Reasoning in Text-Rich VQA Attuned to Diverse Visual Contexts** (EMNLP 2025 Main Conference) <br>
[Taebaek Hwang*](https://www.linkedin.com/in/taebaek-hwang-a66685170/), [Minseo Kim*](https://www.linkedin.com/in/minseo-kim-939293323/), [Gisang Lee](https://www.linkedin.com/in/gisanglee/), 
[Seonuk Kim](https://www.linkedin.com/in/seonuk-kim/), [Hyunjun Eun](https://www.linkedin.com/in/hyunjun-eun-239381196/)

## Abstract
Understanding and reasoning over text within visual contexts poses a significant challenge for Vision-Language Models (VLMs), given the complexity and diversity of real-world scenarios. To address this challenge, text-rich Visual Question Answering (VQA) datasets and evaluation benchmarks have emerged for high-resource languages like English. However, a critical gap persists for low-resource languages such as Korean, where the lack of comprehensive benchmarks hinders robust model evaluation and comparison. To bridge this gap, we introduce **KRETA**, a benchmark for **K**orean **R**eading and r**E**asoning in **T**ext-rich VQA **A**ttuned to diverse visual contexts. KRETA facilitates an in-depth evaluation of both visual text understanding and reasoning capabilities, while also supporting a multifaceted assessment across 15 domains and 26 image types. Additionally, we introduce a semi-automated VQA generation pipeline specifically optimized for text-rich settings, leveraging refined stepwise image decomposition and a rigorous seven-metric evaluation protocol to ensure data quality. While KRETA is tailored for Korean, we hope our adaptable and extensible pipeline will facilitate the development of similar benchmarks in other languages, thereby accelerating multilingual VLM research.
   
<p align="center">
  <img src="https://github.com/user-attachments/assets/fea1b76e-f1a5-4655-b11d-9c1b368d98f6" width="850" />
  <br>
  <sub><em>(a) Distribution of samples across 15 domains (inner ring) and 26 image types (outer ring). Dark green and light green segments in the inner ring represent the number of samples associated with System 2 and System 1, respectively. (b) The semi-automated VQA generation pipeline.</sub></em>
</p>


## Examples
![KRETA_Examples](https://github.com/user-attachments/assets/a650868f-c248-4451-a36d-fa8f7d28b47a)


## LeaderBoard

| Rank | Model               | Release   | Type        | Overall | System1 | System2 |
|------|---------------------|-----------|-------------|---------|---------|---------|
| 1    | Gemini-2.0-flash    | 25.02.05  | Closed      | **85.4** | **98.0** | 69.8 |
| 2    | GPT-4o              | 24.11.20  | Closed      | 84.6    | 95.9    | **70.5** |
| 3    | Claude-3.5-Sonnet   | 24.10.22  | Closed      | 80.5    | 93.4    | 64.5 |
| 4    | [A.X-4.0-VL-LIGHT (7B)](https://huggingface.co/skt/A.X-4.0-VL-Light)   | 25.07.31  | Open-Source | 78.0    | 95.3    | 56.5 |
| 5    | [VARCO-VISION-2.0 (14B)](https://huggingface.co/NCSOFT/VARCO-VISION-2.0-14B)    | 25.07.16  | Open-Source | 75.4    | 93.5    | 53.1 |
| 6    | [KANANA-1.5-V (3B)](https://huggingface.co/kakaocorp/kanana-1.5-v-3b-instruct)   | 25.07.24  | Open-Source | 75.0    | 94.0    | 51.4 |
| 7    | GPT-4o-mini         | 24.07.18  | Closed      | 73.3    | 88.7    | 54.1 |


<details>
<summary>Full Leaderboard (click to expand)</summary>
<table style="width:90%;">
<tr>
<th>Models</th>
<td><b>Open-Source</b></td>
<td><b>Overall</b></td>
<td><b>System1</b></td>
<td><b>System2</b></td>
<td><b>Gov.</b></td>
<td><b>Econ.</b></td>
<td><b>Mktg.</b></td>
<td><b>Comm.</b></td>
<td><b>Edu.</b></td>
<td><b>Med.</b></td>
<td><b>Tech.</b></td>
<td><b>Arts.</b></td>
<td><b>Transp.</b></td>
<td><b>Tour.</b></td>
<td><b>FnB.</b></td>
<td><b>Ent.</b></td>
<td><b>Life.</b></td>
<td><b>Sci.</b></td>
<td><b>Hist.</b></td>
</tr>
<tr>
<th align="left">Gemini-2.0-flash (25.02.05)</th>
<td align="middle">✘</td>
<td><b>85.4</b></td>
<td><b>98.0</b></td>
<td>69.8</td>
<td><b>95.1</b></td>
<td><b>95.2</b></td>
<td><b>99.3</b></td>
<td><b>96.1</b></td>
<td><b>96.7</b></td>
<td><b>92.2</b></td>
<td>93.5</td>
<td>98.8</td>
<td><b>90.4</b></td>
<td><b>98.1</b></td>
<td>93.2</td>
<td>95.2</td>
<td><b>96.6</b></td>
<td><b>44.1</b></td>
<td>78.3</td>
</tr>
<tr>
<th align="left">GPT-4o (24.11.20)</th>
<td align="middle">✘</td>
<td>84.6</td>
<td>95.9</td>
<td><b>70.5</b></td>
<td>93.5</td>
<td>92.3</td>
<td>97.2</td>
<td>90.3</td>
<td><b>96.7</b></td>
<td>91.1</td>
<td><b>96.7</b></td>
<td><b>100.0</b></td>
<td>84.4</td>
<td>93.5</td>
<td><b>93.6</b></td>
<td><b>97.0</b></td>
<td>95.1</td>
<td><b>44.1</b></td>
<td><b>93.3</b></td>
</tr>
<tr style="border-bottom: 1.5px solid">
<th align="left">Claude-3.5-Sonnet (24.10.22)</th>
<td align="middle">✘</td>
<td>80.5</td>
<td>93.4</td>
<td>64.5</td>
<td>93.5</td>
<td>91.3</td>
<td>92.4</td>
<td>87.0</td>
<td>93.0</td>
<td>91.1</td>
<td>87.0</td>
<td>91.6</td>
<td>84.4</td>
<td>94.4</td>
<td>89.8</td>
<td>92.3</td>
<td>92.2</td>
<td>37.4</td>
<td>70.0</td>
</tr>
<tr>
<th align="left">A.X-4.0-VL-LIGHT (25.07.31)</th>
<td align="middle">✅</td>
<td>78.0</td>
<td>95.3</td>
<td>56.5</td>
<td>90.2</td>
<td>87.5</td>
<td>91.7</td>
<td>89.6</td>
<td>94.0</td>
<td>88.9</td>
<td>87.0</td>
<td>92.8</td>
<td>82.0</td>
<td>94.4</td>
<td>86.0</td>
<td>86.3</td>
<td>86.3</td>
<td>33.9</td>
<td>63.3</td>
</tr>
<tr>
<th align="left">VARCO-VISION-2.0 (14B) (25.07.16)</th>
<td align="middle">✅</td>
<td>75.4</td>
<td>93.5</td>
<td>53.1</td>
<td>90.6</td>
<td>94.2</td>
<td>88.3</td>
<td>88.3</td>
<td>90.7</td>
<td>90.0</td>
<td>88.0</td>
<td>89.2</td>
<td>79.0</td>
<td>87.0</td>
<td>83.3</td>
<td>87.5</td>
<td>92.2</td>
<td>26.8</td>
<td>33.3</td>
</tr>
<tr>
<th align="left">KANANA-1.5-V (3B) (25. 07. 24)</th>
<td align="middle">✅</td>
<td>75.0</td>
<td>94.0</td>
<td>51.4</td>
<td>86.5</td>
<td>81.7</td>
<td>94.5</td>
<td>84.4</td>
<td>87.9</td>
<td>80.0</td>
<td>80.4</td>
<td>92.8</td>
<td>77.3</td>
<td>93.5</td>
<td>89.4</td>
<td>85.1</td>
<td>86.8</td>
<td>29.7</td>
<td>48.3</td>
</tr>
<tr>
<th align="left">GPT-4o-mini (24.07.18)</th>
<td align="middle">✘</td>
<td>73.3</td>
<td>88.7</td>
<td>54.1</td>
<td>82.4</td>
<td>82.7</td>
<td>85.5</td>
<td>84.4</td>
<td>87.4</td>
<td>83.3</td>
<td>80.4</td>
<td>89.2</td>
<td>80.2</td>
<td>84.3</td>
<td>81.4</td>
<td>86.3</td>
<td>87.3</td>
<td>30.3</td>
<td>45.0</td>
</tr>
<tr>
<th align="left">VARCO-VISION (14B)</th>
<td align="middle">✅</td>
<td>72.3</td>
<td>90.9</td>
<td>49.3</td>
<td>81.6</td>
<td>87.5</td>
<td>83.4</td>
<td>83.1</td>
<td>84.2</td>
<td>86.7</td>
<td>84.8</td>
<td>79.5</td>
<td>82.6</td>
<td>83.3</td>
<td>76.1</td>
<td>81.5</td>
<td>85.3</td>
<td>33.7</td>
<td>31.7</td>
</tr>
<tr>
<th align="left">Qwen2.5-VL (3B)</th>
<td align="middle">✅</td>
<td>71.8</td>
<td>94.2</td>
<td>43.9</td>
<td>81.6</td>
<td>76.9</td>
<td>85.5</td>
<td>77.9</td>
<td>87.4</td>
<td>80.0</td>
<td>79.3</td>
<td>85.5</td>
<td>75.4</td>
<td>84.3</td>
<td>76.9</td>
<td>87.5</td>
<td>83.3</td>
<td>33.9</td>
<td>36.7</td>
</tr>
<tr>
<th align="left">InternVL2.5 (8B)</th>
<td align="middle">✅</td>
<td>70.8</td>
<td>89.8</td>
<td>47.3</td>
<td>81.6</td>
<td>76.9</td>
<td>85.5</td>
<td>81.8</td>
<td>83.7</td>
<td>81.1</td>
<td>77.2</td>
<td>78.3</td>
<td>76.0</td>
<td>83.3</td>
<td>74.2</td>
<td>78.6</td>
<td>85.8</td>
<td>34.1</td>
<td>38.3</td>
</tr>
<tr>
<th align="left">InternVL2.5 (4B)</th>
<td align="middle">✅</td>
<td>70.7</td>
<td>90.7</td>
<td>45.9</td>
<td>82.0</td>
<td>76.9</td>
<td>87.6</td>
<td>83.1</td>
<td>83.7</td>
<td>78.9</td>
<td>79.3</td>
<td>79.5</td>
<td>75.4</td>
<td>77.8</td>
<td>69.3</td>
<td>81.0</td>
<td>86.3</td>
<td>33.9</td>
<td>46.7</td>
</tr>
<tr>
<th align="left">Qwen2.5-VL (7B)</th>
<td align="middle">✅</td>
<td>68.5</td>
<td>94.5</td>
<td>36.1</td>
<td>80.0</td>
<td>77.9</td>
<td>85.5</td>
<td>81.2</td>
<td>87.4</td>
<td>76.7</td>
<td>75.0</td>
<td>89.2</td>
<td>77.8</td>
<td>82.4</td>
<td>77.7</td>
<td>86.3</td>
<td>85.8</td>
<td>15.1</td>
<td>36.7</td>
</tr>
<tr>
<th align="left">MiniCPM-o-2.6 (8B)</th>
<td align="middle">✅</td>
<td>64.3</td>
<td>84.1</td>
<td>39.9</td>
<td>75.9</td>
<td>83.7</td>
<td>79.3</td>
<td>75.9</td>
<td>76.7</td>
<td>65.6</td>
<td>75.0</td>
<td>73.5</td>
<td>69.5</td>
<td>79.6</td>
<td>67.8</td>
<td>77.4</td>
<td>74.0</td>
<td>25.5</td>
<td>25.0</td>
</tr>
<tr>
<th align="left">Ovis1.6-Gemma2 (9B)</th>
<td align="middle">✅</td>
<td>58.4</td>
<td>68.9</td>
<td>45.4</td>
<td>64.1</td>
<td>69.2</td>
<td>71.0</td>
<td>72.7</td>
<td>60.9</td>
<td>71.1</td>
<td>67.4</td>
<td>53.0</td>
<td>68.9</td>
<td>75.9</td>
<td>65.2</td>
<td>58.9</td>
<td>63.2</td>
<td>30.5</td>
<td>28.3</td>
</tr>
<tr>
<th align="left">LLaVA-OneVision (7B)</th>
<td align="middle">✅</td>
<td>54.0</td>
<td>65.1</td>
<td>40.1</td>
<td>64.1</td>
<td>63.5</td>
<td>63.4</td>
<td>63.6</td>
<td>58.6</td>
<td>55.6</td>
<td>64.1</td>
<td>45.8</td>
<td>68.3</td>
<td>65.7</td>
<td>55.3</td>
<td>55.4</td>
<td>55.9</td>
<td>30.8</td>
<td>33.3</td>
</tr>
<tr>
<th align="left">Deepseek-VL2-small (2.8B)</th>
<td align="middle">✅</td>
<td>53.3</td>
<td>67.3</td>
<td>36.1</td>
<td>61.6</td>
<td>63.5</td>
<td>66.9</td>
<td>63.0</td>
<td>57.2</td>
<td>64.4</td>
<td>68.5</td>
<td>50.6</td>
<td>59.9</td>
<td>63.0</td>
<td>48.9</td>
<td>56.0</td>
<td>57.4</td>
<td>30.8</td>
<td>36.7</td>
</tr>
<tr>
<th align="left">Ovis1.6-Llama3.2 (3B)</th>
<td align="middle">✅</td>
<td>52.2</td>
<td>62.8</td>
<td>39.1</td>
<td>64.5</td>
<td>69.2</td>
<td>60.7</td>
<td>57.1</td>
<td>55.8</td>
<td>54.4</td>
<td>62.0</td>
<td>51.8</td>
<td>60.5</td>
<td>61.1</td>
<td>56.8</td>
<td>52.4</td>
<td>49.5</td>
<td>30.5</td>
<td>31.7</td>
</tr>
<tr>
<th align="left">Deepseek-VL2-tiny (1B)</th>
<td align="middle">✅</td>
<td>48.8</td>
<td>60.8</td>
<td>34.0</td>
<td>57.1</td>
<td>55.8</td>
<td>63.4</td>
<td>58.4</td>
<td>51.2</td>
<td>57.8</td>
<td>57.6</td>
<td>45.8</td>
<td>54.5</td>
<td>58.3</td>
<td>43.9</td>
<td>47.0</td>
<td>54.4</td>
<td>30.5</td>
<td>31.7</td>
</tr>
<tr>
<th align="left">Phi-3.5-Vision (4.2B)</th>
<td align="middle">✅</td>
<td>42.6</td>
<td>52.2</td>
<td>30.8</td>
<td>53.5</td>
<td>55.8</td>
<td>40.0</td>
<td>49.4</td>
<td>43.3</td>
<td>40.0</td>
<td>53.3</td>
<td>50.6</td>
<td>44.3</td>
<td>46.3</td>
<td>42.8</td>
<td>43.5</td>
<td>44.6</td>
<td>27.6</td>
<td>36.7</td>
</tr>
<tr>
<th align="left">LLaVA-OneVision (0.5B)</th>
<td align="middle">✅</td>
<td>42.3</td>
<td>49.6</td>
<td>33.3</td>
<td>51.8</td>
<td>48.1</td>
<td>47.6</td>
<td>44.8</td>
<td>39.5</td>
<td>50.0</td>
<td>44.6</td>
<td>40.9</td>
<td>49.7</td>
<td>51.9</td>
<td>41.7</td>
<td>44.6</td>
<td>46.1</td>
<td>28.0</td>
<td>31.7</td>
</tr>
<tr>
<th align="left">MiniCPM-V-2.6 (8B)</th>
<td align="middle">✅</td>
<td>41.0</td>
<td>50.4</td>
<td>29.4</td>
<td>50.2</td>
<td>54.8</td>
<td>50.3</td>
<td>53.2</td>
<td>44.7</td>
<td>41.1</td>
<td>52.2</td>
<td>33.7</td>
<td>43.7</td>
<td>48.1</td>
<td>43.6</td>
<td>45.8</td>
<td>46.1</td>
<td>18.2</td>
<td>25.0</td>
</tr>
</table>
</details>

## Settings
```bash
make setup # default: GPU=0 (installs paddlepaddle CPU version); for GPU OCR, run: make setup GPU=1
make help # print manual
```

## Environment (.env)

Create a `.env` file at the project root.

```bash
# Set only what you need
OPENAI_API_KEY=<your API key>
GOOGLE_API_KEY=<your API key>
CLAUDE_API_KEY=<your API key>
```

## Text-Rich VQA Generation

Before running, prepare input images:
- Create `data/images` and place your images there (default `INPUT_DIR`), or set `INPUT_DIR` to your custom folder.

```bash
make filter   # 1) filter out low-quality images with OCR Model
make generate # 2) automatically generate VQA using a 4-stage pipeline (options: INPUT_DIR)
make editor   # 3) refine VQA with streamlit-based editor (options: INPUT_DIR, OUTPUT_DIR, SAVE_BATCH)
```

## Evaluation
`eval` folder contains inference and evaluate scripts for the KRETA.

1. `infer_xxx.py`: For model inference
2. `evaluate.py`: For evaluating inference results

### 1. Model Inference

This script loads a specified model and performs inference. To run the script, use the following steps:

```bash
cd eval
python infer/infer_gpt.py [MODEL_NAME] [SETTING]
```

- **`[MODEL_NAME]`**: Specify the model's name (e.g., `gpt-4o-mini`, `gpt-4o-mini-2024-07-18`, etc.).
- **`[SETTING]`**: Specify the prompt setting (e.g., `default`, `direct`).

Example:

```bash
python infer/infer_gpt.py gpt-4o-mini default
python infer/infer_hf_vlm.py kakaocorp/kanana-1.5-v-3b-instruct default
python infer/infer_hf_vlm.py NCSOFT/VARCO-VISION-2.0-14B default
python infer/infer_hf_vlm.py skt/A.X-4.0-VL-Light default
```

### 2. Evaluation

This script evaluates the results generated from the inference step. To run the evaluation, use the following command:

```bash
cd eval
python evaluate.py
```

Once executed, the script will:
- Load the inference results from the `./output` directory.
- Generate and display the evaluation report in the console.
- Save the evaluation report to the `./output` directory.

## Acknowledgement

- [MMMU-Pro](https://github.com/MMMU-Benchmark/MMMU): we would like to thank the authors for providing the codebase that our work builds upon.
- This work was supported by the Korea Institute for Advancement of Technology (KIAT) grant funded by the Ministry of Education, Korea Government, by Seoul National University (Semiconductor-Specialized University), and by Waddle Corporation, KRAFTON, and AttentionX.


If you find KRETA useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{hwang2025kretabenchmarkkoreanreading,
      title={KRETA: A Benchmark for Korean Reading and Reasoning in Text-Rich VQA Attuned to Diverse Visual Contexts}, 
      author={Taebaek Hwang and Minseo Kim and Gisang Lee and Seonuk Kim and Hyunjun Eun},
      year={2025},
      eprint={2508.19944},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2508.19944}, 
}
```

