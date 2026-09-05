# CSKGE：类别信息增强的知识图谱嵌入实验

包含 TransE、RotatE、CS-TransE（CST）、CS-RotatE（CSR），以及四套本地数据。

## 快速开始

项目使用 **uv 管理 Python、虚拟环境和依赖**，不依赖 conda。Python 固定为 3.11.16；`pyproject.toml` 声明依赖，`uv.lock` 锁定完整版本。uv 的项目工作流见 [官方文档](https://docs.astral.sh/uv/guides/projects/)。

当前机器已安装 uv；若当前终端还找不到命令，先运行 `source ~/.local/bin/env`。直接启动，无需激活环境：

```bash
uv run --locked python scripts/smoke_test.py
```

新机器先[安装 uv](https://docs.astral.sh/uv/getting-started/installation/)，再执行：

```bash
bash scripts/setup_env.sh
```

脚本使用 uv 下载独立 Python、按锁文件同步 `.venv`、解压数据并检查依赖。当前锁定并验证 Linux x86_64 / Python 3.11。也可手动执行 `uv sync --locked`。原先的 PyKEEN 开发版依赖改为经运行验证的 `1.10.2`，不宣称与原开发版数值完全一致。发布信息见 [PyKEEN PyPI](https://pypi.org/project/pykeen/1.10.2/)。

## 运行实验

先用小参数验证真实数据流程：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked python train.py \
  -d yago_new -m cs-transe --device cpu -e 1 -b 128 \
  -ed 8 -ced 4 -nen 2 -nenT 2 -eb 8 -stop nop
```

运行原始实验配置（16 组参数均保存在 `configs/experiments.json`）：

```bash
uv run --locked python scripts/run_experiment.py yago_new/cs-transe --device cuda --random_seed 42
uv run --locked python scripts/run_experiment.py --help
uv run --locked python train.py --help
```

命令尾部参数可以覆盖预设，例如 `-e 10 -ed 32 -b 64`。默认 `--device auto` 自动选择可用的 CUDA 或 CPU；当前机器 NVIDIA 驱动不可用，已验证 CPU 流程。原始预设为 1000 轮和较大负采样量，未执行完整复现。

## 目录

| 路径 | 用途 |
| --- | --- |
| `pyproject.toml` / `uv.lock` / `.python-version` | uv 依赖声明、完整锁文件、独立 Python 版本 |
| `train.py` | 参数解析、模型与训练器组装、运行和保存 |
| `utilities.py` | 数据读取与类别三元组拆分 |
| `customize/` | 模型、三阶段训练、采样、早停和 pipeline 扩展 |
| `configs/experiments.json` | 原 README 的全部实验预设 |
| `scripts/` | 环境安装、解压、实验启动、冒烟测试 |
| `docs/` | 中文代码导读、环境记录、数据清单、验证报告 |
| `data.zip` / `data/` | 原始压缩包 / 解压后的四套数据 |
| `models/` | 按数据集、模型和时间保存实验产物 |
| `.venv/` / `.cache/` | 独立环境 / 项目缓存，均不入 Git |

结果目录含 `results.json`（loss、排名指标）、`trained_model.pkl`、`training_triples/`、`metadata.json`、`config.json` 和 `args.json`（实际命令参数）。测试产物位于 `models/smoke/` 和 `models/validation/`。

阅读入口：[代码导读](docs/code_guide.md)、[验证与限制](docs/validation.md)、[原始实验命令](docs/original_experiments.md)。
