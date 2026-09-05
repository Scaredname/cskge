# 环境与验证记录

本记录对应 2026-09-05 本地验证。硬件/软件记录见 `environment.json`，文件行数见 `data_inventory.json`。

## 环境

- 项目内 `.venv`：uv 管理的独立 Python 3.11.16、PyTorch 2.0.0、PyKEEN 1.10.2。
- NumPy 1.26.4、SciPy 1.10.1、pandas 2.0.0、class_resolver 0.5.2。
- `pyproject.toml` 声明依赖，`uv.lock` 锁定完整版本，`.python-version` 固定 Python；`requirements.txt` 仅为兼容导出。
- `uv pip check` 通过，无依赖冲突。
- PyTorch 包包含 CUDA 11.7 runtime，但 `torch.cuda.is_available()` 为 false，`nvidia-smi` 无法连接驱动。CPU 可运行；GPU 完整实验需要主机先提供可用驱动，Python 包安装不能替代内核驱动。
- 项目使用 `PYKEEN_HOME=.cache/pykeen` 作为默认缓存目录；可通过环境变量覆盖。

## 实际执行

| 检查 | 结果 | 覆盖范围 |
| --- | --- | --- |
| `python train.py --help` | 通过 | 全部自定义模块可导入 |
| `python scripts/smoke_test.py` | 四模型通过 | 合成类别图，2 轮、关闭早停、CPU，训练/测试/模型保存 |
| `python scripts/smoke_test.py --epochs 10 --stopper early` | 四模型通过 | 10 轮触发验证，RLRP/早停回调、最佳权重保存、最终测试 |
| 真实 `yago_new` 的 CST | 通过 | 完整 32993 训练三元组、5744 实体、33 关系，1 轮训练和 4092 测试三元组评估 |
| 内置 Nations + TransE | 通过 | 1 轮 CPU 训练、评估、保存，覆盖原先未定义变量的分支 |
| `compileall`、`bash -n`、`git diff --check` | 通过 | Python 语法、安装脚本语法、补丁空白检查 |

真实数据验证命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python train.py \
  -d yago_new -m cs-transe --device cpu -e 1 -b 128 \
  -ed 8 -ced 4 -nen 2 -nenT 2 -eb 8 -stop nop \
  --output-dir models/validation
```

该轮 loss 为约 2.00958，测试集双向 realistic MRR 为约 0.00133260。仅用于确认完整流程，低维单轮结果不代表论文或最佳配置性能。产物在 `models/validation/yago_new/cs-transe/`。

10 轮合成图 smoke 测试最终 loss 均有限，四模型均生成 `results.json` 和 `trained_model.pkl`；详细产物在 `models/smoke/`。脚本生成的临时合成数据在退出时删除，原始数据不变。

## 验证边界

没有执行 16 组 1000 轮完整实验，也没有验证 GPU、精确数值复现、断点续训、类别工厂独立反序列化、分布式/多 worker、候选实体子集评估。NELL/FB/DB 已解压和统计文件，未做全量训练。10 轮测试验证了早停评估与保存路径，未等待 patience 耗尽后的自动终止。

兼容性修复特别是阶段 III 负分数形状修复可能改变旧版结果。原始参数保存在 `configs/experiments.json` 与 `original_experiments.md`；原始源代码仍可通过 Git 查看。算法疑点见 `code_guide.md` 第 7 节。

## uv 迁移

初次验证使用 conda 提供的 Python 3.11.4 创建 venv；现已切换到 uv 0.12.10 下载的独立 CPython 3.11.16，并重建 `.venv`。原来的 10 轮早停及真实 YAGO 验证属于迁移前记录。迁移后重新执行四模型 2 轮 CPU 冒烟测试，覆盖训练、评估和保存。

PyTorch 2.0.0 的跨平台元数据存在差异；`pyproject.toml` 的 dependency-metadata 使用已安装 Linux x86_64 wheel 的原始 Requires-Dist，确保 uv 不漏装 CUDA runtime 依赖。该配置没有更换 PyTorch 或改动模型算法。另将原 venv 自带的 setuptools 65.5.0 显式加入依赖，因为旧版 PyKEEN 仍导入 pkg_resources，新的 setuptools 已不提供该模块。

日常使用 `uv run --locked python ...`；更新依赖使用 `uv add`/`uv lock`，然后执行 `uv sync --locked`。兼容 requirements 文件通过 `uv export --locked --format requirements-txt --no-hashes --no-header --output-file requirements.txt` 生成，不单独维护第二套锁文件。
