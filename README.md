# DBLP Paper Crawler

一个可直接运行的 Python 项目，用于从 DBLP 拉取指定会议/期刊在指定年份范围内的论文，基于论文标题做关键词匹配，并尽力补充摘要与作者单位，再通过 OpenAI 兼容接口生成中文总结和研究方向归类，最终导出 CSV。

项目结构固定为：

```text
dblp-paper-crawler/
├── dblp_paper_crawler.py
├── config.yaml
├── requirements.txt
└── README.md
```

## 1. 安装依赖

建议使用 Python 3.9 及以上版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你的环境里 `python` 已经指向 Python 3，也可以直接使用 `python` 运行。

## 2. 配置 `config.yaml`

程序默认会从当前目录读取 `config.yaml`。如果你想使用其他配置文件，也可以通过 `--config /path/to/config.yaml` 显式指定。

核心配置如下：

```yaml
dblp:
  venues:
    ICML:
      enabled: true
      url: "https://dblp.org/db/conf/icml/index.html"
    NeurIPS:
      enabled: true
      url: "https://dblp.org/db/conf/nips/index.html"
    ICLR:
      enabled: true
      url: "https://dblp.org/db/conf/iclr/index.html"
  year_start: 2020
  year_end: 2025

match_rules:
  - ["privacy", "private", "differential privacy"]
  - ["reinforcement learning", "RLHF", "alignment"]

classification:
  categories:
    - "大语言模型安全"
    - "隐私保护"
    - "强化学习对齐"
    - "智能体安全"
    - "数据泄露攻击"
    - "提示注入与越狱"
    - "模型隐私攻击"
    - "差分隐私"
    - "联邦学习安全"
    - "其他"
  allow_new_category: true

openai:
  host: "https://api.openai.com/v1"
  api_key: "YOUR_API_KEY"
  model: "gpt-4o-mini"
  temperature: 0.2
  max_tokens: 800
  max_retries: 3

llm_output:
  title_translation_enabled: true
  summary_language: "zh"

output:
  csv_dir: "./outputs"

cache:
  enabled: true
  path: "./cache/papers_cache.jsonl"

request:
  sleep_seconds: 1
  timeout_seconds: 20
  max_retries: 3
```

说明：

- `dblp.venues`：推荐使用融合后的 `map` 风格配置。键是会议/期刊名，值可以是 `true/false`，也可以是对象，例如 `enabled: true`、`url: ...`、`stream_query: ...`。
- `dblp.year_start` / `dblp.year_end`：抓取年份范围，闭区间。
- `match_rules`：标题关键词匹配规则。可以配置多个规则组，每个 `[]` 内可以写多个关键词。
- `classification.categories`：候选研究方向标签。
- `classification.allow_new_category`：当候选标签都不合适时，是否允许 AI 在 `ai_suggested_category` 中建议一个新类别。
- `openai`：OpenAI 兼容接口参数，必须从配置文件读取，不会硬编码在代码里。
- `llm_output.title_translation_enabled`：是否生成标题中文翻译；关闭后不会翻译，导出的 `标题翻译` 列会写为 `N/A`。
- `llm_output.summary_language`：摘要总结语言，支持 `zh` 或 `en`。
- `output.csv_dir`：导出目录。程序会按每个会议/期刊各生成一个 CSV，并统一写到这个目录下。
- `cache.path`：JSONL 缓存路径，用于断点续爬。
- `request`：外部请求的节流、超时、重试参数。

`dblp` 配置也支持下面这些写法：

```yaml
dblp:
  venues:
    ICML:
      enabled: true
      url: "https://dblp.org/db/conf/icml/index.html"
    NeurIPS:
      stream_query: "streamid:conf/nips:"
```

另外，旧版分离写法和列表写法依然兼容，例如：

```yaml
dblp:
  venues:
    - "ICML"
    - "NeurIPS"
  venue_stream_overrides:
    ICML: "https://dblp.org/db/conf/icml/index.html"
```

## 3. 运行程序

正常运行：

```bash
python3 dblp_paper_crawler.py
```

不调用模型，仅测试爬取、匹配、补充字段和 CSV 输出：

```bash
python3 dblp_paper_crawler.py --no-llm
```

仅处理前 20 篇匹配论文，适合小规模测试：

```bash
python3 dblp_paper_crawler.py --limit 20
```

也可以组合使用：

```bash
python3 dblp_paper_crawler.py --no-llm --limit 20
```

如果配置文件不在当前目录，可以手动指定：

```bash
python3 dblp_paper_crawler.py --config /path/to/config.yaml
```

只测试 AI 接口配置是否有效，不执行抓取：

```bash
python3 dblp_paper_crawler.py --test-ai
```

如果要测试其他配置文件里的 AI 配置：

```bash
python3 dblp_paper_crawler.py --config /path/to/config.yaml --test-ai
```

只从已有缓存继续补齐，不重新抓取新的 DBLP 论文：

```bash
python3 dblp_paper_crawler.py --resume-only
```

也可以配合限制条数一起使用：

```bash
python3 dblp_paper_crawler.py --resume-only --limit 20
```

## 4. 关键词匹配规则说明

`match_rules` 是一个由多个规则组组成的列表。

每个 `[]` 就是一组关键词：

- 同一个 `[]` 里的多个关键词：组内 `OR`
- 多个 `[]` 之间：组间 `AND`

例如：

```yaml
match_rules:
  - ["a", "b"]
  - ["c", "d"]
```

含义是：

```text
(title contains "a" OR title contains "b")
AND
(title contains "c" OR title contains "d")
```

也就是说：

- 你可以只写一个 `[]`，表示只要命中这一组里的任意一个关键词即可。
- 你也可以写多个 `[]`，表示标题必须同时满足多组条件。

实现细节：

- 只对论文标题匹配，不对摘要匹配。
- 匹配不区分大小写。
- 支持短语匹配，例如 `differential privacy`。
- 先做标准化和子串匹配，再使用 `rapidfuzz` 做轻量模糊匹配。
- 如果 `match_rules` 为空，则默认不过滤标题。

## 5. 数据来源与字段说明

程序的数据优先级如下：

1. DBLP：抓取标题、作者、年份、会议/期刊、DBLP 链接、DOI、外部链接。
2. Crossref：优先按 DOI、再按标题补充摘要与作者单位信息。
3. OpenAlex：补充摘要、作者和机构信息。
4. Semantic Scholar：补充摘要与作者单位信息。
5. arXiv：当 DOI 或论文链接指向 arXiv 时，尝试获取 arXiv 摘要。

重要说明：

- DBLP 本身通常不稳定提供摘要和作者单位，所以这些字段在本项目中属于“尽力获取”。
- 摘要和作者单位无法稳定获取时，会明确写为 `N/A`，不会编造。
- 如果某个数据源限流、失败、无字段或页面结构变化，程序会自动跳过并尝试下一个来源，不会因为单篇论文失败而中断整批任务。

## 6. 输出 CSV 说明

最终 CSV 编码为 `utf-8-sig`，方便 Excel 直接打开。

程序会按 `dblp.venues` 中的配置项分别导出，例如：

- `TPAMI.csv`
- `AAAI.csv`
- `NeurIPS.csv`

这些文件都会统一输出到 `output.csv_dir` 指定的目录中。

每个 CSV 的表头顺序如下：

1. `序号`：导出顺序编号。
2. `标题`：论文标题。
3. `标题翻译`：论文标题中文翻译；若 `llm_output.title_translation_enabled: false`，则为 `N/A`。
4. `链接`：论文主页链接，优先使用 DOI 链接，其次回退到论文外部主页，再回退到 DBLP 链接。
5. `作者`：作者列表，使用英文分号 `;` 分隔。
6. `作者单位`：作者单位列表，使用英文分号 `;` 分隔。
7. `年份`：论文年份。
8. `期刊/会议`：期刊或会议名称。
9. `类别`：模型从候选类别中选择的类别；未调用模型或失败时为 `N/A`。
10. `AI建议新类别`：仅当 `category=其他` 且允许提出新类别时才可能有值，否则为 `N/A`。
11. `摘要总结`：模型基于摘要生成的总结；语言由 `llm_output.summary_language` 决定。

虽然 CSV 只输出这些列，但缓存 JSONL 中会保留更多中间字段，例如：

- `doi`
- `paper_url`
- `dblp_url`
- `abstract`
- `abstract_source`
- `abstract_status`
- `summary_text`
- `summary_language`
- `title_translation`
- `reason`
- `llm_status`
- `affiliation_source`
- `affiliation_mode`

这些字段用于断点续爬、问题排查和后续扩展。

## 7. 缓存与断点续爬

缓存默认启用，格式为 JSONL。

行为如下：

- 每处理完一个阶段，都会把中间结果追加写入缓存。
- 程序重启后，会优先复用已有成功结果。
- 如果上一次停在网络失败、解析失败、LLM 失败等中间状态，下一次运行会继续尝试补齐。
- 如果使用 `--resume-only`，程序不会重新向 DBLP 拉新论文，只会从现有缓存中筛出符合当前配置的记录继续处理。
- 去重优先级为：
  - DOI
  - DBLP URL
  - 标准化后的标题

如果你希望重新完整抓取同一批论文，可以删除缓存文件后重新运行。

## 8. OpenAI 兼容接口说明

程序使用 OpenAI Python SDK，并以 OpenAI 兼容方式调用：

- 支持自定义 `host` / `base_url`
- 支持自定义 `model`
- 支持自定义 `temperature`
- 支持自定义 `max_tokens`
- 支持 `max_retries`
- 优先请求 JSON 输出，并在解析失败时做兜底处理
- 可选生成标题中文翻译
- 支持将摘要总结输出为中文或英文

分类规则：

- `category` 必须来自 `classification.categories`。
- 若模型返回了候选类别之外的标签，程序会自动改写为 `其他` 并记日志。
- 若 `allow_new_category: true` 且候选类别都不合适，则 `category=其他`，`AI建议新类别` 可以写入一个简短建议。
- 如果没有摘要但开启了标题翻译，程序仍可只翻译标题；这时 `类别`、`AI建议新类别` 和 `摘要总结` 都会是 `N/A`。
- 如果没有摘要且标题翻译也关闭，则不会调用模型，`类别`、`AI建议新类别` 和 `摘要总结` 都是 `N/A`。
- 如果模型接口调用失败，`类别` 和 `AI建议新类别` 也会是 `N/A`。

如果 OpenAI 兼容接口报错，请重点检查：

- `openai.host` 是否正确，例如是否需要带 `/v1`
- `openai.api_key` 是否有效
- `openai.model` 是否是该服务支持的模型名
- 目标服务是否支持标准 Chat Completions 接口与 JSON 输出

你也可以先运行下面的命令单独测试 AI 配置：

```bash
python3 dblp_paper_crawler.py --test-ai
```

## 9. 日志与进度

程序会使用 `logging` 输出关键日志，包括：

- 当前处理的会议/期刊和年份
- 当前论文标题
- 标题是否命中关键词
- DOI / 论文链接是否存在
- 摘要获取状态与来源
- 作者单位获取状态
- OpenAI 调用状态
- 当前累计可导出的论文数量

匹配论文的后续处理会使用 `tqdm` 显示进度条。

## 10. 注意事项

- 本项目优先保证“稳定、可运行、可恢复”，而不是强依赖某个单一网站页面结构。
- 程序不会编造摘要、作者单位、类别或任何无法稳定获取的数据。
- 缺失字段统一写为 `N/A`。
- 标题匹配只针对标题，不针对摘要。
- 摘要与作者单位属于尽力补充字段，存在拿不到的情况，这属于预期行为。
