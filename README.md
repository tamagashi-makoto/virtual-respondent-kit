# virtual-respondent-kit

A library for simulating user personas using LLMs. You can execute surveys, A/B tests, and depth interviews using reproducible virtual personalities.

---

## Project Structure

```
persona-agent-simulation/
├── src/persona_sim/       # Library modules (Reusable)
│   ├── config.py          # Config loader (Supports env vars)
│   ├── llm.py             # LLM client initialization
│   ├── prompts.py         # Prompt templates
│   ├── data.py            # Data preparation module
│   ├── survey.py          # Survey execution module
│   ├── ab_test.py         # A/B test execution module
│   └── interview.py       # Interview execution module
│
├── examples/              # Execution scripts (CLI)
│   ├── prepare_data.py    # Prepare data
│   ├── run_survey.py      # Run survey
│   ├── run_ab_test.py     # Run A/B test
│   └── run_interview.py   # Run interview
│
├── config.yaml            # Configuration file
├── .env.example           # Environment variable template
├── requirements.txt       # Dependencies
└── data/                  # Persona data storage

```

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit the .env file to set your Azure OpenAI API Key

```

---

## Quick Start

### 1. Data Preparation

Download and sample persona data from Hugging Face.

```bash
# Default (100 records)
python examples/prepare_data.py

# Specify sample size
python examples/prepare_data.py --sample-size 50 --output data/personas_50.json

```

### 2. Surveys

Conduct a single-question Q&A survey with the personas.

```bash
# Run with default settings
python examples/run_survey.py

# Specify input/output files
python examples/run_survey.py --input data/personas_50.json --output output/survey_50.csv

```

### 3. A/B Testing

Compare receptivity between two options (Plan A vs. Plan B).

```bash
# Run with default settings
python examples/run_ab_test.py

# Specify number of concurrent executions
python examples/run_ab_test.py --concurrent 5

```

### 4. Depth Interviews

An AI interviewer asks the personas deep-dive questions.

```bash
# Run with default settings
python examples/run_interview.py

# Specify max turns and concurrency
python examples/run_interview.py --max-turns 5 --concurrent 3

```

---

## Module Details

### Common Modules

| Module | Description |
| --- | --- |
| `config.py` | Loads configuration. Supports `CONFIG_PATH` and `AZURE_OPENAI_*` environment variables. |
| `llm.py` | Initializes the Azure OpenAI Chat model. |
| `prompts.py` | Generates persona prompts and interviewer prompts. |

### Functional Modules

| Module | Class | Description |
| --- | --- | --- |
| `data.py` | `prepare_persona_data()` | Fetches and samples data from Hugging Face. |
| `survey.py` | `SurveyRunner` | Executes surveys (Single Q&A). |
| `ab_test.py` | `ABTestRunner` | Executes A/B tests (Uses LangGraph). |
| `interview.py` | `InterviewRunner` | Executes depth interviews (Uses LangGraph). |

---

## Configuration

### `config.yaml`

Manages runtime settings. Specify API keys, model parameters, I/O paths, and question text here.

```yaml
azure:
  endpoint: "your_endpoint"
  api_key: "your_api_key"
  api_version: "2025-04-01-preview"
  deployment_name: "gpt-5"

model_params:
  temperature: 1
  max_completion_tokens: 400

survey:
  input_file: "data/personas_100.json"
  output_dir: "output"
  output_file: "survey_results.csv"
  question: |
    [Question]
    Please share your opinion.

ab_test:
  input_file: "data/personas_100.json"
  output_file: "output/ab_test_results.csv"
  plan_a: |
    [Plan A]
    ...
  plan_b: |
    [Plan B]
    ...

interview:
  input_file: "data/personas_100.json"
  output_file: "output/interview_results.csv"
  max_turns: 3
  concurrent_limit: 5
  initial_question: |
    ...

```

### `.env`

You can override settings using environment variables (recommended for secrets like API keys).

```bash
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5
CONFIG_PATH=./config.yaml

```

---

## CLI Options

### Common Options

| Option | Description |
| --- | --- |
| `--config` | Path to config file (default: `./config.yaml`) |
| `--input` | Input JSON file (overrides `config.yaml`) |
| `--output` | Output CSV file (overrides `config.yaml`) |

### Script-Specific Options

| Script | Option | Description |
| --- | --- | --- |
| `prepare_data.py` | `--sample-size` | Sample size |
|  | `--output` | Output file path |
| `run_ab_test.py` | `--concurrent` | Number of concurrent executions |
| `run_interview.py` | `--max-turns` | Max number of turns |
|  | `--concurrent` | Number of concurrent executions |

---

## Usage Examples

### Using as a Library

```python
from persona_sim import load_config, SurveyRunner

# Load config
config = load_config()

# Run survey
runner = SurveyRunner(config)
results = runner.run(
    input_file="data/personas_100.json",
    output_file="output/survey_results.csv"
)

```

### Running for a Single Persona

```python
from persona_sim import load_config
from persona_sim.survey import SurveyRunner

config = load_config()
runner = SurveyRunner(config)

persona = {"age": 30, "sex": "Male", "occupation": "Engineer", ...}
result = runner.run_single(persona, "Question text")
print(result)

```

---

## Output Format

Each execution script outputs a CSV file to the `output/` directory.

### Survey Output Example

| ID | Age | Sex | Occupation | Survey_Answer |
| --- | --- | --- | --- | --- |
| uuid | 30 | Male | Engineer | Answer content... |

### A/B Test Output Example

| ID | Age | Score_A | Score_B | Winner | Reason |
| --- | --- | --- | --- | --- | --- |
| uuid | 30 | 8 | 5 | A | Reason... |

### Interview Output Example

| ID | Age | Conversation_Log | Final_Answer |
| --- | --- | --- | --- |
| uuid | 30 | Conversation history... | Final answer... |

---
