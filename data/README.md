# Dataset Directory

Place your experimental observation files and research notes in this directory.

## Expected Files

- `dataset.txt`: Raw experimental observations containing circuit labels, timestamps, binary outcomes, and repetition information.
- `README.txt`: Documentation describing the dataset format, experiment setup, and field descriptions.
- `1_8.md`: Project notes, research plan, or relevant background materials.

## Formats Supported by the Parser

The parser supports common formats for quantum drift benchmark datasets:
1. **Key-Value / Python Dict-like Blocks**:
   Blocks containing `circuit: <label>`, `times: [...]`, `outcomes: [...]`, and optional `repetitions: [...]` or `repetitions: <int>`.
2. **JSON / Line-delimited JSON (JSONL)**:
   JSON objects containing `"circuit"`, `"times"`, `"outcomes"`, and optional `"repetitions"`.
3. **Tabular / Delimited**:
   Rows with columns corresponding to circuit, timestamp, outcome, and repetition count.

Once you upload the files, run:
```bash
python scripts/inspect_dataset.py --file data/dataset.txt
```

