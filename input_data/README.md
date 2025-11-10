# Input Data Directory

Place your training and test CSV files here.

## Required Format

Your CSV files should have these columns:
- `id`: Unique identifier
- `example`: Text content
- `Label`: Classification label

Example:
```csv
id,example,Label
1,"i feel hopeless because i don't know what to do",sadness
2,"i am so excited about this amazing opportunity",joy
3,"this situation makes me absolutely furious",anger
```

## Dataset Configuration

Make sure to update `config.yaml` with your file names:

```yaml
dataset:
  train_file: your_train.csv
  test_file: your_test.csv
  columns:
    id: id
    text: example
    label: Label
```

## Using Provided Splits

You can copy the provided emotion dataset splits:

```bash
cp data/splits/emotions_train.csv input_data/
cp data/splits/emotions_test.csv input_data/
```
