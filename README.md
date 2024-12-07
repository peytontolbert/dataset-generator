# PyTorch Code Example Generator

A sophisticated Python script that generates diverse PyTorch code examples using OpenAI's GPT models. This tool is designed to create high-quality, production-ready PyTorch code examples for training purposes.

## Features

- 🤖 Leverages OpenAI's GPT models for code generation
- 🔍 Uses FAISS for similarity detection to ensure unique examples
- 📊 Hierarchical category-based example generation
- 🔄 Automatic duplicate detection and refinement
- 💾 Organized output in JSONL format
- 🎯 Customizable generation parameters

## Prerequisites
```bash
pip install openai faiss-cpu python-dotenv numpy
```


## Setup

1. Clone this repository
2. Create a `.env` file in the root directory
3. Add your OpenAI API key to the `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```


## Configuration

Key parameters in `main.py`:

- `MODEL_NAME`: The OpenAI model to use (default: "gpt-4o-mini")
- `EMBEDDING_MODEL`: Model for creating embeddings (default: "text-embedding-ada-002")
- `BATCH_SIZE`: Number of examples to generate per batch (default: 20)
- `SIMILARITY_THRESHOLD`: Threshold for duplicate detection (default: 0.9)
- `DATASET_OUTPUT_DIR`: Output directory for generated examples

## Usage

Run the script:
```bash
python main.py
```


The script will:
1. Generate a hierarchical roadmap for code examples
2. Create examples for each subcategory
3. Check for duplicates using FAISS
4. Save unique examples to JSONL files

## Output Structure

Generated examples are saved in the `dataset_output` directory as JSONL files:
```
dataset_output/
├── Category_Name_UUID1.jsonl
├── Category_Name_UUID2.jsonl
└── ...
```


Each JSONL entry contains:
- `example`: The generated code example
- `subcategory`: The subcategory name
- `timestamp`: Generation timestamp

## Features in Detail

### Roadmap Generation
- Automatically creates a hierarchical structure for example generation
- Customizable categories and subcategories

### Duplicate Detection
- Uses FAISS index for efficient similarity search
- Automatically refines similar examples to ensure uniqueness

### Quality Control
- Entropy evaluation using LLM
- Example refinement based on quality assessment
- Production-ready code generation

## Contributing

Feel free to submit issues and enhancement requests!

## License

[MIT License](LICENSE)