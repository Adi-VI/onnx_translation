# onnx_translation

A Flutter package for offline LLM (large language model) inference, enabling ONNX-based translation models with tokenizer support. Designed for seamless integration with MarianMT models exported from Hugging Face’s Helsinki-NLP collection. Enables efficient offline translation in Flutter apps.

## Features

- Load and run encoder-decoder ONNX translation models.
- Support for tokenizer vocabulary and special token handling.
- Configurable asset paths for models and tokenizer files.
- Compatible with MarianMT models exported to ONNX from Hugging Face.
- Simple API to generate translations with optional initial language tokens.
- Built on top of the official onnxruntime Flutter bindings for fast inference.

## Getting started

### Prerequisites

- Flutter SDK installed
- ONNX MarianMT model files and tokenizer JSONs exported locally
- onnxruntime Flutter package installed in your app

### Installation

Add the package locally or as a dependency in your `pubspec.yaml`:

```yaml
dependencies:
  onnx_translation:
    path: ../onnx_translation
  onnxruntime: ^1.4.1
  ffi: ^2.0.0
```

Declare your model assets in your app’s `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/my_model/encoder_model.onnx
    - assets/my_model/decoder_model.onnx
    - assets/my_model/vocab.json
    - assets/my_model/tokenizer_config.json
    - assets/my_model/generation_config.json
```

Place the corresponding ONNX model and tokenizer files inside `assets/my_model/` (or your chosen folder).
If you do not specify modelBasePath, the package will look for assets under the default folder assets/onnx_model/.

## Usage

Import and initialize the model:

```dart
import 'package:onnx_translation/onnx_translation.dart';

void main() async {
  final model = OnnxModel();
  await model.init(modelBasePath: 'assets/my_model');
  final output = await model.runModel("Hello world", initialLangToken: '>>ara<<');
  print('Translated: $output');
}
```

Use the optional `initialLangToken` to specify target language tokens if your model requires it.

## Model Preparation

This package is tested with Helsinki-NLP MarianMT models exported to ONNX format using Hugging Face’s transformers and optimum libraries.
Use the following Python script to export the ONNX translation model and tokenizer locally:

```python
from transformers import MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from pathlib import Path

model_name_or_path = "Helsinki-NLP/opus-mt-en-hi"

# Load tokenizer from Hugging Face Hub
tokenizer = MarianTokenizer.from_pretrained(model_name_or_path, use_fast=False)

# Load and convert model (automatic ONNX export if missing)
model_onnx = ORTModelForSeq2SeqLM.from_pretrained(
    model_name_or_path,
    local_files_only=False
)

# Prepare save directory
onnx_save_path = Path(__file__).parent.parent / "onnx_model_en_hi"
onnx_save_path.mkdir(exist_ok=True)

# Save ONNX model and tokenizer
model_onnx.save_pretrained(onnx_save_path)
tokenizer.save_pretrained(onnx_save_path)

print(f"ONNX model and tokenizer saved to {onnx_save_path}")
```

### Required files in your model folder

After exporting, the model folder (e.g., onnx_model_en_hi) should contain at minimum the following files:
- encoder_model.onnx — ONNX model file for the encoder
- decoder_model.onnx — ONNX model file for the decoder
- vocab.json — tokenizer vocabulary mapping tokens to IDs
- tokenizer_config.json — tokenizer configuration file
- generation_config.json — generation configuration parameters
- special_tokens_map.json (optional) — special token mappings

Additional files like the following might also be present but are not currently required by this package:
- config.json
- decoder_with_past_model.onnx
- source.spm
- target.spm

## Additional information

Contributions and issues are welcome via GitHub.
Please file issues for bug reports or feature requests.
This project is licensed under the MIT License. See the LICENSE file for details.
