import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

/// A class for running ONNX translation models with tokenizer support.
///
/// This class handles loading ONNX encoder and decoder models,
/// tokenizer vocabulary and configuration files from Flutter assets,
/// and running the translation generation loop with tokenization and detokenization.
///
/// The package itself does not bundle any model or tokenizer files.
/// Instead, it expects the app using this package to provide those assets,
/// and specify their location during initialization.
///
/// Example usage:
/// ```dart
/// final model = OnnxModel();
/// await model.init(modelBasePath: 'assets/my_model_folder');
/// String translated = await model.runModel('Hello world');
/// print(translated);
/// model.release();
/// ```
///
/// The `modelBasePath` parameter in `init` sets the folder where your model assets are.
/// Alternatively, you can specify each asset path explicitly.
///
/// Note: The package depends on the `onnxruntime` Flutter package for running ONNX models.
class OnnxModel {
  late OrtSession _encoderSession;
  late OrtSession _decoderSession;
  late Map<String, int> _vocab;        // token -> id
  late Map<int, String> _reverseVocab; // id -> token

  /// Token IDs loaded from tokenizer/generation configs if available.
  late int eosTokenId;
  late int padTokenId;
  late int unkTokenId;

  /// Regex to detect special tokens like `<2hi>`, `<en>`, etc.
  final RegExp _specialTokenRegex = RegExp(r'<[^>]+>');

  /// Default constructor.
  OnnxModel();

  /// Initializes the ONNX model sessions and loads tokenizer vocabulary and config files.
  ///
  /// The model assets should be located in your Flutter app's assets folder,
  /// and declared in `pubspec.yaml` of your app.
  ///
  /// [modelBasePath]: Optional base path prefix to locate all model assets.
  /// If provided, default asset filenames are appended to this path.
  ///
  /// [encoderAsset]: Optional explicit path to encoder ONNX model asset.
  ///
  /// [decoderAsset]: Optional explicit path to decoder ONNX model asset.
  ///
  /// [vocabAsset]: Optional explicit path to tokenizer vocabulary JSON asset.
  ///
  /// [tokenizerConfigAsset]: Optional explicit path to tokenizer config JSON asset.
  ///
  /// [generationConfigAsset]: Optional explicit path to generation config JSON asset.
  ///
  /// Throws an exception if loading assets or initializing ONNX sessions fails.
  Future<void> init({
    String? modelBasePath,
    String? encoderAsset,
    String? decoderAsset,
    String? vocabAsset,
    String? tokenizerConfigAsset,
    String? generationConfigAsset,
  }) async {
    // Compose asset paths if modelBasePath is provided
    encoderAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}encoder_model.onnx'
        : 'assets/onnx_model/encoder_model.onnx';

    decoderAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}decoder_model.onnx'
        : 'assets/onnx_model/decoder_model.onnx';

    vocabAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}vocab.json'
        : 'assets/onnx_model/vocab.json';

    tokenizerConfigAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}tokenizer_config.json'
        : 'assets/onnx_model/tokenizer_config.json';

    generationConfigAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}generation_config.json'
        : 'assets/onnx_model/generation_config.json';

    OrtEnv.instance.init();

    // Load vocabulary JSON
    final vocabStr = await rootBundle.loadString(vocabAsset);
    final vocabJson = jsonDecode(vocabStr) as Map<String, dynamic>;
    _vocab = vocabJson.map((k, v) => MapEntry(k, (v as num).toInt()));
    _reverseVocab =
        Map.fromEntries(_vocab.entries.map((e) => MapEntry(e.value, e.key)));

    // Initialize token IDs with sane defaults
    eosTokenId = _reverseVocab.entries.firstWhere(
      (e) => e.value == '</s>',
      orElse: () => MapEntry(0, '</s>'),
    ).key;
    unkTokenId = _reverseVocab.entries.firstWhere(
      (e) => e.value == '<unk>',
      orElse: () => MapEntry(1, '<unk>'),
    ).key;
    padTokenId = _reverseVocab.entries.firstWhere(
      (e) => e.value == '<pad>',
      orElse: () => MapEntry((_vocab['<pad>'] ?? -1) >= 0 ? _vocab['<pad>']! : 0, '<pad>'),
    ).key;

    // Attempt to load tokenizer_config.json for pad/unk tokens
    try {
      final tcfg = await rootBundle.loadString(tokenizerConfigAsset);
      final tc = jsonDecode(tcfg) as Map<String, dynamic>;
      if (tc.containsKey('pad_token')) {
        final padTok = tc['pad_token'] as String;
        if (_vocab.containsKey(padTok)) padTokenId = _vocab[padTok]!;
      }
      if (tc.containsKey('unk_token')) {
        final unkTok = tc['unk_token'] as String;
        if (_vocab.containsKey(unkTok)) unkTokenId = _vocab[unkTok]!;
      }
    } catch (_) {
      // Ignore if tokenizer_config.json missing
    }

    // Attempt to load generation_config.json for eos token id or token string
    try {
      final gcfg = await rootBundle.loadString(generationConfigAsset);
      final gc = jsonDecode(gcfg) as Map<String, dynamic>;
      if (gc.containsKey('eos_token_id')) {
        eosTokenId = (gc['eos_token_id'] as num).toInt();
      } else if (gc.containsKey('eos_token')) {
        final tok = gc['eos_token'] as String;
        if (_vocab.containsKey(tok)) eosTokenId = _vocab[tok]!;
      }
    } catch (_) {
      // Ignore if generation_config.json missing
    }

    // Final sanity fallback
    eosTokenId = eosTokenId >= 0 ? eosTokenId : 0;
    padTokenId = padTokenId >= 0 ? padTokenId : eosTokenId;
    unkTokenId = unkTokenId >= 0 ? unkTokenId : 1;

    // Load ONNX encoder and decoder models as bytes from assets
    final encData = await rootBundle.load(encoderAsset);
    final decData = await rootBundle.load(decoderAsset);
    final encBytes = encData.buffer.asUint8List();
    final decBytes = decData.buffer.asUint8List();

    final sessionOptions = OrtSessionOptions();
    _encoderSession = OrtSession.fromBuffer(encBytes, sessionOptions);
    _decoderSession = OrtSession.fromBuffer(decBytes, sessionOptions);
  }

  /// Tokenizes input text into token IDs.
  ///
  /// Supports detecting special tokens enclosed in angle brackets, e.g. `<en>`.
  /// Unknown tokens are mapped to the unknown token ID.
  ///
  /// Returns a list of token IDs including an appended end-of-sequence token.
  List<int> tokenize(String text) {
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (normalized.isEmpty) return [eosTokenId];

    final tokenIds = <int>[];
    final tokens = _vocab.keys.toList()..sort((a, b) => b.length.compareTo(a.length));

    int pos = 0;
    while (pos < normalized.length) {
      final match = _specialTokenRegex.matchAsPrefix(normalized, pos);
      if (match != null) {
        final specialTok = match.group(0)!;
        if (_vocab.containsKey(specialTok)) {
          tokenIds.add(_vocab[specialTok]!);
          pos += specialTok.length;
          while (pos < normalized.length && normalized[pos] == ' ') {
            pos++;
          }
          continue;
        }
      }

      bool matched = false;
      for (final tok in tokens) {
        if (tok.length > normalized.length - pos) continue;
        final substr = normalized.substring(pos, pos + tok.length);

        if (substr == tok) {
          tokenIds.add(_vocab[tok]!);
          pos += tok.length;
          matched = true;
          break;
        }
        if (tok.startsWith('▁')) {
          final plainTok = tok.substring(1);
          if ((pos == 0 || normalized[pos - 1] == ' ') &&
              normalized.substring(pos, pos + plainTok.length) == plainTok) {
            tokenIds.add(_vocab[tok]!);
            pos += plainTok.length;
            matched = true;
            break;
          }
        }
      }

      if (!matched) {
        tokenIds.add(unkTokenId);
        pos += 1;
      } else {
        while (pos < normalized.length && normalized[pos] == ' ') {
          pos++;
        }
      }
    }

    tokenIds.add(eosTokenId);
    return tokenIds;
  }

  /// Applies softmax function on the list of logits.
  List<double> softmax(List<double> logits) {
    final maxValue = logits.reduce(max);
    final exps = logits.map((l) => exp(l - maxValue)).toList();
    final sumExp = exps.fold<double>(0.0, (a, b) => a + b);
    return exps.map((e) => e / sumExp).toList();
  }

  /// Checks if a token string is punctuation.
  bool _isPunctuation(String token) {
    const punctuations = {
      '.', ',', '!', '?', ':', ';', '-', '—', '(', ')', '[', ']', '"', '\''
    };
    return punctuations.contains(token);
  }

  /// Converts token IDs back into a readable string.
  ///
  /// Removes end-of-sequence tokens and handles spacing around punctuation and special tokens.
  String detokenize(List<int> tokenIds) {
    final tokens = tokenIds.map((id) => _reverseVocab[id] ?? '<unk>').toList();
    final buffer = StringBuffer();

    for (int i = 0; i < tokens.length; i++) {
      final tok = tokens[i];

      if (tok == '</s>') continue;

      if (tok.startsWith('<') && tok.endsWith('>')) {
        if (buffer.isNotEmpty) buffer.write(' ');
        buffer.write(tok);
        continue;
      }

      if (tok.startsWith('▁')) {
        if (buffer.isNotEmpty) buffer.write(' ');
        buffer.write(tok.substring(1));
      } else if (_isPunctuation(tok)) {
        buffer.write(tok);
      } else {
        buffer.write(tok);
      }
    }

    return buffer.toString().trim();
  }

  /// Runs the ONNX translation model on the input text and generates output text.
  ///
  /// [inputText]: The input string to translate or generate text from.
  ///
  /// [initialLangToken]: Optional language token prefix to prepend to input (e.g., '>>ara<<').
  ///
  /// [maxNewTokens]: Maximum tokens to generate (default is 50).
  ///
  /// Returns the generated string output after decoding.
  ///
  /// Throws if the ONNX runtime sessions fail during execution.
  Future<String> runModel(
    String inputText, {
    String? initialLangToken,
    int maxNewTokens = 50,
  }) async {
    String textToTokenize = inputText;
    if (initialLangToken != null && initialLangToken.isNotEmpty) {
      textToTokenize = '$initialLangToken $inputText';
    }

    final inputIds = tokenize(textToTokenize);
    final seqLen = inputIds.length;

    final attentionMask = List<int>.filled(seqLen, 1);

    final inputTensor =
        OrtValueTensor.createTensorWithDataList(inputIds, [1, seqLen]);
    final attentionMaskTensor =
        OrtValueTensor.createTensorWithDataList(attentionMask, [1, seqLen]);

    final encoderInputs = {
      'input_ids': inputTensor,
      'attention_mask': attentionMaskTensor,
    };

    final encoderOutputs =
        await _encoderSession.runAsync(OrtRunOptions(), encoderInputs);
    final encoderHiddenStates = encoderOutputs![0];

    final decoderInputIds = <int>[padTokenId];
    final generatedIds = <int>[];

    for (int step = 0; step < maxNewTokens; step++) {
      final decInputTensor = OrtValueTensor.createTensorWithDataList(
          decoderInputIds, [1, decoderInputIds.length]);

      final decoderInputs = {
        'input_ids': decInputTensor,
        'encoder_hidden_states': encoderHiddenStates!,
        'encoder_attention_mask': attentionMaskTensor,
      };

      final decoderOutputs =
          await _decoderSession.runAsync(OrtRunOptions(), decoderInputs);

      final logitsTensor = decoderOutputs![0];
      if (logitsTensor == null || logitsTensor.value == null) break;

      final raw = logitsTensor.value as List<dynamic>;
      final lastStepLogits =
          (raw[0] as List<dynamic>).last as List<dynamic>; // List<double>

      final logits = lastStepLogits.map((e) => (e as num).toDouble()).toList();
      final probs = softmax(logits);

      int nextToken = 0;
      double best = double.negativeInfinity;
      for (int i = 0; i < probs.length; i++) {
        if (probs[i] > best) {
          best = probs[i];
          nextToken = i;
        }
      }

      generatedIds.add(nextToken);

      if (nextToken == eosTokenId) break;

      decoderInputIds.add(nextToken);
    }

    final translated = detokenize(generatedIds);
    return translated;
  }

  /// Releases all ONNX resources used by this instance.
  ///
  /// After calling this method, the model instance should not be used.
  void release() {
    try {
      _encoderSession.release();
      _decoderSession.release();
      OrtEnv.instance.release();
    } catch (_) {}
  }
}
