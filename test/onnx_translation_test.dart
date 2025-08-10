import 'package:flutter_test/flutter_test.dart';
import 'package:onnx_translation/onnx_translation.dart';

void main() {
  test('OnnxModel can be instantiated', () {
    final model = OnnxModel();
    expect(model, isA<OnnxModel>());
  });

  test('OnnxModel initialization completes', () async {
    final model = OnnxModel();
    // We only test that init completes without throwing here.
    // You may need to mock or skip actual asset loading in unit tests.
    await model.init();
    expect(true, isTrue); // Just confirm no exceptions
  });
}
