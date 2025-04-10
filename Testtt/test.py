from sheng_dholuo_translator import CulturalTranslator

translator = CulturalTranslator()
result = translator.translate("Mambo vipi?", lang_filter="Sheng-English")
print(result['translation'])  # Should print: What’s up?
print(result['vibe'])  # Should print: Casual, greeting
