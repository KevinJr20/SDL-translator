# Sheng-Dholuo Translator SDK

A cultural nuance translator for Sheng (Kenyan urban slang) and Dholuo (Luo language), built with love and tech swagger. Translate phrases, capture vibes, and even train AI models with your own data.

## Features
- Translation using CSV lookup, fuzzy matching, and AI (`facebook/m2m100_418M`).
- Context-aware translations (`casual`, `formal`, `romantic`).
- Sentiment-based vibe prediction.
- Interactive CLI with commands like `translate`, `add`, `history`, `clear`, `search-vibe`, `random`, `stats`, `export`, `train`.
- Fine-tune the AI model with custom data.

## Upcoming Features
- Integration with advanced AI like Grok for more nuanced translations.
- Real-time data collection from X posts to expand the phrase database.

## Installation

Install via pip:

```bash
pip install sheng-dholuo-translator
```

### Requirements

`Python 3.8+`,
`pandas`, `colorama`, `fuzzywuzzy` (automatically installed)

Usage

#### Translate a Phrase

```python
from sheng_dholuo_translator import CulturalTranslator

translator = CulturalTranslator("phrases.csv")
result = translator.translate("Mambo vipi?", lang_filter="Sheng-English")
print(result['translation'])  # Output: What’s up?
print(result['vibe'])  # Output: Casual, greeting
```

##### Add a New Phrase

```python
translator.add_phrase("Niko freshi", "I’m fresh", "Sheng-English", "Confident, stylish")
```

###### More Features

Reverse translation (English to Sheng/Dholuo)
Search by vibe (e.g., "hype")
Get stats on your phrase collection
Export data for AI training

###### Contributing

Add more Sheng and Dholuo phrases by using the `add_phrase` method or submitting a pull request on GitHub.

###### Author

Built by `Kevin Omondi Jr.` with Kenyan pride and AI dreams.
