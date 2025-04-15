import os
import pandas as pd
import random
from colorama import init, Fore, Style
from fuzzywuzzy import fuzz
init()


class CulturalTranslator:
    def __init__(self):
        self.csv_file = os.path.join(os.path.dirname(__file__), "phrases.csv")
        self.phrases = pd.read_csv(self.csv_file)

    def translate(self, source_phrase, lang_filter=None, reverse=False):
        df = self.phrases
        if lang_filter:
            df = df[df['Language Pair'] == lang_filter]

        source_col = 'Target Phrase' if reverse else 'Source Phrase'
        target_col = 'Source Phrase' if reverse else 'Target Phrase'

        result = df[df[source_col].str.lower() == source_phrase.lower()]
        if not result.empty:
            row = result.iloc[0]
            return {
                "translation": row[target_col],
                "vibe": row['Vibe Note'],
                "language_pair": row['Language Pair']
            }

        suggestions = []
        for _, row in df.iterrows():
            score = fuzz.partial_ratio(source_phrase.lower(), row[source_col].lower())
            if score > 50:
                suggestions.append((score, row[source_col], row[target_col]))
        suggestions.sort(reverse=True)
        if suggestions:
            return {"suggestions": [{"score": s[0], "source": s[1], "target": s[2]} for s in suggestions[:3]]}
        return {"error": "Phrase not found!"}

    def search_by_vibe(self, vibe_keyword):
        matches = self.phrases[self.phrases['Vibe Note'].str.lower().str.contains(vibe_keyword.lower(), na=False)]
        if not matches.empty:
            return matches[['Source Phrase', 'Target Phrase', 'Language Pair', 'Vibe Note']].to_dict('records')
        return {"error": "No vibes match that keyword!"}

    def add_phrase(self, source_phrase, target_phrase, lang_pair, vibe):
        new_row = pd.DataFrame({
            'Source Phrase': [source_phrase],
            'Target Phrase': [target_phrase],
            'Language Pair': [lang_pair],
            'Vibe Note': [vibe]
        })
        self.phrases = pd.concat([self.phrases, new_row], ignore_index=True)
        self.phrases.to_csv(self.csv_file, index=False)
        return {"success": "Phrase added!"}

    def get_random_phrase(self):
        row = self.phrases.sample(1).iloc[0]
        return {
            "source": row['Source Phrase'],
            "translation": row['Target Phrase'],
            "vibe": row['Vibe Note'],
            "language_pair": row['Language Pair']
        }

    def get_stats(self):
        total_phrases = len(self.phrases)
        sheng_count = len(self.phrases[self.phrases['Language Pair'] == 'Sheng-English'])
        dholuo_count = len(self.phrases[self.phrases['Language Pair'] == 'Dholuo-English'])
        vibe_counts = self.phrases['Vibe Note'].value_counts().to_dict()
        return {
            "total_phrases": total_phrases,
            "sheng_count": sheng_count,
            "dholuo_count": dholuo_count,
            "vibe_breakdown": vibe_counts
        }

    def export_training_data(self, output_file="training_data.json"):
        data = self.phrases[['Source Phrase', 'Target Phrase', 'Language Pair']].to_dict('records')
        pd.DataFrame(data).to_json(output_file, orient='records', indent=4)
        return {"success": f"Exported to {output_file}!"}


if __name__ == "__main__":
    translator = CulturalTranslator()
    print(f"{Fore.CYAN}Welcome to the Sheng-Dholuo Translator, bro!{Style.RESET_ALL}")
    while True:
        print(f"\n{Fore.YELLOW}Options:{Style.RESET_ALL}")
        print("1. Translate a phrase")
        print("2. Search by vibe")
        print("3. Add a new phrase")
        print("4. Reverse translate (English to Sheng/Dholuo)")
        print("5. Get a random phrase")
        print("6. Show stats")
        print("7. Export training data")
        print("8. Quit")
        choice = input("What’s your move? (1/2/3/4/5/6/7/8): ")

        if choice == '8':
            print(f"{Fore.GREEN}Catch you later, bro! Usitense!{Style.RESET_ALL}")
            break

        elif choice == '1':
            phrase = input("Enter a Sheng or Dholuo phrase: ")
            lang = input("Filter by language? (Sheng-English, Dholuo-English, or leave blank): ").strip() or None
            result = translator.translate(phrase, lang, reverse=False)

            if "translation" in result:
                print(f"{Fore.GREEN}Translation: {result['translation']}{Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}Vibe: {result['vibe']}{Style.RESET_ALL}")
                print(f"Language: {result['language_pair']}")
            elif "suggestions" in result:
                print(f"{Fore.YELLOW}No exact match, but did you mean one of these?{Style.RESET_ALL}")
                for sug in result['suggestions']:
                    print(f"- {sug['source']} → {sug['target']} (Similarity: {sug['score']}%)")
            else:
                print(f"{Fore.RED}{result['error']}{Style.RESET_ALL}")

        elif choice == '2':
            vibe = input("Enter a vibe keyword (e.g., hype, calm): ")
            result = translator.search_by_vibe(vibe)
            if "error" in result:
                print(f"{Fore.RED}{result['error']}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Phrases with '{vibe}' vibes:{Style.RESET_ALL}")
                for match in result:
                    print(f"- {match['Source Phrase']} → {match['Target Phrase']} ({match['Vibe Note']})")

        elif choice == '3':
            source = input("Enter the Sheng/Dholuo phrase: ")
            target = input("Enter the English translation: ")
            lang = input("Language pair (Sheng-English or Dholuo-English): ")
            vibe = input("What’s the vibe? (e.g., casual, hype): ")
            result = translator.add_phrase(source, target, lang, vibe)
            print(f"{Fore.GREEN}{result['success']}{Style.RESET_ALL}")

        elif choice == '4':
            phrase = input("Enter an English phrase: ")
            lang = input("Target language? (Sheng-English, Dholuo-English, or leave blank): ").strip() or None
            result = translator.translate(phrase, lang, reverse=True)

            if "translation" in result:
                print(f"{Fore.GREEN}Translation: {result['translation']}{Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}Vibe: {result['vibe']}{Style.RESET_ALL}")
                print(f"Language: {result['language_pair']}")
            elif "suggestions" in result:
                print(f"{Fore.YELLOW}No exact match, but did you mean one of these?{Style.RESET_ALL}")
                for sug in result['suggestions']:
                    print(f"- {sug['source']} → {sug['target']} (Similarity: {sug['score']}%)")
            else:
                print(f"{Fore.RED}{result['error']}{Style.RESET_ALL}")

        elif choice == '5':
            result = translator.get_random_phrase()
            print(f"{Fore.CYAN}Random Phrase:{Style.RESET_ALL}")
            print(f"- {result['source']} → {result['translation']}")
            print(f"Vibe: {result['vibe']}")
            print(f"Language: {result['language_pair']}")

        elif choice == '6':
            stats = translator.get_stats()
            print(f"{Fore.CYAN}Translator Stats:{Style.RESET_ALL}")
            print(f"Total Phrases: {stats['total_phrases']}")
            print(f"Sheng Phrases: {stats['sheng_count']}")
            print(f"Dholuo Phrases: {stats['dholuo_count']}")
            print(f"Vibe Breakdown:")
            for vibe, count in stats['vibe_breakdown'].items():
                print(f"- {vibe}: {count}")

        elif choice == '7':
            result = translator.export_training_data()
            print(f"{Fore.GREEN}{result['success']}{Style.RESET_ALL}")
