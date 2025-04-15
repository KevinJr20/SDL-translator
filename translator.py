import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import random
import argparse
import json
from colorama import init, Fore, Style
from fuzzywuzzy import fuzz
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
init()

class CulturalTranslator:
    def __init__(self):
        self.csv_file = os.path.join(os.path.dirname(__file__), "phrases.csv")
        self.phrases = pd.read_csv(self.csv_file)
        self.model_name = "facebook/m2m100_418M"
        self.fine_tuned_model_path = "./trained_model"
        self.base_model_cache_path = "./base_model_cache"  # Cache for base model
        self.tokenizer = None
        self.model = None
        self.history_file = "translation_history.json"
        self.load_model()
        self.load_history()

    def load_model(self):
        try:
            if os.path.exists(self.fine_tuned_model_path):
                print(f"{Fore.CYAN}Loading fine-tuned model from {self.fine_tuned_model_path}{Style.RESET_ALL}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.fine_tuned_model_path)
            else:
                print(f"{Fore.YELLOW}Fine-tuned model not found.{Style.RESET_ALL}")
                if os.path.exists(self.base_model_cache_path):
                    print(f"{Fore.CYAN}Loading cached base model from {self.base_model_cache_path}{Style.RESET_ALL}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_cache_path)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_cache_path)
                else:
                    print(f"{Fore.YELLOW}Downloading base model {self.model_name} (this may take a while)...{Style.RESET_ALL}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                    print(f"{Fore.CYAN}Caching base model to {self.base_model_cache_path}{Style.RESET_ALL}")
                    self.tokenizer.save_pretrained(self.base_model_cache_path)
                    self.model.save_pretrained(self.base_model_cache_path)
                print(f"{Fore.YELLOW}Consider training the model (Option 8) for better translations!{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {e}{Style.RESET_ALL}")
            self.model = None

    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            else:
                self.history = []
        except Exception as e:
            print(f"{Fore.RED}Error loading history: {e}{Style.RESET_ALL}")
            self.history = []

    def save_history(self, entry):
        try:
            self.history.append(entry)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            print(f"{Fore.RED}Error saving history: {e}{Style.RESET_ALL}")

    def clear_history(self):
        try:
            self.history = []
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
            return {"success": "Translation history cleared!"}
        except Exception as e:
            return {"error": f"Failed to clear history: {e}"}

    def predict_vibe(self, translation):
        translation = translation.lower()
        # Expanded vibe detection with Sheng and Dholuo keywords
        score_calm = sum(1 for word in ["good", "okay", "fine", "peace", "ber", "maber"] if word in translation)
        score_hype = sum(1 for word in ["cool", "hype", "arrived", "party", "noma", "poa"] if word in translation)
        score_stressed = sum(1 for word in ["hungry", "trouble", "fool", "kwea"] if word in translation)
        score_funny = sum(1 for word in ["funny", "joke", "cheka", "kicheko"] if word in translation)
        score_romantic = sum(1 for word in ["love", "babe", "honey", "papa", "mrembo"] if word in translation)

        scores = {
            "Calm, positive": score_calm,
            "Hype, energetic": score_hype,
            "Desperate, stressed": score_stressed,
            "Funny": score_funny,
            "Romantic": score_romantic
        }
        max_vibe = max(scores, key=scores.get)
        max_score = scores[max_vibe]
        return max_vibe if max_score > 0 else "Neutral"

    def detect_language(self, phrase):
        phrase = phrase.lower()
        # Simple heuristic for language detection
        sheng_keywords = ["mambo", "noma", "poa", "vipi", "fiti", "msee"]
        dholuo_keywords = ["ber", "awinjo", "maber", "ni", "kwe", "chunya"]

        sheng_score = sum(1 for word in sheng_keywords if word in phrase)
        dholuo_score = sum(1 for word in dholuo_keywords if word in phrase)

        if sheng_score > dholuo_score:
            return "Sheng"
        elif dholuo_score > sheng_score:
            return "Dholuo"
        else:
            # If scores are tied or no keywords match, default to Sheng (more common in mixed contexts)
            return "Sheng"

    def train_model(self, training_data_file="training_data.json", output_dir="./trained_model"):
        if not self.model or not self.tokenizer:
            print(f"{Fore.RED}Model not loaded. Cannot train.{Style.RESET_ALL}")
            return {"error": "Model not loaded"}

        print(f"{Fore.CYAN}Loading training data from {training_data_file}{Style.RESET_ALL}")
        try:
            data = pd.read_json(training_data_file)
            dataset = Dataset.from_pandas(data)
            print(f"Dataset loaded with {len(dataset)} examples.")
        except Exception as e:
            print(f"{Fore.RED}Error loading training data: {e}{Style.RESET_ALL}")
            return {"error": "Failed to load training data"}

        def tokenize(batch):
            inputs = self.tokenizer(batch['Source Phrase'], padding="max_length", truncation=True, max_length=128, src_lang="en")
            targets = self.tokenizer(batch['Target Phrase'], padding="max_length", truncation=True, max_length=128, src_lang="en")
            return {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": targets.input_ids
            }

        print(f"{Fore.CYAN}Tokenizing dataset...{Style.RESET_ALL}")
        tokenized_dataset = dataset.map(tokenize, batched=True)
        print(f"Tokenization complete.")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            learning_rate=5e-5,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        print(f"{Fore.CYAN}Starting training...{Style.RESET_ALL}")
        trainer.train()
        print(f"{Fore.GREEN}Training complete!{Style.RESET_ALL}")

        print(f"{Fore.CYAN}Saving model to {output_dir}{Style.RESET_ALL}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"{Fore.GREEN}Model saved successfully.{Style.RESET_ALL}")

        return {"success": f"Model trained and saved to {output_dir}"}

    def translate(self, source_phrase, lang_filter=None, reverse=False, use_ai=False):
        if use_ai and self.model and self.tokenizer:
            if reverse:
                src_lang = "en"
                if lang_filter == "Sheng-English":
                    tgt_lang = "sw"
                elif lang_filter == "Dholuo-English":
                    tgt_lang = "sw"
                else:
                    tgt_lang = "sw"
            else:
                tgt_lang = "en"
                # Detect language to set src_lang more accurately
                detected_lang = self.detect_language(source_phrase)
                if detected_lang == "Sheng":
                    src_lang = "sw"  # Sheng is Swahili-based
                elif detected_lang == "Dholuo":
                    src_lang = "sw"  # Still using Swahili as a proxy, but we can improve this later
                else:
                    src_lang = "sw"  # Default to Swahili

            self.tokenizer.src_lang = src_lang
            input_ids = self.tokenizer(source_phrase, return_tensors="pt", padding=True, truncation=True).input_ids
            outputs = self.model.generate(input_ids, forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang))
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            vibe = self.predict_vibe(translation)
            result = {
                "translation": translation,
                "vibe": f"AI-generated ({vibe})",
                "language_pair": lang_filter or "Unknown"
            }
            self.save_history({"source": source_phrase, "result": result})
            return result

        df = self.phrases
        if lang_filter:
            df = df[df['Language Pair'] == lang_filter]
        
        source_col = 'Target Phrase' if reverse else 'Source Phrase'
        target_col = 'Source Phrase' if reverse else 'Target Phrase'
        
        result = df[df[source_col].str.lower() == source_phrase.lower()]
        if not result.empty:
            row = result.iloc[0]
            result = {
                "translation": row[target_col],
                "vibe": row['Vibe Note'],
                "language_pair": row['Language Pair']
            }
            self.save_history({"source": source_phrase, "result": result})
            return result
        
        suggestions = []
        for _, row in df.iterrows():
            score = fuzz.partial_ratio(source_phrase.lower(), row[source_col].lower())
            if score > 50:
                suggestions.append((score, row[source_col], row[target_col]))
        suggestions.sort(reverse=True)
        if suggestions:
            result = {"suggestions": [{"score": s[0], "source": s[1], "target": s[2]} for s in suggestions[:3]]}
            self.save_history({"source": source_phrase, "result": result})
            return result
        result = {"error": "Phrase not found!"}
        self.save_history({"source": source_phrase, "result": result})
        return result
    
    def batch_translate(self, phrases, lang_filter=None, reverse=False, use_ai=False, export_file=None):
        results = []
        for phrase in phrases:
            result = self.translate(phrase, lang_filter, reverse, use_ai)
            results.append({"phrase": phrase, "result": result})
        
        if export_file:
            try:
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
                print(f"{Fore.GREEN}Exported translations to {export_file}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error exporting translations: {e}{Style.RESET_ALL}")
        
        return results
    
    def recommend_similar_vibe(self, vibe):
        matches = self.phrases[self.phrases['Vibe Note'].str.lower().str.contains(vibe.lower(), na=False)]
        if not matches.empty:
            row = matches.sample(1).iloc[0]  # Fixed the bug
            return {
                "source": row['Source Phrase'],
                "translation": row['Target Phrase'],
                "vibe": row['Vibe Note'],
                "language_pair": row['Language Pair']
            }
        return {"error": "No similar vibe found!"}
    
    def search_by_vibe(self, vibe_keyword):
        matches = self.phrases[self.phrases['Vibe Note'].str.lower().str.contains(vibe_keyword.lower(), na=False)]
        if not matches.empty:
            return matches[['Source Phrase', 'Target Phrase', 'Language Pair', 'Vibe Note']].to_dict('records')
        return {"error": "No vibes match that keyword!"}
    
    def add_phrase(self, source_phrase, target_phrase, lang_pair, vibe):
        if lang_pair not in ["Sheng-English", "Dholuo-English"]:
            return {"error": "Language pair must be 'Sheng-English' or 'Dholuo-English'"}
        new_row = pd.DataFrame({
            'Source Phrase': [source_phrase],
            'Target Phrase': [target_phrase],
            'Language Pair': [lang_pair],
            'Vibe Note': [vibe]
        })
        self.phrases = pd.concat([self.phrases, new_row], ignore_index=True)
        try:
            self.phrases.to_csv(self.csv_file, index=False)
            return {"success": "Phrase added!"}
        except Exception as e:
            return {"error": f"Failed to save phrase: {e}"}
    
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
        try:
            pd.DataFrame(data).to_json(output_file, orient='records', indent=4)
            return {"success": f"Exported to {output_file}!"}
        except Exception as e:
            return {"error": f"Failed to export training data: {e}"}
    
    def view_history(self):
        return self.history

def main():
    parser = argparse.ArgumentParser(description="Sheng-Dholuo Translator CLI")
    subparsers = parser.add_subparsers(dest="command")

    translate_parser = subparsers.add_parser("translate", help="Translate a phrase")
    translate_parser.add_argument("phrase", help="Phrase to translate")
    translate_parser.add_argument("--lang", help="Language pair (Sheng-English, Dholuo-English)")
    translate_parser.add_argument("--reverse", action="store_true", help="Reverse translation (English to Sheng/Dholuo)")
    translate_parser.add_argument("--ai", action="store_true", help="Use AI for translation")

    batch_parser = subparsers.add_parser("batch", help="Translate multiple phrases from a file")
    batch_parser.add_argument("file", help="File with phrases (one per line)")
    batch_parser.add_argument("--lang", help="Language pair (Sheng-English, Dholuo-English)")
    batch_parser.add_argument("--reverse", action="store_true", help="Reverse translation (English to Sheng/Dholuo)")
    batch_parser.add_argument("--ai", action="store_true", help="Use AI for translation")
    batch_parser.add_argument("--export", help="Export results to a file")

    subparsers.add_parser("history", help="View translation history")

    subparsers.add_parser("interactive", help="Run in interactive mode")

    args = parser.parse_args()

    translator = CulturalTranslator()

    if not args.command or args.command == "interactive":
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
            print("8. Train AI model")
            print("9. View translation history")
            print("10. Clear translation history")
            print("11. Quit")
            choice = input("What’s your move? (1-11): ").strip()
            
            # Validate choice
            if not choice.isdigit() or int(choice) < 1 or int(choice) > 11:
                print(f"{Fore.RED}Please enter a number between 1 and 11.{Style.RESET_ALL}")
                continue
            
            choice = int(choice)
            
            if choice == 11:
                print(f"{Fore.GREEN}Catch you later, bro! Stay noma!{Style.RESET_ALL}")
                break
            
            elif choice == 1:
                phrase = input("Enter a Sheng or Dholuo phrase: ").strip()
                if not phrase:
                    print(f"{Fore.RED}Phrase cannot be empty.{Style.RESET_ALL}")
                    continue
                lang = input("Filter by language? (Sheng-English, Dholuo-English, or leave blank): ").strip() or None
                if lang and lang not in ["Sheng-English", "Dholuo-English"]:
                    print(f"{Fore.RED}Language must be 'Sheng-English' or 'Dholuo-English'.{Style.RESET_ALL}")
                    continue
                use_ai = input("Use AI for translation? (y/n): ").strip().lower()
                if use_ai not in ['y', 'n']:
                    print(f"{Fore.RED}Please enter 'y' or 'n'.{Style.RESET_ALL}")
                    continue
                use_ai = use_ai == 'y'
                result = translator.translate(phrase, lang, reverse=False, use_ai=use_ai)
                
                if "translation" in result:
                    print(f"{Fore.GREEN}Translation: {result['translation']}{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}Vibe: {result['vibe']}{Style.RESET_ALL}")
                    print(f"Language: {result['language_pair']}")
                    vibe = result['vibe'].split(' (')[0]
                    rec = translator.recommend_similar_vibe(vibe)
                    if "source" in rec:
                        print(f"\n{Fore.CYAN}Try this similar vibe:{Style.RESET_ALL}")
                        print(f"- {rec['source']} → {rec['translation']} ({rec['vibe']})")
                elif "suggestions" in result:
                    print(f"{Fore.YELLOW}No exact match, but did you mean one of these?{Style.RESET_ALL}")
                    for sug in result['suggestions']:
                        print(f"- {sug['source']} → {sug['target']} (Similarity: {sug['score']}%)")
                else:
                    print(f"{Fore.RED}{result['error']}{Style.RESET_ALL}")
            
            elif choice == 2:
                vibe = input("Enter a vibe keyword (e.g., hype, calm): ").strip()
                if not vibe:
                    print(f"{Fore.RED}Vibe keyword cannot be empty.{Style.RESET_ALL}")
                    continue
                result = translator.search_by_vibe(vibe)
                if "error" in result:
                    print(f"{Fore.RED}{result['error']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Phrases with '{vibe}' vibes:{Style.RESET_ALL}")
                    for match in result:
                        print(f"- {match['Source Phrase']} → {match['Target Phrase']} ({match['Vibe Note']})")
            
            elif choice == 3:
                source = input("Enter the Sheng/Dholuo phrase: ").strip()
                if not source:
                    print(f"{Fore.RED}Source phrase cannot be empty.{Style.RESET_ALL}")
                    continue
                target = input("Enter the English translation: ").strip()
                if not target:
                    print(f"{Fore.RED}Target phrase cannot be empty.{Style.RESET_ALL}")
                    continue
                lang = input("Language pair (Sheng-English or Dholuo-English): ").strip()
                if lang not in ["Sheng-English", "Dholuo-English"]:
                    print(f"{Fore.RED}Language pair must be 'Sheng-English' or 'Dholuo-English'.{Style.RESET_ALL}")
                    continue
                vibe = input("What’s the vibe? (e.g., casual, hype): ").strip()
                if not vibe:
                    print(f"{Fore.RED}Vibe cannot be empty.{Style.RESET_ALL}")
                    continue
                result = translator.add_phrase(source, target, lang, vibe)
                if "success" in result:
                    print(f"{Fore.GREEN}{result['success']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}{result['error']}{Style.RESET_ALL}")
            
            elif choice == 4:
                phrase = input("Enter an English phrase: ").strip()
                if not phrase:
                    print(f"{Fore.RED}Phrase cannot be empty.{Style.RESET_ALL}")
                    continue
                lang = input("Target language? (Sheng-English, Dholuo-English, or leave blank): ").strip() or None
                if lang and lang not in ["Sheng-English", "Dholuo-English"]:
                    print(f"{Fore.RED}Language must be 'Sheng-English' or 'Dholuo-English'.{Style.RESET_ALL}")
                    continue
                use_ai = input("Use AI for translation? (y/n): ").strip().lower()
                if use_ai not in ['y', 'n']:
                    print(f"{Fore.RED}Please enter 'y' or 'n'.{Style.RESET_ALL}")
                    continue
                use_ai = use_ai == 'y'
                result = translator.translate(phrase, lang, reverse=True, use_ai=use_ai)
                
                if "translation" in result:
                    print(f"{Fore.GREEN}Translation: {result['translation']}{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}Vibe: {result['vibe']}{Style.RESET_ALL}")
                    print(f"Language: {result['language_pair']}")
                    vibe = result['vibe'].split(' (')[0]
                    rec = translator.recommend_similar_vibe(vibe)
                    if "source" in rec:
                        print(f"\n{Fore.CYAN}Try this similar vibe:{Style.RESET_ALL}")
                        print(f"- {rec['source']} → {rec['translation']} ({rec['vibe']})")
                elif "suggestions" in result:
                    print(f"{Fore.YELLOW}No exact match, but did you mean one of these?{Style.RESET_ALL}")
                    for sug in result['suggestions']:
                        print(f"- {sug['source']} → {sug['target']} (Similarity: {sug['score']}%)")
                else:
                    print(f"{Fore.RED}{result['error']}{Style.RESET_ALL}")
            
            elif choice == 5:
                result = translator.get_random_phrase()
                print(f"{Fore.CYAN}Random Phrase:{Style.RESET_ALL}")
                print(f"- {result['source']} → {result['translation']}")
                print(f"Vibe: {result['vibe']}")
                print(f"Language: {result['language_pair']}")
            
            elif choice == 6:
                stats = translator.get_stats()
                print(f"{Fore.CYAN}Translator Stats:{Style.RESET_ALL}")
                print(f"Total Phrases: {stats['total_phrases']}")
                print(f"Sheng Phrases: {stats['sheng_count']}")
                print(f"Dholuo Phrases: {stats['dholuo_count']}")
                print(f"Vibe Breakdown:")
                for vibe, count in stats['vibe_breakdown'].items():
                    print(f"- {vibe}: {count}")
            
            elif choice == 7:
                result = translator.export_training_data()
                if "success" in result:
                    print(f"{Fore.GREEN}{result['success']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}{result['error']}{Style.RESET_ALL}")
            
            elif choice == 8:
                result = translator.train_model()
                if "success" in result:
                    print(f"{Fore.GREEN}{result['success']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}{result['error']}{Style.RESET_ALL}")
            
            elif choice == 9:
                history = translator.view_history()
                if history:
                    print(f"{Fore.CYAN}Translation History:{Style.RESET_ALL}")
                    for entry in history:
                        print(f"\nSource: {entry['source']}")
                        if "translation" in entry['result']:
                            print(f"Translation: {entry['result']['translation']}")
                            print(f"Vibe: {entry['result']['vibe']}")
                            print(f"Language: {entry['result']['language_pair']}")
                        elif "suggestions" in entry['result']:
                            print("No exact match, but suggestions were:")
                            for sug in entry['result']['suggestions']:
                                print(f"- {sug['source']} → {sug['target']} (Similarity: {sug['score']}%)")
                        else:
                            print(f"Error: {entry['result']['error']}")
                else:
                    print(f"{Fore.RED}No translation history yet.{Style.RESET_ALL}")
            
            elif choice == 10:
                result = translator.clear_history()
                if "success" in result:
                    print(f"{Fore.GREEN}{result['success']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}{result['error']}{Style.RESET_ALL}")

    elif args.command == "translate":
        result = translator.translate(args.phrase, args.lang, args.reverse, args.ai)
        if "translation" in result:
            print(f"Translation: {result['translation']}")
            print(f"Vibe: {result['vibe']}")
            print(f"Language: {result['language_pair']}")
        elif "suggestions" in result:
            print("No exact match, but did you mean one of these?")
            for sug in result['suggestions']:
                print(f"- {sug['source']} → {sug['target']} (Similarity: {sug['score']}%)")
        else:
            print(result['error'])

    elif args.command == "batch":
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                phrases = [line.strip() for line in f if line.strip()]
            results = translator.batch_translate(phrases, args.lang, args.reverse, args.ai, export_file=args.export)
            for res in results:
                print(f"\nPhrase: {res['phrase']}")
                if "translation" in res['result']:
                    print(f"Translation: {res['result']['translation']}")
                    print(f"Vibe: {res['result']['vibe']}")
                    print(f"Language: {res['result']['language_pair']}")
                elif "suggestions" in res['result']:
                    print("No exact match, but did you mean one of these?")
                    for sug in res['result']['suggestions']:
                        print(f"- {sug['source']} → {sug['target']} (Similarity: {sug['score']}%)")
                else:
                    print(res['result']['error'])
        except Exception as e:
            print(f"{Fore.RED}Error reading batch file: {e}{Style.RESET_ALL}")

    elif args.command == "history":
        history = translator.view_history()
        if history:
            print("Translation History:")
            for entry in history:
                print(f"\nSource: {entry['source']}")
                if "translation" in entry['result']:
                    print(f"Translation: {entry['result']['translation']}")
                    print(f"Vibe: {entry['result']['vibe']}")
                    print(f"Language: {entry['result']['language_pair']}")
                elif "suggestions" in entry['result']:
                    print("No exact match, but suggestions were:")
                    for sug in entry['result']['suggestions']:
                        print(f"- {sug['source']} → {sug['target']} (Similarity: {sug['score']}%)")
                else:
                    print(f"Error: {entry['result']['error']}")
        else:
            print("No translation history yet.")

if __name__ == "__main__":
    main()
