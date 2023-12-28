from googletrans import Translator

class LanguageTranslator:
    def __init__(self):
        self.translator = Translator()
        
    def translate_text(self, text, target_language='en'):
        try:
            translation = self.translator.translate(text, dest=target_language)
            if translation and translation.text is not None:
                return translation.text
            else:
                print("Translation result is None.")
                return ""
        except Exception as e:
            print(f"Translation failed: {e}")
            return ""
        
