"""
Generation by NGrams starter
"""

# pylint:disable=too-many-locals, unused-import, unused-variable

from main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open(
        "./lab_3_generate_by_ngrams/assets/Harry_Potter.txt", "r", encoding="utf-8"
    ) as text_file:
        text = text_file.read()

    processor = TextProcessor("_")
    encoded = processor.encode(text)
    assert encoded, "Text can not be encoded"

    print(encoded[:200])

    decoded = processor.decode(encoded)
    assert decoded, "Text can not be decoded"

    print("Начало декодированного текста")
    print(decoded[:200])
    print("Конец декодированного текста")
    print(decoded[-200:])

    model = NGramLanguageModel(encoded, 7)
    built = model.build()
    print(f"Результат метода: {built}")
    print(model._n_gram_frequencies)

    generator = GreedyTextGenerator(model, processor)
    generated_text = generator.run("Vernon", 51)
    print(f"Сгенерированный текст:\n{generated_text}")
    assert generated_text

if __name__ == "__main__":
    main()
