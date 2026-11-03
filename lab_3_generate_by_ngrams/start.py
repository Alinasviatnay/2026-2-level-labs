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
    decoded = processor.decode (encoded)

    print("The start of decoded text")
    print(decoded[:200])
    print("The end of decoded text")
    print(decoded[-200:])
    assert decoded

    model = NGramLanguageModel(encoded, 7)
    built = model.build()
    print(f"The result of method: {built}")
    assert built

    generator = GreedyTextGenerator(model, processor)
    generated_text = generator.run("Vernon", 51)
    print(f"The generated text:\n{generated_text}")
    assert generated_text

if __name__ == "__main__":
    main()
