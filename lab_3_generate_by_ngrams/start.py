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
    decoded = processor.decode(encoded)

    print("Decoded text start:")
    print(decoded[:200])
    print("Decoded text end:")
    print(decoded[-200:])
    assert decoded

    n_gram_size = 7
    model = NGramLanguageModel(encoded, n_gram_size)
    built = model.build()
    print(f"Build result: {built}")

    generator = GreedyTextGenerator(model, processor)

    prompt = "Vernon"
    seq_len = 51
    generated_text = generator.run(prompt, seq_len)
    print(f"Generated text:\n{generated_text}")

    assert generated_text

if __name__ == "__main__":
    main()
