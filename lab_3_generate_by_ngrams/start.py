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

    text_processor = TextProcessor("_")
    encoded_corpus = text_processor.encode(text)

    n_gram_size = 7
    language_model = NGramLanguageModel(encoded_corpus, n_gram_size)
    language_model.build()

    greedy_text_generator = GreedyTextGenerator(language_model, text_processor)

    prompt = "Vernon"
    seq_len = 51
    generated_text = greedy_text_generator.run(prompt, seq_len)

    assert generated_text

if __name__ == "__main__":
    main()
