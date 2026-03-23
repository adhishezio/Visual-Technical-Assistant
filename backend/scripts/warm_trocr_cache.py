from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_NAME = "microsoft/trocr-large-printed"


def main() -> None:
    TrOCRProcessor.from_pretrained(MODEL_NAME)
    VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)


if __name__ == "__main__":
    main()
