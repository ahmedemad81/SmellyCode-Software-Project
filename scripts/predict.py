import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

from src.models.model_loader import get_tokenizer
from src.constants import LABELS, LABEL_TO_ID, ID_TO_LABEL, MAX_LENGTH, MODEL_NAME, MODEL_OUTPUT_DIR
from src.prompts.builders import build_classifier_input


def load_finetuned_model(base_model_name: str, adapter_dir: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model


@torch.inference_mode()
def predict_code_smell(model, tokenizer, code: str, max_length: int = MAX_LENGTH) -> str:
    text = build_classifier_input(code)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    outputs = model(**inputs)
    pred_id = int(torch.argmax(outputs.logits, dim=-1).item())
    return ID_TO_LABEL[pred_id]


def main():
    tokenizer = get_tokenizer(MODEL_NAME)
    model = load_finetuned_model(MODEL_NAME, MODEL_OUTPUT_DIR)

    sample_code = """
public class Customer {
    private String name;
    private String email;
    private String phone;

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }

    public String getPhone() { return phone; }
    public void setPhone(String phone) { this.phone = phone; }
}
"""

    prediction = predict_code_smell(model, tokenizer, sample_code)
    print("Predicted label:", prediction)


if __name__ == "__main__":
    main()