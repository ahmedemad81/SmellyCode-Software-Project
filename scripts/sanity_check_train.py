import torch

from src.models.model_loader import get_model, get_tokenizer
from src.constants import ID_TO_LABEL, MAX_LENGTH
from src.prompts.builders import build_classifier_input

MODEL_NAME = "Qwen/Qwen3-0.6B"

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer(MODEL_NAME)

    print("Loading model...")
    model = get_model(MODEL_NAME, use_4bit=True)
    model.eval()

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

    print("\nTokenizing...")
    text = build_classifier_input(sample_code)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(model.device)

    print("Running classification...")
    with torch.inference_mode():
        outputs = model(**inputs)
        pred_id = int(torch.argmax(outputs.logits, dim=-1).item())

    print("\nSanity check passed.")
    print("Predicted label:", ID_TO_LABEL[pred_id])


if __name__ == "__main__":
    main()