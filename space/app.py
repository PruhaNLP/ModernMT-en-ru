import gradio as gr
import torch
from transformers import AutoModel, AutoTokenizer

# ======Settings=========
MODEL_NAME = "PruhaNLP/ModernMT-en-ru-EXP"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
model.eval()
print("Model loaded!")


def translate(text: str, num_beams: int = 4, max_length: int = 256) -> str:
    if not text.strip():
        return ""
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="English", placeholder="Enter english text to translate...", lines=4),
        gr.Slider(1, 8, value=4, step=1, label="Beam size"),
        gr.Slider(32, 512, value=256, step=32, label="Max length"),
    ],
    outputs=gr.Textbox(label="Russian", lines=4),
    title="ModernMT EN→RU",
    description="English to Russian translation",
    examples=[
        ["I just want to drink vanilla Coke every night and play Clash Royale.", 4, 256],
        ["The vortex structure in the Bose-Einstein condensate, described by the nonlinear Gross-Pitaevsky equation, demonstrates quantum interference and topological stability.", 4, 256],
        ["Machine translation is a subfield of computational linguistics.", 4, 256],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
