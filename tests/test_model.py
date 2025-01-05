
import unittest
import torch
from unsloth import FastLanguageModel

class TestFineTunedModel(unittest.TestCase):
    def setUp(self):
        """Set up the test environment by loading the fine-tuned model and tokenizer."""
        self.model_path = "finetuned_model"  # Path to the fine-tuned model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=2048,
            load_in_4bit=True
        )
        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_model_loading(self):
        """Test if the model is loaded successfully and is in evaluation mode."""
        self.assertIsNotNone(self.model, "Model failed to load.")
        self.assertEqual(self.model.training, False, "Model is not in evaluation mode.")

    def test_inference_device(self):
        """Test if the model is correctly moved to the appropriate device."""
        self.assertEqual(str(self.model.device), str(self.device), "Model is not on the correct device.")

    def test_text_generation(self):
        """Test if the model generates text for a given prompt."""
        prompt = "Once upon a time"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertTrue(len(generated_text) > 0, "Model failed to generate text.")

    def test_inference_speed(self):
        """Test the inference speed for generating text."""
        import time
        prompt = "Quick test of inference speed."
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9
            )
        duration = time.time() - start_time
        self.assertLess(duration, 5, "Inference took too long (more than 5 seconds).")

if __name__ == "__main__":
    unittest.main()
